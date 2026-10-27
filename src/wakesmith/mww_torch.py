"""microWakeWord training in PyTorch, on a GPU (Intel XPU, NVIDIA CUDA) or the CPU; export stays microWakeWord's.

Reads the project's mww/training.yaml and features and reproduces microWakeWord's recipe:
- batches: a feature set picked by sampling_weight, a clip at random, cut or zero-padded to the input length by the
  set's truncation strategy. Built in worker processes, one gather per shard (microWakeWord: 128 Python picks);
- the same MixedNet (wakesmith's MODEL_ARGS), Keras' initialisation, Adam defaults and BatchNorm settings;
- the loss as microWakeWord computes it: it passes penalty (B,) x class weight (B,1), a (B,B) matrix, so Keras
  takes mean(loss x penalty) x mean(class weight): class weights scale the whole batch, not each sample;
- validation every eval_step_interval: the validation set (truncate_start) and the ambient set (windows every
  stride x 10 ms), recall and false alarms per hour at 101 cutoffs, "average viable recall"; the best step's
  weights are kept (microWakeWord's rule when minimization_metric is null).
The best weights are copied into microWakeWord's Keras MixedNet (identical scores) and saved as
<train_dir>/best_weights.weights.h5; microWakeWord's own `--train 0` run then builds the streaming int8 TFLite.

On a Ryzen 5 7600X + Arc Pro B70, per 1000 steps: microWakeWord's trainer ~45 s; this one ~7 s on the GPU,
~16 s on the CPU. Validation is 0.4 s on the GPU instead of ~11 s.

usage (from a project's mww/ folder): python -m wakesmith.mww_torch --device xpu:0 [--workers N] [--seed S]
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch._dynamo  # must load before TensorFlow does: imported after it, it segfaults
from torch import nn

from .mww import MODEL_ARGS, more_open_files

UINT16_SCALE = 0.0390625  # microWakeWord rescales uint16 feature sets by this
CUTOFFS = np.linspace(0.0, 1.0, 101)


def model_flags(training_yaml: str = "training.yaml") -> argparse.Namespace:
    """MODEL_ARGS as microWakeWord's mixednet flags."""
    a = dict(zip(MODEL_ARGS[1::2], MODEL_ARGS[2::2]))
    return argparse.Namespace(
        training_config=training_yaml,
        pointwise_filters=a["--pointwise_filters"],
        repeat_in_block=a["--repeat_in_block"],
        mixconv_kernel_sizes=a["--mixconv_kernel_sizes"],
        residual_connection=a["--residual_connection"],
        first_conv_filters=int(a["--first_conv_filters"]),
        first_conv_kernel_size=int(a["--first_conv_kernel_size"]),
        stride=int(a["--stride"]),
        spatial_attention=0,
        pooled=0,
        max_pool=0,
    )


class MixedNet(nn.Module):
    """microWakeWord's MixedNet in its training (non-streaming) form. Keras' (B, T, 1, C) is Conv1d's (B, C, T)."""

    def __init__(self, n_features: int, length: int, flags: argparse.Namespace | None = None):
        super().__init__()
        from microwakeword.mixednet import parse

        f = flags or model_flags()
        if any(parse(f.residual_connection)) or any(r != 1 for r in parse(f.repeat_in_block)):
            raise ValueError("residual connections and repeated blocks are not ported")
        self.first = nn.Conv1d(
            n_features, f.first_conv_filters, f.first_conv_kernel_size, stride=f.stride, bias=False
        )
        ch = f.first_conv_filters
        self.blocks = nn.ModuleList()
        self.splits = []
        for filters, ks in zip(parse(f.pointwise_filters), parse(f.mixconv_kernel_sizes)):
            splits = [ch // len(ks)] * len(ks)
            splits[0] += ch - sum(splits)
            self.splits.append(splits)
            block = nn.ModuleDict(
                {
                    "dw": nn.ModuleList(nn.Conv1d(c, c, k, groups=c) for c, k in zip(splits, ks)),
                    "pw": nn.Conv1d(ch, filters, 1, bias=False),
                    "bn": nn.BatchNorm1d(filters, eps=1e-3, momentum=0.01),  # Keras: momentum 0.99
                }
            )
            self.blocks.append(block)
            ch = filters
        with torch.no_grad():
            t = self._features(torch.zeros(1, n_features, length)).shape
        self.dense = nn.Linear(t[1] * t[2], 1)
        for mod in self.modules():  # Keras defaults: glorot-uniform kernels, zero biases
            if isinstance(mod, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(mod.weight)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)

    def _features(self, x):
        x = torch.relu(self.first(x))
        for b, splits in zip(self.blocks, self.splits):
            outs = [conv(part) for conv, part in zip(b["dw"], torch.split(x, splits, dim=1))]
            t = min(o.shape[-1] for o in outs)
            x = torch.cat([o[..., -t:] for o in outs], dim=1)  # StridedDrop: drop from the front
            x = torch.relu(b["bn"](b["pw"](x)))
        return x

    def forward(self, x):  # x: (B, T, F) as the features come; returns logits (B, 1)
        h = self._features(x.transpose(1, 2))
        return self.dense(h.transpose(1, 2).flatten(1))  # Keras' Flatten order: time, then channel


def copy_into_keras(model: MixedNet, keras_model) -> None:
    """PyTorch MixedNet -> microWakeWord's Keras MixedNet, layer by layer in Keras' order."""
    layers = iter(layer for layer in keras_model.layers if layer.get_weights())

    def put(arrays):
        layer = next(layers)
        want = [w.shape for w in layer.get_weights()]
        if want != [a.shape for a in arrays]:
            raise ValueError(f"{layer.name}: {want} != {[a.shape for a in arrays]}")
        layer.set_weights(arrays)

    def t(p):
        return p.detach().cpu().numpy()

    put([t(model.first.weight).transpose(2, 1, 0)[:, None]])  # (out, in, k) -> (k, 1, in, out)
    for b in model.blocks:
        for conv in b["dw"]:  # depthwise (c, 1, k) -> (k, 1, c, 1), bias (c,)
            put([t(conv.weight)[:, 0, :].T[:, None, :, None], t(conv.bias)])
        put([t(b["pw"].weight)[:, :, 0].T[None, None]])  # 1x1 (out, in, 1) -> (1, 1, in, out)
        bn = b["bn"]
        put([t(bn.weight), t(bn.bias), t(bn.running_mean), t(bn.running_var)])
    put([t(model.dense.weight).T, t(model.dense.bias)])
    if next(layers, None) is not None:
        raise ValueError("the Keras model has more weighted layers than the PyTorch one")


# -- training data -------------------------------------------------------------------------------


class _Shard:
    def __init__(self, d: Path):
        from mmap_ninja.ragged import RaggedMmap

        r = RaggedMmap(str(d))
        self.data = r.memmap.reshape(-1, 40)  # every clip's frames in a row
        self.first_frame = np.asarray(r.starts, np.int64) // 40
        self.frames = (np.asarray(r.ends, np.int64) - np.asarray(r.starts, np.int64)) // 40
        self.scale = UINT16_SCALE if self.data.dtype == np.uint16 else 1.0


class FeatureSet:
    """One `features` entry of training.yaml, its training split."""

    def __init__(self, fs: dict, mode: str = "training"):
        self.label = float(fs["truth"])
        self.weight = float(fs["sampling_weight"])
        self.penalty = float(fs["penalty_weight"])
        self.trunc = fs["truncation_strategy"]
        if self.trunc not in ("random", "truncate_start", "truncate_end", "split"):
            raise ValueError(f"truncation strategy {self.trunc} is not supported")
        dirs = sorted(Path(fs["features_dir"], mode).resolve().glob("**/*_mmap/"))
        self.shards = [_Shard(d) for d in dirs]
        self.ends = np.cumsum([len(s.frames) for s in self.shards], dtype=np.int64)
        self.n = int(self.ends[-1]) if len(self.ends) else 0

    def sample(self, k: int, length: int, rng: np.random.Generator) -> np.ndarray:
        """k random clips as fixed_length_spectrogram() cuts them: zeros in front when short."""
        out = np.zeros((k, length, 40), np.float32)
        which = rng.integers(0, self.n, k)
        shard_of = np.searchsorted(self.ends, which, side="right")
        for s_i in np.unique(shard_of):
            rows = np.flatnonzero(shard_of == s_i)
            sh = self.shards[s_i]
            idx = which[rows] - (self.ends[s_i - 1] if s_i else 0)
            n = sh.frames[idx]
            if self.trunc == "random":  # np.random.randint(0, n - length)
                off = np.where(
                    n > length,
                    (rng.random(len(rows)) * np.maximum(n - length, 1)).astype(np.int64),
                    0,
                )
            elif self.trunc == "truncate_end":
                off = np.zeros_like(n)
            else:
                off = np.maximum(n - length, 0)
            t = np.arange(length)[None, :] - np.maximum(length - n, 0)[:, None]  # < 0: padding
            frame = sh.first_frame[idx][:, None] + off[:, None] + np.maximum(t, 0)
            vals = sh.data[frame.reshape(-1)].reshape(len(rows), length, 40)
            out[rows] = np.where((t >= 0)[:, :, None], vals.astype(np.float32) * sh.scale, 0.0)
        return out


class Sampler:
    def __init__(self, config: dict, seed: int):
        self.sets = [s for s in (FeatureSet(fs) for fs in config["features"]) if s.n]
        w = np.array([s.weight for s in self.sets])
        self.p = w / w.sum()
        self.length = config["spectrogram_length"]
        self.rng = np.random.default_rng(seed)

    def batch(self, size: int):
        which = self.rng.choice(len(self.sets), size, p=self.p)
        x = np.empty((size, self.length, 40), np.float32)
        y = np.empty(size, np.float32)
        w = np.empty(size, np.float32)
        for i, s in enumerate(self.sets):
            rows = np.flatnonzero(which == i)
            if len(rows):
                x[rows] = s.sample(len(rows), self.length, self.rng)
                y[rows], w[rows] = s.label, s.penalty
        return x, y, w


class _Batches(torch.utils.data.IterableDataset):
    def __init__(self, config: dict, size: int, seed: int):
        self.config, self.size, self.seed = config, size, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        s = Sampler(self.config, self.seed * 1000 + (info.id if info else 0))
        while True:
            yield s.batch(self.size)


def batch_stream(config: dict, size: int, workers: int, seed: int):
    """Endless training batches from `workers` processes (forked: they share the memory-mapped features)."""
    return iter(
        torch.utils.data.DataLoader(
            _Batches(config, size, seed),
            batch_size=None,
            num_workers=workers,
            prefetch_factor=4,
            persistent_workers=True,
            multiprocessing_context="fork",
        )
    )


# -- validation ----------------------------------------------------------------------------------


def validation_sets(config: dict, dev: torch.device) -> dict:
    """microWakeWord's own validation and ambient batches, built once and kept on the device."""
    from microwakeword.data import FeatureHandler

    h = FeatureHandler(config)
    n = config["spectrogram_length"]
    vx, vy, _ = h.get_data(
        "validation",
        batch_size=config["batch_size"],
        features_length=n,
        truncation_strategy="truncate_start",
    )
    out = {
        "vx": torch.from_numpy(np.asarray(vx, np.float32)).to(dev),
        "vy": np.asarray(vy, np.float32),
    }
    if h.get_mode_size("validation_ambient") > 0:
        ax, _, _ = h.get_data(
            "validation_ambient",
            batch_size=config["batch_size"],
            features_length=n,
            truncation_strategy="split",
        )
        out["ax"] = torch.from_numpy(np.asarray(ax, np.float32)).to(dev)
        out["hours"] = h.get_mode_duration("validation_ambient") / 3600.0
    return out


@torch.no_grad()
def _scores(model, x, chunk: int = 8192) -> np.ndarray:
    model.eval()
    out = torch.cat([torch.sigmoid(model(x[i : i + chunk]))[:, 0] for i in range(0, len(x), chunk)])
    model.train()
    return out.float().cpu().numpy()


def metrics(p: np.ndarray, y: np.ndarray, ambient: np.ndarray | None, hours: float) -> dict:
    """validate_nonstreaming() from microWakeWord's train.py, on scores (Keras counts score > cutoff)."""
    above = p[None, :] > CUTOFFS[:, None]
    pos = y > 0.5
    tp, fp, fn = (above & pos).sum(1), (above & ~pos).sum(1), (~above & pos).sum(1)
    m = {
        "accuracy": float(((p > 0.5) == pos).mean()),
        "recall": tp[50] / max(tp[50] + fn[50], 1),
        "precision": tp[50] / max(tp[50] + fp[50], 1),
        "recall_at_no_faph": 0.0,
        "cutoff_for_no_faph": 0.0,
        "ambient_false_positives": 0,
        "ambient_false_positives_per_hour": 0.0,
        "average_viable_recall": 0.0,
    }
    if ambient is None:
        return m
    afp = (ambient[None, :] > CUTOFFS[:, None]).sum(1)
    recall = tp / np.maximum(tp + fn, 1)
    faph = afp / hours
    cutoff, r0 = 1.0, float(recall[-1])
    for i, c in enumerate(CUTOFFS):
        if faph[i] == 0:
            cutoff, r0 = float(c), float(recall[i])
            break
    if faph[0] > 2:  # recall at 2 false alarms per hour, by linear interpolation
        j = 1
        while faph[j] > 2:
            j += 1
        x0, y0, x1, y1 = faph[j - 1], recall[j - 1], faph[j], recall[j]
        r2 = (y0 * (x1 - 2.0) + y1 * (2.0 - x0)) / (x1 - x0)
    else:
        j, r2 = 0, recall[0]
    xs, ys = [2.0], [r2]
    for i in range(j, len(recall)):
        if faph[i] != xs[-1]:
            xs.append(faph[i])
            ys.append(recall[i])
    m.update(
        recall_at_no_faph=r0,
        cutoff_for_no_faph=cutoff,
        ambient_false_positives=int(afp[50]),
        ambient_false_positives_per_hour=float(faph[50]),
        average_viable_recall=float(np.trapezoid(np.flip(ys), np.flip(xs)) / 2.0),
    )
    return m


def validate(model, v: dict) -> dict:
    amb = _scores(model, v["ax"]) if "ax" in v else None
    return metrics(_scores(model, v["vx"]), v["vy"], amb, v.get("hours", 1.0))


# -- training ------------------------------------------------------------------------------------


def train(config: dict, dev: torch.device, workers: int, seed: int, log=print) -> MixedNet:
    torch.manual_seed(seed)
    stream = batch_stream(
        config, config["batch_size"], workers, seed
    )  # fork before the big tensors exist
    v = validation_sets(config, dev)
    model = MixedNet(40, config["spectrogram_length"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-7)  # Keras' Adam defaults
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    steps = config["training_steps"]
    total, every = sum(steps), config["eval_step_interval"]
    phase = np.repeat(np.arange(len(steps)), steps)
    log(
        f"PyTorch on {dev}, {workers} batch workers, seed {seed}; validation {len(v['vy'])} clips, "
        f"ambient {len(v.get('ax', []))} windows ({v.get('hours', 0):.2f} h)"
    )
    best, best_step, best_state = -1.0, 0, None
    key = config["maximization_metric"]
    t0 = time.perf_counter()
    for step in range(1, total + 1):
        ph = phase[step - 1]
        for group in opt.param_groups:
            group["lr"] = config["learning_rates"][ph]
        x, y, w = next(stream)
        cw = torch.where(
            y > 0.5, config["positive_class_weight"][ph], config["negative_class_weight"][ph]
        ).mean()
        x, y, w = (a.to(dev, non_blocking=True) for a in (x, y, w))
        opt.zero_grad(set_to_none=True)
        loss = (loss_fn(model(x)[:, 0], y) * w).mean() * cw.item()
        loss.backward()
        opt.step()
        if (
            step % 50 == 0
        ):  # the line wakesmith's progress bar reads (microWakeWord prints it every step)
            print(
                f"Validation Batch #{step // every + 1:d}: Loss = {loss.item():.4f}; Mini-Batch #{step % every:d}",
                end="\r",
                flush=True,
            )
        if step % every == 0 or step == total:
            m = validate(model, v)
            log(
                f"Step {step} (nonstreaming): Validation: recall at no faph = {m['recall_at_no_faph'] * 100:.3f} "
                f"with cutoff {m['cutoff_for_no_faph']:.2f}, accuracy = {m['accuracy'] * 100:.2f}%, recall = "
                f"{m['recall'] * 100:.2f}%, precision = {m['precision'] * 100:.2f}%, ambient false positives = "
                f"{m['ambient_false_positives']}, estimated false positives per hour = "
                f"{m['ambient_false_positives_per_hour']:.5f}, average viable recall = "
                f"{m['average_viable_recall']:.9f} ({time.perf_counter() - t0:.0f} s)"
            )
            if m[key] > best:
                best, best_step, best_state = m[key], step, copy.deepcopy(model.state_dict())
    log(
        f"best: step {best_step}, {key} = {best:.4f}; {total} steps in {time.perf_counter() - t0:.0f} s"
    )
    model.load_state_dict(best_state)
    return model.cpu().eval()


def main() -> None:
    more_open_files()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--threads", type=int, default=0, help="CPU threads for PyTorch (0: its default)"
    )
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    import tensorflow as tf

    tf.config.set_visible_devices(
        [], "GPU"
    )  # TensorFlow only builds the Keras copy: keep it off the card
    from microwakeword import mixednet
    from microwakeword.model_train_eval import load_config

    flags = model_flags()
    config = load_config(flags, mixednet)
    seed = a.seed if a.seed is not None else random.SystemRandom().randrange(2**31)
    model = train(
        config, torch.device(a.device), a.workers, seed, log=lambda s: print(s, flush=True)
    )
    km = mixednet.model(flags, config["training_input_shape"], config["batch_size"])
    copy_into_keras(model, km)
    out = Path(config["train_dir"])
    out.mkdir(parents=True, exist_ok=True)
    km.save_weights(out / "best_weights.weights.h5")
    km.save_weights(
        out / "last_weights.weights.h5"
    )  # the best weights: the last step's are not kept
    sys.stdout.flush()
    os._exit(0)  # don't wait on the batch workers' shutdown


if __name__ == "__main__":
    main()
