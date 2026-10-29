"""Piper ONNX voices as PyTorch VITS models, so they run batched on the GPU like the English generators.

Piper publishes most voices only as ONNX (an inference graph). Its weights are the VITS model's, under
their PyTorch names except where the export renamed or folded them:
- the conv weights of the flow's WaveNet layers are anonymous (onnx::Conv_N): the Conv node's name gives
  the layer (/flow/flows.6/enc/in_layers.0/Conv is flow.flows.6.enc.in_layers.0);
- the phoneme embedding is stored as "sid" when the voice has one speaker (the Gather of /enc_p/emb);
- the duration flow's first affine layer keeps m but stores exp(-logs), as it runs in reverse.
Weight norm is already folded into the weights, so the model is built and stripped of it before loading.

The hyperparameters are read from the weight shapes; the dilations are Piper's (they leave no trace in
the weights). Reading ONNX is protobuf parsing: no code from the file runs. load() checks that every
weight the generator needs was found.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch

_PARAM = re.compile(r"^(enc_p|dec|flow|dp|emb_g)\.")
# what the export leaves out on purpose: the posterior encoder, the duration flows used only in training,
# and dp.flows.1, which VITS skips when it runs the duration flow in reverse
_UNUSED = ("enc_q.", "dp.post_", "dp.flows.1.")


def _arrays(path: Path) -> dict[str, np.ndarray]:
    import onnx
    from onnx import numpy_helper

    graph = onnx.load(str(path)).graph
    init = {t.name: t for t in graph.initializer}
    out = {k: numpy_helper.to_array(t) for k, t in init.items() if _PARAM.match(k)}
    for node in graph.node:
        name = node.name.lstrip("/")
        if node.op_type in ("Conv", "ConvTranspose"):
            layer = name.rsplit("/", 1)[0].replace("/", ".")
            for i, part in ((1, "weight"), (2, "bias")):
                if len(node.input) > i and node.input[i] in init:
                    out.setdefault(f"{layer}.{part}", numpy_helper.to_array(init[node.input[i]]))
        elif name == "enc_p/emb/Gather" and node.input[0] in init:
            out["enc_p.emb.weight"] = numpy_helper.to_array(init[node.input[0]])
        elif name == "dp/flows.0/Mul" and node.input[1] in init:
            out["dp.flows.0.logs"] = -np.log(numpy_helper.to_array(init[node.input[1]]))
    return out


def _hparams(w: dict[str, np.ndarray]) -> dict:
    n_vocab, hidden = w["enc_p.emb.weight"].shape
    n_layers = len({k.split(".")[3] for k in w if k.startswith("enc_p.encoder.attn_layers.")})
    ups = sorted(int(k.split(".")[2]) for k in w if re.match(r"dec\.ups\.\d+\.weight$", k))
    kernels = [w[f"dec.ups.{i}.weight"].shape[2] for i in ups]
    n_res = len({k.split(".")[2] for k in w if k.startswith("dec.resblocks.")})
    per = n_res // len(ups)
    res1 = any(k.startswith("dec.resblocks.0.convs1.") for k in w)
    res_k = [
        w[f"dec.resblocks.{j}.{'convs1' if res1 else 'convs'}.0.weight"].shape[2]
        for j in range(per)
    ]
    emb_g = w.get("emb_g.weight")
    return {
        "n_vocab": int(n_vocab),
        "spec_channels": 513,  # the posterior encoder's input; not used for synthesis
        "segment_size": 32,
        "inter_channels": int(w["enc_p.proj.weight"].shape[0] // 2),
        "hidden_channels": int(hidden),
        "filter_channels": int(w["enc_p.encoder.ffn_layers.0.conv_1.weight"].shape[0]),
        "n_heads": int(hidden // w["enc_p.encoder.attn_layers.0.emb_rel_k"].shape[2]),
        "n_layers": n_layers,
        "kernel_size": int(w["enc_p.encoder.ffn_layers.0.conv_1.weight"].shape[2]),
        "p_dropout": 0.0,
        "resblock": "1" if res1 else "2",
        "resblock_kernel_sizes": tuple(int(k) for k in res_k),
        "resblock_dilation_sizes": ((1, 3, 5),) * per if res1 else ((1, 2), (2, 6), (3, 12))[:per],
        "upsample_rates": tuple(int(k) // 2 for k in kernels),
        "upsample_initial_channel": int(w["dec.conv_pre.weight"].shape[0]),
        "upsample_kernel_sizes": tuple(int(k) for k in kernels),
        "n_speakers": int(emb_g.shape[0]) if emb_g is not None else 1,
        "gin_channels": int(emb_g.shape[1]) if emb_g is not None else 0,
        "use_sdp": any(k.startswith("dp.flows.") for k in w),
    }


def _strip_weight_norm(model: torch.nn.Module) -> None:
    for mod in model.modules():
        if hasattr(mod, "weight_g"):
            torch.nn.utils.remove_weight_norm(mod)


def load(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """The voice at path (.onnx, with its .onnx.json) as an eval-mode SynthesizerTrn, and its config."""
    from .vits.models import SynthesizerTrn

    w = _arrays(Path(path))
    model = SynthesizerTrn(**_hparams(w))
    _strip_weight_norm(model)
    want = {k for k in model.state_dict() if not k.startswith(_UNUSED)}
    missing = sorted(want - w.keys())
    if missing:
        raise RuntimeError(f"{Path(path).name}: weights not found in the ONNX graph: {missing[:5]}")
    model.load_state_dict({k: torch.from_numpy(np.array(w[k])) for k in want}, strict=False)
    model.eval().to(device)
    config = json.loads(Path(f"{path}.json").read_text())
    return model, config
