"""Everything wakesmith downloads: where from, checksum, size, license. Nothing here is bundled.

Assets live under the data home (WAKESMITH_HOME, default ~/.local/share/wakesmith) and are shared by
all projects. `profile` says which data profile uses an asset: "personal" assets carry
non-commercial terms, so models trained with them are for personal use only.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

HF = "https://huggingface.co"
PSG = "https://github.com/rhasspy/piper-sample-generator"
PSG_COMMIT = "2971426"  # model configs (.pt.json) at this commit


def home() -> Path:
    return Path(os.environ.get("WAKESMITH_HOME", Path.home() / ".local/share/wakesmith"))


@dataclass(frozen=True)
class Asset:
    name: str
    path: str  # relative to home()
    url: str = ""  # single file (sha256-verified); empty for HF folder snapshots
    sha256: str = ""
    size: int = 0  # bytes, as downloaded
    unzip: bool = False  # extract the downloaded .zip into `path`'s parent and delete it
    hf_repo: str = ""  # HF dataset snapshot instead of a single file
    hf_patterns: tuple[str, ...] = ()
    license: str = ""
    source: str = ""
    profile: str = "both"  # both | personal
    used_for: str = ""

    @property
    def local(self) -> Path:
        return home() / self.path

    def present(self) -> bool:
        p = self.local
        if self.hf_repo or self.unzip:
            return (p / ".complete").exists() or (
                p.is_dir() and any(p.iterdir()) and p.is_symlink()
            )
        return p.exists()


ASSETS: dict[str, Asset] = {
    a.name: a
    for a in [
        Asset(
            "tts-libritts-high",
            "tts/en-us-libritts-high.pt",
            url=f"{PSG}/releases/download/v1.0.0/en-us-libritts-high.pt",
            sha256="5258f21adf77c29248d9a89d51ca9492f6e75a498140584dcbed0edcab81960a",
            size=255_226_835,
            license="MIT (rhasspy/piper-checkpoints); trained on LibriTTS, CC BY 4.0",
            source=PSG,
            used_for="synthetic speech (904 English speakers)",
        ),
        Asset(
            "tts-libritts-high-config",
            "tts/en-us-libritts-high.pt.json",
            url=f"https://raw.githubusercontent.com/rhasspy/piper-sample-generator/{PSG_COMMIT}/models/en-us-libritts-high.pt.json",
            sha256="91e9f6f51ad1dc3e7ee57bbdfaaa549c0cb9e88a1c3aeed224e4469ddd745add",
            size=21_301,
            license="MIT",
            source=PSG,
            used_for="synthetic speech",
        ),
        Asset(
            "tts-libritts-r-medium",
            "tts/en_US-libritts_r-medium.pt",
            url=f"{PSG}/releases/download/v1.0.0/en_US-libritts_r-medium.pt",
            sha256="e95ee53770bf598c354a6e6dbfc95ccb259aeeb501d35a86be8a767429ab0ff6",
            size=204_089_915,
            license="MIT (rhasspy/piper-checkpoints); trained on LibriTTS-R, CC BY 4.0",
            source=PSG,
            used_for="synthetic speech (904 English speakers)",
        ),
        Asset(
            "tts-libritts-r-medium-config",
            "tts/en_US-libritts_r-medium.pt.json",
            url=f"https://raw.githubusercontent.com/rhasspy/piper-sample-generator/{PSG_COMMIT}/models/en_US-libritts_r-medium.pt.json",
            sha256="119118e510d0b8a7a0c8649a0668640d6db9b00239e874169d964853a8d15848",
            size=25_782,
            license="MIT",
            source=PSG,
            used_for="synthetic speech",
        ),
        Asset(
            "musan-noise",
            "noise/musan",
            hf_repo="FluidInference/musan",
            hf_patterns=("**/*.wav",),
            size=600_000_000,
            license="CC BY 4.0",
            source=f"{HF}/datasets/FluidInference/musan",
            used_for="background noise for augmentation",
        ),
        Asset(
            "mit-rirs",
            "rirs/mit",
            hf_repo="davidscripka/MIT_environmental_impulse_responses",
            hf_patterns=("16khz/*.wav",),
            size=10_000_000,
            license="no license stated by the source",
            source="https://mcdermottlab.mit.edu/Reverb/IR_Survey.html",
            profile="personal",
            used_for="room reverb for augmentation",
        ),
        *[
            Asset(
                f"mww-{n.replace('_', '-')}",
                f"mww/negative_datasets/{n}",
                url=f"{HF}/datasets/kahrendt/microwakeword/resolve/main/{n}.zip",
                sha256=sha,
                size=size,
                unzip=True,
                license="CC BY-NC 4.0",
                source=f"{HF}/datasets/kahrendt/microwakeword",
                profile="personal",
                used_for="microWakeWord negative examples",
            )
            for n, sha, size in [
                (
                    "speech",
                    "68b6b811b347b8321169f9e75ea2b9be73245943b4470af9532ca288d9f1b3cc",
                    3_183_001_091,
                ),
                (
                    "dinner_party",
                    "18a0885d595ced7faa73736a8680206f5ba6e80113ca3e6ce43130e510aac18f",
                    444_310_142,
                ),
                (
                    "no_speech",
                    "722dbcc967275c64c9581ecb85729549830a9094ce605ee22b2ddab9dbabf3c8",
                    2_000_317_854,
                ),
                (
                    "dinner_party_eval",
                    "e08e23a5e654dd4415a1c41a4b6cd02d1f983c821512367ad937bac1aeed8299",
                    82_329_019,
                ),
            ]
        ],
        Asset(
            "oww-melspectrogram",
            "oww/melspectrogram.onnx",
            url="https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx",
            sha256="ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f",
            size=1_087_958,
            license="openWakeWord release asset: Apache-2.0 per its README (its pre-trained models are CC BY-NC-SA 4.0; unclear which covers this file)",
            source="https://github.com/dscripka/openWakeWord",
            profile="personal",
            used_for="openWakeWord frontend",
        ),
        Asset(
            "oww-embedding",
            "oww/embedding_model.onnx",
            url="https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx",
            sha256="70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f",
            size=1_326_578,
            license="openWakeWord release asset, from Google's speech_embedding (Apache-2.0); see oww-melspectrogram",
            source="https://github.com/dscripka/openWakeWord",
            profile="personal",
            used_for="openWakeWord frontend",
        ),
        Asset(
            "oww-acav-features",
            "oww/openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
            url=f"{HF}/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
            sha256="721a66d0682c65a1b5c1da0aa109409cede1d20e28b15235c344b000cbb7654f",
            size=17_280_000_128,
            license="CC BY-NC-SA 4.0",
            source=f"{HF}/datasets/davidscripka/openwakeword_features",
            profile="personal",
            used_for="openWakeWord negative examples (2,000 h of general audio)",
        ),
        Asset(
            "oww-validation-features",
            "oww/validation_set_features.npy",
            url=f"{HF}/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy",
            sha256="a56a8a0f8e0efb91900acc6de4c0cdf4c564842e8475a7d49b36c039e17a690f",
            size=184_836_608,
            license="CC BY-NC-SA 4.0",
            source=f"{HF}/datasets/davidscripka/openwakeword_features",
            profile="personal",
            used_for="openWakeWord false-trigger rate estimate",
        ),
    ]
}

ASSETS["cmudict"] = Asset(
    "cmudict",
    "lexicon/cmudict.dict",
    url="https://raw.githubusercontent.com/cmusphinx/cmudict/74790861f652b15e4ac49015a90074ad62a27690/cmudict.dict",
    sha256="81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22",
    size=3_600_000,
    license="BSD-2-Clause (CMU)",
    source="https://github.com/cmusphinx/cmudict",
    used_for="pronunciations for sound-alike suggestions",
)

TTS_MODELS = {"libritts-high": "tts-libritts-high", "libritts-r-medium": "tts-libritts-r-medium"}


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def fetch(asset: Asset, progress: bool = True) -> Path:
    """Download (and verify / unpack) an asset unless it is already present."""
    if asset.present():
        return asset.local
    dest = asset.local
    dest.parent.mkdir(parents=True, exist_ok=True)
    if asset.hf_repo:
        from huggingface_hub import snapshot_download

        tmp = dest.with_name(dest.name + ".part")
        snapshot_download(
            repo_id=asset.hf_repo,
            repo_type="dataset",
            allow_patterns=list(asset.hf_patterns),
            local_dir=str(tmp),
        )
        shutil.rmtree(tmp / ".cache", ignore_errors=True)
        if dest.exists():
            shutil.rmtree(dest)
        tmp.rename(dest)
        (dest / ".complete").touch()
        return dest
    target = dest.with_suffix(".zip") if asset.unzip else dest
    part = target.with_name(target.name + ".part")
    _download(asset.url, part, asset.size, progress, asset.name)
    if asset.sha256:
        got = sha256_file(part)
        if got != asset.sha256:
            part.unlink()
            raise RuntimeError(f"{asset.name}: checksum mismatch (got {got}, want {asset.sha256})")
    if asset.unzip:
        with zipfile.ZipFile(part) as z:
            z.extractall(dest.parent)
        part.unlink()
        if not dest.is_dir():
            raise RuntimeError(f"{asset.name}: archive did not contain {dest.name}/")
        (dest / ".complete").touch()
    else:
        part.rename(dest)
    return dest


def _download(url: str, out: Path, size: int, progress: bool, label: str) -> None:
    import urllib.request

    from tqdm import tqdm

    have = out.stat().st_size if out.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req) as r:
        if have and r.status != 206:
            have = 0  # server ignored the range: start over
        total = int(r.headers.get("Content-Length", 0)) + have or size
        with (
            open(out, "ab" if have else "wb") as f,
            tqdm(
                total=total,
                initial=have,
                unit="B",
                unit_scale=True,
                desc=label,
                disable=not progress,
            ) as bar,
        ):
            while chunk := r.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))


def link_existing(asset: Asset, existing: Path) -> None:
    """Use a copy the user already has (e.g. from an earlier setup) instead of downloading."""
    dest = asset.local
    dest.parent.mkdir(parents=True, exist_ok=True)
    if asset.sha256 and existing.is_file() and sha256_file(existing) != asset.sha256:
        raise RuntimeError(f"{existing} is not {asset.name} (checksum differs)")
    if dest.exists() or dest.is_symlink():
        dest.unlink() if not dest.is_dir() or dest.is_symlink() else shutil.rmtree(dest)
    dest.symlink_to(existing.resolve(), target_is_directory=existing.is_dir())
