"""Stage 2 — Quality filtering and text normalisation."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CleanerConfig:
    # language ID filter
    language_id_enabled: bool = True
    language_id_language: str = "vi"
    language_id_min_confidence: float = 0.7
    language_id_model_path: str = "artifacts/lid.176.ftz"

    # length filter
    length_enabled: bool = True
    min_chars: int = 50
    max_chars: int = 100_000

    # non-alphanumeric ratio filter
    non_alphanum_enabled: bool = True
    non_alphanum_max_ratio: float = 0.3

    # repeated character ratio filter
    repeated_char_enabled: bool = True
    repeated_char_max_ratio: float = 0.2

    # transformations
    strip_html: bool = True
    normalize_unicode: bool = True
    normalize_whitespace: bool = True
    strip_urls: bool = False
    word_segment_enabled: bool = False
    word_segment_save_dir: str = "artifacts/vncorenlp"

    @classmethod
    def from_yaml(cls, cfg: dict) -> "CleanerConfig":
        f = cfg.get("filters", {})
        t = cfg.get("transformations", {})
        lid = f.get("language_id", {})
        length = f.get("length", {})
        na = f.get("non_alphanum_ratio", {})
        rc = f.get("repeated_char_ratio", {})
        ws = t.get("word_segment", {})
        return cls(
            language_id_enabled=lid.get("enabled", True),
            language_id_language=lid.get("language", "vi"),
            language_id_min_confidence=lid.get("min_confidence", 0.7),
            length_enabled=length.get("enabled", True),
            min_chars=length.get("min_chars", 50),
            max_chars=length.get("max_chars", 100_000),
            non_alphanum_enabled=na.get("enabled", True),
            non_alphanum_max_ratio=na.get("max_ratio", 0.3),
            repeated_char_enabled=rc.get("enabled", True),
            repeated_char_max_ratio=rc.get("max_ratio", 0.2),
            strip_html=t.get("strip_html", True),
            normalize_unicode=t.get("normalize_unicode", True),
            normalize_whitespace=t.get("normalize_whitespace", True),
            strip_urls=t.get("strip_urls", False),
            word_segment_enabled=ws.get("enabled", False) if isinstance(ws, dict) else bool(ws),
            word_segment_save_dir=ws.get("save_dir", "artifacts/vncorenlp") if isinstance(ws, dict) else "artifacts/vncorenlp",
        )


_FASTTEXT_MODEL_URL = (
    "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
)


def _download_fasttext_model(dest: Path) -> None:
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading FastText language ID model → {dest} ...")
    urllib.request.urlretrieve(_FASTTEXT_MODEL_URL, dest)
    print("  Done.")



_HTML_TAG = re.compile(r"<[^>]+>")
_HTML_ENTITIES = [
    ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
    ("&nbsp;", " "), ("&quot;", '"'), ("&#39;", "'"),
]
_MULTI_SPACE = re.compile(r"[ \t\r\f\v]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_URL = re.compile(r"https?://\S+")


class TextCleaner:
    def __init__(self, config: CleanerConfig):
        self._cfg = config
        self._ft_model = None
        self._vncorenlp = None
        if config.language_id_enabled:
            self._load_fasttext()
        if config.word_segment_enabled:
            self._load_vncorenlp()

    def _load_fasttext(self) -> None:
        import fasttext
        import fasttext.FastText as _ft_module
        import numpy as _np

        # fasttext calls np.array(probs, copy=False) which raises in NumPy ≥ 2.0.
        # Patch the module-level `np` reference once so it uses np.asarray instead.
        if not getattr(_ft_module, "_np2_patched", False):
            _real_np = _np

            class _NpShim:
                def __getattr__(self, name: str):
                    return getattr(_real_np, name)

                def array(self, obj, *args, copy=None, **kwargs):  # type: ignore[override]
                    if copy is False:
                        return _real_np.asarray(obj, *args, **kwargs)
                    if copy is not None:
                        kwargs["copy"] = copy
                    return _real_np.array(obj, *args, **kwargs)

            _ft_module.np = _NpShim()  # type: ignore[assignment]
            _ft_module._np2_patched = True  # type: ignore[attr-defined]

        model_path = Path(self._cfg.language_id_model_path)
        if not model_path.exists():
            _download_fasttext_model(model_path)
        self._ft_model = fasttext.load_model(str(model_path))

    def _load_vncorenlp(self) -> None:
        import py_vncorenlp
        save_dir = self._cfg.word_segment_save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        py_vncorenlp.download_model(save_dir=save_dir)
        self._vncorenlp = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=save_dir)

    def transform(self, text: str) -> str:
        if self._cfg.strip_html:
            text = _HTML_TAG.sub(" ", text)
            for entity, replacement in _HTML_ENTITIES:
                text = text.replace(entity, replacement)
        if self._cfg.normalize_unicode:
            text = unicodedata.normalize("NFC", text)
        if self._cfg.strip_urls:
            text = _URL.sub("", text)
        if self._cfg.normalize_whitespace:
            text = _MULTI_SPACE.sub(" ", text)
            text = _MULTI_NEWLINE.sub("\n\n", text)
            text = text.strip()
        if self._cfg.word_segment_enabled and self._vncorenlp is not None:
            sentences: list[str] = self._vncorenlp.word_segment(text)
            text = "\n".join(sentences)
        return text

    def clean(self, text: str) -> str | None:
        text = self.transform(text)
        if not self._passes_filters(text):
            return None
        return text

    def _passes_filters(self, text: str) -> bool:
        cfg = self._cfg
        n = len(text)

        if cfg.length_enabled:
            if n < cfg.min_chars or n > cfg.max_chars:
                return False

        if cfg.non_alphanum_enabled and n > 0:
            non_alnum = sum(1 for c in text if not c.isalnum() and not c.isspace())
            if non_alnum / n > cfg.non_alphanum_max_ratio:
                return False

        if cfg.repeated_char_enabled and n > 0:
            non_ws = [c for c in text if not c.isspace()]
            if non_ws:
                most_common_count = Counter(non_ws).most_common(1)[0][1]
            else:
                most_common_count = 0
            if non_ws and most_common_count / len(non_ws) > cfg.repeated_char_max_ratio:
                return False

        if cfg.language_id_enabled and self._ft_model is not None:
            labels, scores = self._ft_model.predict(text.replace("\n", " "))
            lang = str(labels[0]).replace("__label__", "")
            conf = float(scores[0])
            if lang != cfg.language_id_language or conf < cfg.language_id_min_confidence:
                return False

        return True


# ── Parallel worker helpers ───────────────────────────────────────────────────
# Module-level so they are picklable by ProcessPoolExecutor.

_worker_cleaner: "TextCleaner | None" = None


def _init_worker(cfg: CleanerConfig) -> None:
    global _worker_cleaner
    _worker_cleaner = TextCleaner(cfg)


def _clean_shard(shard_dir: str) -> tuple[list[str], int]:
    """Load one Arrow shard, apply cleaning, return (kept_texts, total_input)."""
    import datasets as hf_datasets
    texts: list[str] = hf_datasets.load_from_disk(shard_dir)["text"]
    kept = [c for t in texts if (c := _worker_cleaner.clean(t)) is not None]  # type: ignore[union-attr]
    return kept, len(texts)


# ── Public entry point ────────────────────────────────────────────────────────

def clean_dataset(config_path: str) -> None:
    import datasets as hf_datasets
    import yaml

    from vi_llm.data.utils import is_done, mark_done

    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f)

    input_dir = Path(cfg_dict["input_dir"])
    output_dir = Path(cfg_dict["output_dir"])
    shard_size = cfg_dict.get("shard_size", 10_000)
    num_workers = cfg_dict.get("num_workers", 1)

    if is_done(output_dir):
        print(f"Skipping clean — already done at {output_dir}")
        return

    cfg = CleanerConfig.from_yaml(cfg_dict)
    shard_dirs = sorted(str(p) for p in input_dir.rglob("shard_*"))

    if not shard_dirs:
        print(f"No shards found in {input_dir} — run make download first.")
        return

    print(f"Cleaning {len(shard_dirs)} shards with {num_workers} worker(s)...")

    executor = None
    if num_workers == 1:
        _init_worker(cfg)
        shard_results = map(_clean_shard, shard_dirs)
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        executor = ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=multiprocessing.get_context("fork"),
            initializer=_init_worker,
            initargs=(cfg,),
        )
        shard_results = executor.map(_clean_shard, shard_dirs)

    shard_idx = 0
    buffer: list[str] = []
    total_in = total_out = 0

    from tqdm import tqdm

    try:
        with tqdm(total=len(shard_dirs), desc="Cleaning shards", unit="shard") as pbar:
            for batch, n_in in shard_results:
                total_in += n_in
                buffer.extend(batch)
                total_out += len(batch)
                while len(buffer) >= shard_size:
                    _flush_shard(buffer[:shard_size], output_dir, shard_idx, hf_datasets)
                    buffer = buffer[shard_size:]
                    shard_idx += 1
                pct = total_out / total_in * 100 if total_in else 0
                pbar.set_postfix(kept=total_out, total=total_in, pct=f"{pct:.1f}%")
                pbar.update(1)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    if buffer:
        _flush_shard(buffer, output_dir, shard_idx, hf_datasets)

    mark_done(output_dir)
    print(f"\nCleaned: {total_out}/{total_in} documents kept  ({output_dir})")


def _flush_shard(texts: list[str], output_dir: Path, shard_idx: int, hf_datasets) -> None:
    shard_path = output_dir / f"shard_{shard_idx:05d}"
    hf_datasets.Dataset.from_dict({"text": texts}).save_to_disk(str(shard_path))
