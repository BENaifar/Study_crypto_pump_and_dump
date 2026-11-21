"""One-time, non-destructive copy of the authors' shipped inputs into ``data/`` (plan §2.3)."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from pnd.config import Config
from pnd.pipeline.registry import register

logger = logging.getLogger(__name__)


def _copy_tree_files(src_dir: Path, dst_dir: Path) -> tuple[int, int]:
    copied = skipped = 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(src_dir.iterdir()):
        if not src.is_file() or src.name == ".DS_Store":
            continue
        dst = dst_dir / src.name
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            skipped += 1
            continue
        shutil.copy2(src, dst)
        copied += 1
    return copied, skipped


@register
class MigrateDataStage:
    """Copy (never move) shipped Telegram / CoinGecko / embedding files into the new data layout."""

    name = "migrate-data"

    def run(self, cfg: Config) -> None:
        root = cfg.root
        jobs = [
            (root / "Data" / "Telegram" / "raw", cfg.paths.telegram.raw_dir),
            (root / "Data" / "Telegram" / "Labeled", cfg.paths.pd_log.raw.parent),
            (root / "TargetCoinPrediction" / "FeatGeneration" / "raw", cfg.paths.negative_samples.required_files.parent),
        ]
        for src, dst in jobs:
            if src.resolve() == dst.resolve():
                raise RuntimeError(f"Refusing to copy {src} onto itself (case-insensitive file system?)")
        for src, dst in jobs:
            copied, skipped = _copy_tree_files(src, dst)
            logger.info("%s -> %s: copied=%d skipped=%d", src, dst, copied, skipped)
        wv_src = root / "TargetCoinPrediction" / "FeatGeneration" / "feature" / "wv_embedding.pkl"
        wv_dst = cfg.paths.model.wv_embedding
        wv_dst.parent.mkdir(parents=True, exist_ok=True)
        if not (wv_dst.exists() and wv_dst.stat().st_size == wv_src.stat().st_size):
            shutil.copy2(wv_src, wv_dst)
            logger.info("%s -> %s: copied", wv_src, wv_dst)
        else:
            logger.info("%s -> %s: skipped", wv_src, wv_dst)
