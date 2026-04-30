"""Merge labeled-soundscape h5 into a pseudo-label h5 in-place.

The pseudo_h5 is consumed by `PseudoLabelDataset` (concatenated to the train
set in `scripts/train.py`). Adding the 66 labeled soundscapes ensures the
ssval-aware student trains on both pseudo (10k+ unlabeled) and ground-truth
labeled soundscapes — same in-domain coverage as the v49 ssval campaign.

Keys that already exist in the pseudo h5 (e.g. if the pseudo_v7.csv happened
to include the labeled stems) are OVERWRITTEN with the ground-truth label
vectors. Spectrograms come from the labeled h5 because both are config-matched
recomputes of the same audio.

Usage:
    python scripts/merge_labeled_into_pseudo_h5.py \
        --labeled-h5 data/cache/soundscapes_config_a.h5 \
        --target-h5  data/cache/pseudo_v7_cfga.h5
"""
import argparse
import logging
from pathlib import Path
import h5py

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled-h5", required=True, type=Path)
    parser.add_argument("--target-h5", required=True, type=Path)
    args = parser.parse_args()

    if not args.labeled_h5.exists():
        raise FileNotFoundError(args.labeled_h5)
    if not args.target_h5.exists():
        raise FileNotFoundError(args.target_h5)

    with h5py.File(args.labeled_h5, "r") as src, h5py.File(args.target_h5, "a") as dst:
        if "spectrograms" not in dst:
            raise RuntimeError(f"{args.target_h5} missing spectrograms group")
        if "labels" not in dst:
            raise RuntimeError(f"{args.target_h5} missing labels group")
        src_keys = list(src["spectrograms"].keys())
        log.info("Source keys: %d  target keys before: %d",
                 len(src_keys), len(dst["spectrograms"]))
        added, replaced = 0, 0
        for k in src_keys:
            if k in dst["spectrograms"]:
                del dst["spectrograms"][k]
                replaced += 1
            else:
                added += 1
            dst["spectrograms"].create_dataset(k, data=src["spectrograms"][k][:])
            if k in dst["labels"]:
                del dst["labels"][k]
            dst["labels"].create_dataset(k, data=src["labels"][k][:])
        log.info("Added %d, replaced %d. Target keys after: %d",
                 added, replaced, len(dst["spectrograms"]))

if __name__ == "__main__":
    main()
