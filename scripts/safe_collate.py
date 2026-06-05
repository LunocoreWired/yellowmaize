"""
================================================================================
 scripts/safe_collate.py — DataLoader corruption guard
================================================================================
 Provides a collate function that filters None items returned by __getitem__
 when an image fails to load at training time.

 USAGE in every DataLoader:
   from scripts.safe_collate import safe_collate
   DataLoader(dataset, collate_fn=safe_collate, ...)

 The training loop must handle the case where an entire batch is None:
   for batch in loader:
       if batch is None:
           continue
================================================================================
"""

import torch
from torch.utils.data.dataloader import default_collate

# Module-level skip counter — reset per epoch by calling reset_skip_counter()
_skipped = 0


def safe_collate(batch: list):
    """
    Filter None entries (failed __getitem__) before collating.
    Returns None if the entire batch is invalid — caller must skip.
    """
    global _skipped
    valid = [item for item in batch if item is not None]
    _skipped += len(batch) - len(valid)
    if not valid:
        return None
    return default_collate(valid)


def reset_skip_counter() -> None:
    global _skipped
    _skipped = 0


def get_skip_count() -> int:
    return _skipped
