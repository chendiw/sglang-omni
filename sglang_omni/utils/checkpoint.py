# SPDX-License-Identifier: Apache-2.0
"""Checkpoint resolution helpers."""

from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=None)
def resolve_checkpoint(checkpoint: str) -> str:
    """Resolve a checkpoint spec to a local directory.

    Memoized because stages sharing a process each resolve the same spec,
    and every repeat call costs a Hub revalidation round trip at startup.
    """
    if os.path.isdir(checkpoint):
        return checkpoint
    else:
        pass
    from huggingface_hub import snapshot_download

    # note (db-ol): a checkpoint spec may pin a snapshot as
    # <repo-id>@<revision>, the format .github/scripts/ensure_hf_models.sh
    # uses. Without the @ the latest revision is resolved.
    repo_id, _, revision = checkpoint.partition("@")
    return snapshot_download(repo_id, revision=revision or None)
