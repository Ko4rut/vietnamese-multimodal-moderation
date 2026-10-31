"""Colab Data Lake mount resolution, independent of Google APIs and Colab.

This helper never mounts Drive, creates folders, or authenticates on behalf of
another user. It validates the path chosen by the currently signed-in user.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_LAKE_NAME = "vietnamese-multilabel-topic-classification"


def resolve_mounted_lake(
    mount_point: str | Path,
    *,
    override: str | Path | None = None,
) -> Path:
    """Resolve an explicitly selected Data Lake or a matching MyDrive shortcut.

    For team access, first share the lake in Google Drive and add a shortcut
    with the standard folder name to each collaborator's My Drive, or set
    DATA_LAKE_ROOT explicitly to their mounted shared-folder location.
    """
    custom = override or os.environ.get("DATA_LAKE_ROOT")
    root = (
        Path(custom).expanduser()
        if custom
        else Path(mount_point).expanduser() / "MyDrive" / DEFAULT_LAKE_NAME
    )
    if not root.is_dir():
        raise FileNotFoundError(
            f"Data Lake folder not available at: {root}. "
            "Make sure Drive is mounted in this Colab session and the signed-in "
            "account has access. Add the shared folder as a shortcut named "
            f"'{DEFAULT_LAKE_NAME}' to My Drive, or set DATA_LAKE_ROOT to the "
            "actual mounted path. Cloning GitHub does not grant Drive access."
        )
    return root
