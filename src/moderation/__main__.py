"""Minimal executable entrypoint; inference comes in later phases."""

import argparse

from moderation import __version__
from moderation.config import load_taxonomy


def main() -> None:
    parser = argparse.ArgumentParser(description="Vietnamese moderation research scaffold")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--list-labels",
        action="store_true",
        help="Print the current research label identifiers",
    )
    args = parser.parse_args()
    if args.list_labels:
        for label in load_taxonomy()["labels"]:
            print(label["id"])
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
