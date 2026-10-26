"""Small CLI for inspecting package metadata; pipelines are not implemented yet."""

import argparse

from vietnamese_topic_classification import __version__
from vietnamese_topic_classification.config import get_domain_labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vietnamese text topic classification (25 VTSNLP domains)"
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--list-domains", action="store_true", help="Print the 25 source-domain labels"
    )
    args = parser.parse_args()

    if args.list_domains:
        for label in get_domain_labels():
            print(label)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
