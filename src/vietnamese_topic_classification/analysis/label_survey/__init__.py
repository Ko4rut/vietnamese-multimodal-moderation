"""Survey of the machine-assigned ``domain`` labels of VTSNLP/vietnamese_curated_dataset.

The package collects evidence only (counts, confusion, suspected mislabels, what-if
scenarios, an audit pack). It never decides which labels to keep, merge or drop, and it
never writes into Bronze/Silver/Gold/experiments snapshots. See ``docs/LABEL_SURVEY.md``.

Importing the package is cheap; heavy dependencies are imported by the submodules.
"""
