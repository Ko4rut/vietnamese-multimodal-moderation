"""Errors raised by the Bronze data access layer."""


class BronzeException(Exception):
    """Base exception for Bronze operations."""


class BronzeManifestError(BronzeException):
    """Bronze source manifest is missing or invalid."""


class BronzeFileNotFoundError(BronzeException):
    """The Bronze snapshot referenced by a manifest is not available."""


class BronzeIntegrityError(BronzeException):
    """Bronze file contents do not match their recorded manifest."""


class BronzeFormatError(BronzeException):
    """A Bronze JSONL record is malformed."""
