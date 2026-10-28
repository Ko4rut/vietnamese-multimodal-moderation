"""Exceptions for Silver pipeline operations (invalid records are not exceptions)."""


class SilverException(Exception):
    """Base class for Silver processing failures."""


class SilverPublicationError(SilverException):
    """Silver artifacts could not be safely published."""


class SilverReportError(SilverException):
    """Silver quality report is missing, unreadable or malformed."""


class SilverFileNotFoundError(SilverException):
    """Silver snapshot referenced by the report was not found."""


class SilverIntegrityError(SilverException):
    """Silver snapshot does not match its quality report."""


class SilverFormatError(SilverException):
    """Silver JSONL contains an invalid record."""
