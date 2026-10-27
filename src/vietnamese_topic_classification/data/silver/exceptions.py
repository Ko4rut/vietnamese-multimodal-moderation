"""Exceptions for Silver pipeline operations (invalid records are not exceptions)."""


class SilverException(Exception):
    """Base class for Silver processing failures."""


class SilverPublicationError(SilverException):
    """Silver artifacts could not be safely published."""
