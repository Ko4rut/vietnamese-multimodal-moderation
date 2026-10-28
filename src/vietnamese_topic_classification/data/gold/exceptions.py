"""Gold-specific errors."""


class GoldError(Exception):
    """Base error for Gold processing."""


class GoldValidationError(GoldError):
    """Silver input or Gold row violates its schema."""


class GoldLeakageError(GoldError):
    """Near-duplicate scan could not complete within configured safety limits."""


class GoldPublicationError(GoldError):
    """An immutable output could not be safely published."""


class GoldManifestError(GoldError):
    """Gold manifest is malformed or cannot be read."""


class GoldIntegrityError(GoldError):
    """Gold data differs from its manifest."""
