"""Custom exceptions for VisionGuard."""


class VisionGuardError(Exception):
    """Base exception."""


class DatasetError(VisionGuardError):
    """Raised when dataset operations fail."""


class AnnotationError(VisionGuardError):
    """Raised when annotation conversion fails."""
