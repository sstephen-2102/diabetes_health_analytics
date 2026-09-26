class DataLoadError(Exception):
    """Raised when a dataset cannot be loaded."""

class DataValidationError(Exception):
    """Raised when dataset validation fails."""

class MissingColumnError(DataValidationError):
    """Raised when a required column is missing."""

class InvalidParameterError(Exception):
    """Raised when an analytical parameter is invalid."""

class UnsupportedAnalysisError(Exception):
    """Raised when an unsupported analysis is requested."""

class ModelTrainingError(Exception):
    """Raised when model training fails."""
