class InvalidLabelError(ValueError):
    """ Base class for exceptions in this module. """

    def __init__(self, message: str = "Invalid labels") -> None:
        super().__init__(message)

class InvalidLabelCountError(InvalidLabelError):
    """Exception raised for invalid label numbers."""

    def __init__(self, message: str = "The label number is invalid") -> None:
        super().__init__(message)

class LabelsOverlapError(InvalidLabelError):
    """Exception raised for invalid label numbers."""

    def __init__(self, message: str = "The labels are potentially overlapping, creating a data leak") -> None:
        super().__init__(message)

class TooManyLabelsError(InvalidLabelCountError):
    """Exception raised when too many labels are provided."""

    def __init__(self, message: str = "The number of labels is too high") -> None:
        super().__init__(message)

class NotEnoughLabelsError(InvalidLabelCountError):
    """Exception raised when too few labels are provided."""

    def __init__(self, message: str = "The number of labels is too low") -> None:
        super().__init__(message)

class NoLabelsError(InvalidLabelCountError):
    """Exception raised when no labels are provided."""

    def __init__(self, message: str = "No labels provided") -> None:
        super().__init__(message)

class AnnotationFileNotSetError(InvalidLabelError):
    """Exception raised when the annotation file is not set."""

    def __init__(self, message: str = "Annotation file not set") -> None:
        super().__init__(message)


class OrthomosaicNotSetError(ValueError):
    """Exception raised when the Orthomosaic file is not set."""

    def __init__(self, message: str = "Annotation file not set") -> None:
        super().__init__(message)


class ProjectionError(ValueError):
    pass