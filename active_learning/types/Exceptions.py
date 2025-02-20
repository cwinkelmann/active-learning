class InvalidLabelCountError(ValueError):
    """Exception raised for invalid label numbers."""

    def __init__(self, message: str = "The label number is invalid") -> None:
        super().__init__(message)


class TooManyLabelsError(InvalidLabelCountError):
    """Exception raised when too many labels are provided."""

    def __init__(self, message: str = "The number of labels is too high") -> None:
        super().__init__(message)

class NotEnoughLabelsError(InvalidLabelCountError):
    """Exception raised when too few labels are provided."""

    def __init__(self, message: str = "The number of labels is too low") -> None:
        super().__init__(message)