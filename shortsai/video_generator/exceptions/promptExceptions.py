class BasePromptException(Exception):
    """
    Base exception class for all prompt exceptions
    """
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class InvalidLocationError(BasePromptException):
    """
    This class represents an error for an invalid location.
    """

    def __init__(self, location) -> None:
        """
        Initialize InvalidLocationError with the provided location.
        
        Args:
            location (str): The invalid location.
        """
        self.location = location
        message = f"{self.location} is not a valid url/path"
        super().__init__(message)


class UnsupportedFileFormatError(BasePromptException):
    """
    This class represents an error for an unsupported file format.
    """

    def __init__(self, file:str=None, supported_format: str = None) -> None:
        """
        Initialize UnsupportedFileFormatError with the provided file format.
        
        Args:
            file (str): The unsupported file format.
        """
        message = f"Expected file of type {supported_format}, got {file}"
        super().__init__(message)
