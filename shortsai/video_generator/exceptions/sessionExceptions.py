class BaseSessionException(Exception):
    """
    Base exception class for all prompt exceptions
    """
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)

class ServerTimeOutError(BaseSessionException):
    def __init__(self, location) -> None:
        """
        Initialize ServerTimeOutError with the provided location.
        
        Args:
            location (str): The location of the server.
        """
        self.message = f"server at {location} did not respond"
        super().__init__(self.message)


class ResourceNotFoundError(BaseSessionException):
    def __init__(self, location) -> None:
        """
        Initialize ResourceNotFoundError with the provided location.
        
        Args:
            location (str): The location of the server.
        """
        self.message = f"resource at {location} not found"
        super().__init__(self.message)


class ServerError(BaseSessionException):
    def __init__(self, location, status_code) -> None:
        """
        Initialize ServerError with the provided location.
        
        Args:
            location (str): The location of the server.
        """
        self.message = f"server at {location} returned an error {status_code}"
        super().__init__(self.message)