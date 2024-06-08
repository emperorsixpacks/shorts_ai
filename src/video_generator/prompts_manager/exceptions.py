class BasePromptException(Exception):

    def __init__(self, message:str) -> None:
        self.message = message
        super().__init__(self.message)

class InvalidLocationError(BasePromptException):
    def __init__(self, location) -> None:
        self.location = location
        message = f"{self.location} is not a valid url/path" 
        super().__init__(message)


