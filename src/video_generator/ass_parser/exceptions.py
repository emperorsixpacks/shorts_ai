class BaseFileException(Exception):
    
    
    def __init__(self, message) -> None:
        self.message = message
        super().__init__(self.message)
        

class UnsupportedFileFormatError(BaseFileException):
    
    def __init__(self, file:str=None):
        message =  f"Expected file of type .ass, got {file}"
        super().__init__(message=message)