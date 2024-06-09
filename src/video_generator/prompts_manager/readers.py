import re
from dataclasses import dataclass, field

from video_generator.exceptions.sessionExceptions import ServerTimeOutError
from video_generator.exceptions.promptExceptions import  UnsupportedFileFormat

@dataclass
class BaseReader:

    file_path: str
    file_name: str = field(init=False)
    file_type: str = field(init=False)

    def __post_init__(self):
        self.file_name, self.file_type = self.return_name_and_type()

    def return_name_and_type(self):
        """
        Returns the name and type of a file based on its file path.

        Parameters:
            self (object): The current instance of the class.

        Returns:
            tuple: A tuple containing the file name (str) and file extension (str).

        Raises:
            ValueError: If the file path is invalid.

        """
        pattern = re.compile(r"([^\/\\\s]+)\.([a-zA-Z0-9]+)$")
        match = pattern.search(self.file_path)
        if not match:
            raise ValueError(f"Invalid file path {self.file_path}")

        file_name = match.group(1)
        file_extension = match.group(2)
        return file_name, file_extension
    
    def _read_from_url(self, session):
        try:
            responce = session.get_content()
        except TimeoutError as e:
            raise ServerTimeOutError(location=self.file_path) from e
        return responce
    
    def _open_file(self, mode):
        try:
            with open(self.file_path, mode.lower(), encoding="utf-8") as f:
                return f
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {self.file_path}") from e
        finally:
            f.close()


class TextReader(BaseReader):
     
    def __post_init__(self):
        if self.file_type != "txt":
            raise UnsupportedFileFormat(file=self.file_type, supported_format="txt")
        
    def read_from_url(self, session):
        """
        Reads the content from a URL using the session object.

        Returns:
            str: The content of the URL.

        Raises:
            ServerTimeOutError: If the server at the specified location does not respond within the timeout period.
        """
        return self._read_from_url(session=session)

    def read_from_path(self):
        """
        Reads the contents of a file located at `self.file_path` and returns it.

        Returns:
            str: The contents of the file.

        Raises:
            FileNotFoundError: If the file at `self.file_path` does not exist.
            PermissionError: If the file at `self.file_path` cannot be read.
        """
        return self._open_file(mode="r").read()


