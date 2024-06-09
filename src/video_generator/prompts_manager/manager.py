import os
from enum import StrEnum
from typing import Self
from video_generator.prompts_manager.protocols import FileReader

from video_generator.session_manager import Session
from video_generator.exceptions.promptExceptions import InvalidLocationError, UnsupportedFileFormat
from video_generator.exceptions.sessionExceptions import ServerTimeOutError

class SupportedFileTypes(StrEnum):
    TXT = "txt"

class PromptsBase:
    def __init__(self, location) -> None:
        self.location = location
        self.contents: str = None
        self.session: Session = None

    @staticmethod
    def check_path(path: str) -> bool:
        """
        A method to check if the specified path exists.

        Parameters:
            path (str): The path to check.

        Returns:
            bool: True if the path exists, False otherwise.
        """
        return os.path.exists(path)

    @staticmethod
    def path_is_url(url) -> bool:
        """
        A static method to check if a given path is a URL.

        Parameters:
            url (str): The path to check.

        Returns:
            bool: True if the path is a URL, False otherwise.
        """
        if url.startswith(("https://", "http://", "www.")):
            return True
        return False

    @property
    def location(self):
        """
        Getter method for the 'location' property.
        Returns the value of the '_location' attribute.
        """
        return self._location

    @location.setter
    def set_location(self, location: str):
        """
        Sets the location of the prompt. Validates the location based on URL or path.
        Raises InvalidLocationError if the location is invalid.

        Parameters:
            location (str): The location to be set for the prompt.

        Returns:
            None
        """
        if self.path_is_url(location):
            self.session = Session(location=location)
            if self.session.ping() != 200:
                raise InvalidLocationError(location=location)

        if not self.check_path(location):
            raise InvalidLocationError(location=location)

        self._location = location

    def read_prompt(self, reader:FileReader) -> Self:
        """
        Reads a prompt from a file or URL and sets the contents of the prompt.

        Parameters:
            reader (FileReader): The FileReader instance used to read the prompt.

        Returns:
            Self: The current instance of the class.

        Raises:
            UnsupportedFileFormat: If the file type is not supported.

        """
        if reader.file_type not in SupportedFileTypes:
            raise UnsupportedFileFormat(file=reader.file_type, supported_format=SupportedFileTypes)
        if self.path_is_url(self.location):
            self.contents = reader.read_from_url(session=self.session)
        else:
           self.contents = reader.read_from_path()

        return self

    def __del__(self):
        self.session.close()
