import os
import re
from enum import StrEnum
from dataclasses import dataclass, field

from video_generator.session_manager import Session
from video_generator.exceptions.sessionExceptions import ServerTimeOutError
from video_generator.exceptions.promptExceptions import (
    UnsupportedFileFormatError,
    InvalidLocationError,
)


class ReaderType(StrEnum):
    URL = "URL"
    FILE = "FILE"

class BaseReader:

    def __init__(self, file_path: str):
        self.file_path: str = file_path
        self.file_name, self.file_type = self.return_name_and_type()
        if self.path_is_url(self.file_path):
            self._type = ReaderType.URL
            self.session: Session = Session(location=self.file_path)
        self._type = ReaderType.FILE

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
            if self.session.ping() != 200:
                raise InvalidLocationError(location=location)

        if not self.check_path(location):
            raise InvalidLocationError(location=location)

        self._location = location

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

    def _read_from_url(self):
        try:
            responce = self.session.get_content()
        except TimeoutError as e:
            raise ServerTimeOutError(location=self.file_path) from e
        return responce

    def _open_file(self):
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return f
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {self.file_path}") from e
        finally:
            f.close()


class TextReader(BaseReader):

    def __post_init__(self):
        if self.file_type != "txt":
            raise UnsupportedFileFormatError(
                file=self.file_type, supported_format="txt"
            )

    def read(self):
        reader = {ReaderType.URL: self._read_from_url, ReaderType.FILE: self._open_file}

        return reader.get(self._type)()