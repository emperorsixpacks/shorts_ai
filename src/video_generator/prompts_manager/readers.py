import os
import re
from enum import StrEnum
from dataclasses import dataclass, field

from video_generator.utils import return_base_dir
from video_generator.session_manager import Session
from video_generator.exceptions.sessionExceptions import ServerTimeOutError
from video_generator.exceptions.promptExceptions import (
    UnsupportedFileFormatError,
    InvalidLocationError,
)


class ReaderType(StrEnum):
    URL = "URL"
    FILE = "FILE"


@dataclass
class BaseReader:
    """
    Base class for readers.
    """

    file_path: str
    file_name: str = field(init=False)
    file_type: str = field(init=False)
    session: Session = field(init=False, default=None)
    _type: ReaderType = field(init=False, default=None)

    def __post_init__(self):
        self.file_path = self.set_file_path()
        self.file_name, self.file_type = self.return_name_and_type()
        if self.path_is_url(self.file_path):
            self._type = ReaderType.URL
            self.session: Session = Session(location=self.file_path)
        self._type = ReaderType.FILE

    @staticmethod
    def prompts_dir() -> str | None:
        prompts_dir = os.path.join(return_base_dir(), "prompts")
        if not os.path.exists(prompts_dir):
            return None
        return prompts_dir

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

    def set_file_path(self):
        """
        Sets the file_path of the prompt. Validates the file_path based on URL or path.
        Raises InvalidLocationError if the file_path is invalid.

        Parameters:
            file_path (str): The file_path to be set for the prompt.

        Returns:
            None
        """
        if self.path_is_url(self.file_path):
            if self.session.ping() != 200:
                raise InvalidLocationError(location=self.file_path)

        prompt_path = os.path.join(BaseReader.prompts_dir(), self.file_path)
        if not self.check_path(prompt_path):
            raise InvalidLocationError(location=self.file_path)
        return prompt_path

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

    def __del__(self):
        if self.session is not None:
            self.session.close()


class TextReader(BaseReader):
    """Class for reading txt file types

    Args:
        file_path (str): The path to the file.

    Raises:
        UnsupportedFileFormatError: If file_type is not txt.
    """

    def __post_init__(self):
        if self.file_type != "txt":
            raise UnsupportedFileFormatError(
                file=self.file_type, supported_format="txt"
            )

    def read(self):
        """
        Reads the contents of a file or URL based on the type specified in the instance variable `_type`.

        Returns:
            The contents of the file or URL as a string.

        Raises:
            KeyError: If the `_type` is not recognized.
        """
        reader = {ReaderType.URL: self._read_from_url, ReaderType.FILE: self._open_file}

        return reader.get(self._type)()


# TODO look into converting the read method into a class method
