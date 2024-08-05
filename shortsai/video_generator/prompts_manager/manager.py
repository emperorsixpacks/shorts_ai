"""
Should be removed, not necessary 

This module provides functionality for managing prompts in the video generation pipeline.

It includes classes and functions for reading prompts from files, checking the validity of file locations,
and handling exceptions related to prompts.

Classes:
    SupportedFileTypes: An enumeration of supported file types for prompts.
    PromptManager: A class for managing prompts, including reading prompts from files and checking file locations.
"""

from enum import StrEnum
from typing import Self
from dataclasses import dataclass, field

from video_generator.prompts_manager.protocols import FileReader

from video_generator.exceptions.promptExceptions import (
    UnsupportedFileFormatError,
)


class SupportedFileTypes(StrEnum):
    TXT = "txt"


@dataclass
class PromptManager:
    file_reader: FileReader
    contents: str = field(default=None, init=False)
    def read_prompt(self) -> Self:
        """
        Reads a prompt from a file or URL and sets the contents of the prompt.
        Returns:
            Self: The current instance of the class.

        Raises:
            UnsupportedFileFormatError: If the file type is not supported.

        """
        if self.file_reader is None:
            raise ValueError("file_reader cannot be None")

        if self.file_reader.file_type not in SupportedFileTypes:
            raise UnsupportedFileFormatError(
                file=self.file_reader.file_type, supported_format=SupportedFileTypes
            )
        
        self.contents: str | None  = self.file_reader.read()
        return self

# TODO write tests
