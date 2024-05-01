from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field
from enum import StrEnum
from datetime import datetime

if TYPE_CHECKING:
    from settings import AWSSettings


class SupportedMediaFileType(StrEnum):
    """
    Supported media file types for downloading and converting
    """

    VIDEO = "mp4"
    AUDIO = "wav"


@dataclass
class MediaFile:
    """
    Dataclass for storing metadata about a media file

    name: name of the file
    size: size of the file in bytes
    duration: duration of the media file in seconds
    file_type: file type of the media file (e.g. "mp4", "wav")
    """

    name: str
    file_type: SupportedMediaFileType
    size: int = None
    duration: int = None
    url: str = None
    author: str = None
    timestamp: int = field(default_factory=lambda: int(datetime.now().timestamp()))

    def __post_init__(self):
        self.name = self.name.replace(" ", "_").lower()

    def return_formated_name(self):
        """
        Returns a formatted name for the media file based on the file type.

        Returns:
            str: The formatted name of the media file.
        """
        match self.file_type:
            case SupportedMediaFileType.VIDEO:
                return f"video_{self.name}-{self.timestamp}.{self.file_type}"

            case SupportedMediaFileType.AUDIO:
                return f"audio_{self.name}-{self.timestamp}.{self.file_type}"

    @property
    def location(self, aws_settings: AWSSettings):
        """
        Returns the location of the media file.

        Returns:
            str: The location of the media file.
            None: If the name of the media file is None.
        """
        if self.name is None:
            return None
        return f"{aws_settings.fastly_url}{self.return_formated_name()}"


class AWSS3Method(StrEnum):
    """
    Supported AWS S3 methods
    """

    PUT = "put_object"
    GET = "get_object"


@dataclass
class Story:
    """
    Dataclass for storing metadata about a story.

    Attributes:
        prompt (str): The prompt for the story.
        text (str): The story text.
    """

    prompt: str
    text: str

    @property
    def length(self) -> int:
        """
        Returns the length of the story in words.

        Returns:
            int: The length of the story in words.
        """
        return len(self.text.split())


@dataclass
class WikiPage:
    """
    Dataclass for storing metadata about a Wikipedia page

    page_title: title of the Wikipedia page (in lower case)
    text: text content of the Wikipedia page
    """

    page_title: str
    text: str

    def __post_init__(self):
        self.page_title = self.page_title.lower()

    def __str__(self):
        return self.page_title
