from __future__ import annotations
from typing import TYPE_CHECKING, Self

from dataclasses import dataclass, field
from enum import StrEnum
from datetime import datetime
import ffmpeg

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
    url: str = None
    author: str = None
    timestamp: int = field(default_factory=lambda: int(datetime.now().timestamp()), init=False)

    def __post_init__(self):
        self.name: str = self.name.replace(" ", "_").lower()
        self.duration: int = None

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

    def set_location(self, aws_settings: AWSSettings) -> Self:
        """
        Returns the location of the media file.

        Returns:
            str: The location of the media file.
            None: If the name of the media file is None.
        """
        if self.name is None:
            return None
        self.url = f"{aws_settings.fastly_url}{self.return_formated_name()}"
        return self

    def set_duration(self, url: str = None) -> Self:
        """
        Retrieves and returns the duration of the media file.

        Args:
            url (str, optional): The URL of the media file. If not provided, uses the location attribute of the media file.

        Returns:
            int: The duration of the media file in seconds.
            None: If the location attribute is None or the media file cannot be probed.
        """

        if url is None:
            url = self.url
        if self.url is None:
            return None

        self.duration = ffmpeg.probe(url)["format"]["duration"]
        return self


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
