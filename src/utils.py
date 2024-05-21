"""
Utility functions for common tasks in the project.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Self

from dataclasses import dataclass, field
from enum import StrEnum
from datetime import datetime

from botocore.exceptions import ClientError
import ffmpeg

if TYPE_CHECKING:
    from src.settings import AWSSettings


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
        self.name: str = self._format_name()
        self.duration: int = None

    def _format_name(self) -> str:
        """
        Returns a formatted name for the media file based on the file type.

        Returns:
            str: The formatted name of the media file.
        """
        self.name = self.name.replace(" ", "_").lower()
        match self.file_type:
            case SupportedMediaFileType.VIDEO:
                return f"video_{self.name}-{self.timestamp}.{self.file_type}"

            case SupportedMediaFileType.AUDIO:
                return f"audio_{self.name}-{self.timestamp}.{self.file_type}"
        
    def set_location(self, settings: AWSSettings) -> Self:
        """
        Sets the location URL of the media file by combining the fastly URL from the provided 'settings' with the file name.
        
        Args:
            settings (AWSSettings): The settings object containing the fastly URL.
        
        Returns:
            Self: The updated instance of the MediaFile with the URL set.
        """
        self.url = f"{settings.fastly_url}{self.name}"

    def set_duration(self) -> Self:
        """
        Retrieves and returns the duration of the media file.

        Args:
            url (str, optional): The URL of the media file. If not provided, uses the location attribute of the media file.

        Returns:
            int: The duration of the media file in seconds.
            None: If the location attribute is None or the media file cannot be probed.
        """
            
        if self.url is not None:
            self.duration = float(ffmpeg.probe(self.url)["format"]["duration"])
            return self
        return None
    
    def get_s3_location(self, settings: AWSSettings):
        return f"{settings.s3_url}{self.name}"
        


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




def upload_file_to_s3(
    aws_client,
    *,
    media_file: MediaFile,
    file_location,
    aws_settings: AWSSettings,
):
    """
    Uploads file content to a specified S3 URL using PUT request.

    Parameters:
    url (str): The S3 URL to upload the file to.
    file_content (bytes): The content of the file to upload.

    Returns:
    None
    """
    try:
        print("uplading file to s3")
        with open(file_location, "rb") as file:
            aws_client.upload_fileobj(
                Fileobj=file, Bucket=aws_settings.s3_bucket, Key=media_file.name, ExtraArgs={
                    "ContentType":media_file.file_type
                }
            )
    except ClientError:
        return False
    print("Done uploading file to s3")
    return True