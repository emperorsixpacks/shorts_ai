"""
This module provides functionality for working with media files and FFmpeg filters.
"""

from typing import List, TypeVar, Self, Any
import logging
from tempfile import NamedTemporaryFile

import ffmpeg
from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator
from py_ffmpeg.exceptions import UnsupportedMediaFileError
from utils import MediaFile, SupportedMediaFileType, upload_file_to_s3
from settings import AWSSettings

FilterableStream = TypeVar("FilterableStream", "ffmpeg.nodes.FilterableStream", str)
error_logger = logging.getLogger("error_logger")
error_logger.setLevel(logging.ERROR)
error_looger_handler = logging.FileHandler("logs/error.log")
error_logger.addHandler(error_looger_handler)


class InputFile(BaseModel):
    """
    A class representing an input file for PyFFmpeg.

    Attributes:
        media_file (str): The location of the input media file.
        duration (float): The duration of the media file in seconds.
        stream (FilterableStream): The ffmpeg stream object for the input media file.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    media_file: MediaFile
    stream: FilterableStream = Field(
        default=None,
        init=False,
        description="The ffmpeg stream object for the input media file.",
    )

    @model_validator(mode="after")
    def get_media_stream(self) -> Self:
        """
        Generates the ffmpeg stream object for the input media file.

        Returns:
            Self: The input file object with the updated stream attribute.

        Raises:
            UnsupportedMediaFileError: If the media file is not of a supported type.
        """
        if not isinstance(self.media_file.file_type, SupportedMediaFileType):
            raise UnsupportedMediaFileError()
        self.stream = ffmpeg.input(self.media_file.url)
        if self.media_file.file_type == SupportedMediaFileType.VIDEO:
            self.media_file = self.media_file.set_duration()
        return self


class PyFFmpeg(BaseModel):
    """
    A class for processing media files with FFmpeg.

    The class takes in a list of video input files and a single audio input file,
    and generates a single output file with the concatenated video and audio.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    video: List[InputFile] | List[FilterableStream]
    audio: InputFile
    output_location: MediaFile
    overwrite: bool = Field(default=True)
    aws_client: Any
    aws_settings: AWSSettings
    filter_stream: FilterableStream = Field(init=False, default=None)
    
    # @classmethod
    @field_validator("video", mode="after")
    def rescale_video(cls, videos: List[InputFile]) -> List[FilterableStream]:
        """
        Validates the "video" field after it has been set.

        Args:
            cls (Type[Model]): The model class.
            videos (List[InputFile]): The list of input files.

        Returns:
            List[FilterableStream]: The list of filtered video streams.

        Raises:
            None
        """
        # print(videos)
        video_streams = []
        for video in videos:
            video_stream = video.stream 
            video_stream = ffmpeg.filter(video_stream, "scale", 1080, 1920)
            video_stream = ffmpeg.filter(video_stream, "setsar", 1, 1)
            video_streams.append(video_stream)

        return video_streams

    def concatinate_video(self) -> Self:
        """
        Concatenates multiple video streams into a single stream using ffmpeg.

        Returns:
            Self: The modified object with the concatenated video stream.
        """
        self.filter_stream = ffmpeg.concat(*[stream for stream in self.video], v=1, a=0)
        return self

    def trim_video(self, end: int = None) -> Self:
        """
        Trims the video to a specified duration.

        Args:
            end (int, optional): The duration to trim the video to in seconds.
                If not provided, the entire video will be trimmed.

        Returns:
            Self: The modified object with the trimmed video.
        """
        if end is None:
            end = self.audio.media_file.duration

        self.filter_stream = ffmpeg.trim(
            self.filter_stream, end=self.audio.media_file.duration
        )

        return self

    def execute(self) -> str:
        """
        Runs the FFmpeg process to output the filtered stream with audio to a specified location.

        Returns:
            MediaFile: The generated media file object.
        """

        stream = ffmpeg.output(
            self.filter_stream,
            self.audio.stream,
            "pipe:",
            shortest=None,
            f='mpegts'
        )
        
        try:
            process = ffmpeg.run(
                stream,
                overwrite_output=self.overwrite,
                capture_stdout=True,
                capture_stderr=True,
            )
            with NamedTemporaryFile() as temp_file:
                with open(temp_file.name, "wb") as f:
                    f.write(process[0])
                upload = upload_file_to_s3(
                    aws_client=self.aws_client,
                    media_file=self.output_location,
                    file_location=temp_file.name,
                    aws_settings=self.aws_settings,
                )

                if not upload:
                    print("failed to upload file to s3")
                    error_logger.error("Failed to uplad output file to S3")
                    return None
            
        except ffmpeg.Error as e:
            error_logger.error("Error occurred while running FFmpeg: %s", e.stderr)
            return None

        return self.output_location
