from typing import List, TypeVar, Self
import logging

import ffmpeg
from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator
from py_ffmpeg.exceptions import UnsupportedMediaFileError

from utils import MediaFile, SupportedMediaFileType
from settings import AWSSettings


FilterableStream = TypeVar("FilterableStream", "ffmpeg.nodes.FilterableStream", str)


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
    aws_settings: AWSSettings = Field(init=False, default=None)

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
        self.media_file = self.media_file.set_duration()
        return self


class PyFFmpeg(BaseModel):
    """
    A class for generating a video file by concatenating multiple video files and audio file using FFmpeg.

    Attributes:
        video (List[InputFile]): A list of video input files.
        audio_file (InputFile): The audio input file.
        overwrite (bool): If True, will overwrite existing output file. (default: True)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    video: List[InputFile]
    audio: InputFile
    output_location: MediaFile
    overwrite: bool = Field(default=True)
    filter_stream: FilterableStream = Field(init=False, default=None)

    @field_validator("video", mode="after")
    def reduce_video_quality(self) -> List[FilterableStream]:
        video_streams = []
        for video in self.video:
            video_stream = video.stream
            video_stream = ffmpeg.filter(video_stream, "scale", 406, 720)
            video_stream = ffmpeg.filter(video_stream, "setsar", 1, 1)
            video_streams.append(video_stream)

        self.video.clear()
        self.video.extend(video_streams)

        return self.video

    def concatinate_video(self) -> Self:
        """
        Concatenates multiple video streams into a single stream using ffmpeg.

        Returns:
            Self: The modified object with the concatenated video stream.
        """
        self.filter_stream = ffmpeg.concat(
            *[video.stream for video in self.video], v=1, a=0
        )
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
            end = self.audio_file.media_file.duration

        self.filter_stream = ffmpeg.trim(
            self.filter_stream, end=self.audio_file.media_file.duration
        )

        return self

    def execute(self) -> str:
        """
        Runs the FFmpeg process to output the filtered stream with audio to a specified location.

        Returns:
            MediaFile: The generated media file object.
        """

        process = ffmpeg.output(
            self.filter_stream,
            self.audio_file.stream,
            self.output_location.url,
            shortest=None,
        )
        try:
            process.run(overwrite_output=self.overwrite)

        except ffmpeg.Error() as e:
            # add loggin here
            return none

        return self.output_location
