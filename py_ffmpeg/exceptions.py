class BaseFFmpegExceptions(Exception):

    message: str = None

    def __init__(self, **kwargs) -> None:
        super().__init__(self.message)


class UnsupportedMediaFileError(BaseFFmpegExceptions):
    message = "File must of type .mp4 or .wav"

class UnsupportedVideoFileError(BaseFFmpegExceptions):
    message = "File must of type .mp4"


class UnsupportedAudioFileError(BaseFFmpegExceptions):
    message = "File must of type .wav"
