from unittest.mock import patch, MagicMock
from src.py_ffmpeg.main import PyFFmpeg, InputFile
from src.utils import MediaFile, SupportedMediaFileType
from src.settings import AWSSettings


# Test Data 
input_medai_file_1 = MediaFile(name="test_input_file_1", url="test_video.mp4", file_type=SupportedMediaFileType.VIDEO)
input_medai_file_2 = MediaFile(name="test_input_file_2", url="test_audio.mp3", file_type=SupportedMediaFileType.AUDIO)
input_video = InputFile(media_file=input_medai_file_1)
input_audio = InputFile(media_file=input_medai_file_2)
output_location = MediaFile(ame="test", url="output.mp4")
aws_settings = AWSSettings(bucket="test_bucket", region="us-west-1")

# Mock FFmpeg functions
mock_video_stream = MagicMock()
mock_audio_stream = MagicMock()
input_video.stream = mock_video_stream
input_audio.stream = mock_audio_stream

# Mock duration
input_audio.media_file.duration = 60

def test_initialization():
    # Test successful initialization
    ffmpeg_instance = PyFFmpeg(
        video=[input_video],
        audio=input_audio,
        output_location=output_location,
        aws_client=MagicMock(),
        aws_settings=aws_settings
    )
    
    assert ffmpeg_instance.video == [input_video]
    assert ffmpeg_instance.audio == input_audio
    assert ffmpeg_instance.output_location == output_location
    assert ffmpeg_instance.overwrite is True

def test_rescale_video():
    with patch('ffmpeg.filter', return_value=mock_video_stream) as mock_filter:
        scaled_videos = PyFFmpeg.rescale_video([input_video])
        assert len(scaled_videos) == 1
        mock_filter.assert_called_with(mock_video_stream, "setsar", 1, 1)

def test_concatinate_video():
    ffmpeg_instance = PyFFmpeg(
        video=[input_video],
        audio=input_audio,
        output_location=output_location,
        aws_client=MagicMock(),
        aws_settings=aws_settings
    )
    
    with patch('ffmpeg.concat', return_value=mock_video_stream):
        ffmpeg_instance.concatinate_video()
        assert ffmpeg_instance.filter_stream == mock_video_stream

def test_trim_video():
    ffmpeg_instance = PyFFmpeg(
        video=[input_video],
        audio=input_audio,
        output_location=output_location,
        aws_client=MagicMock(),
        aws_settings=aws_settings
    )
    
    with patch('ffmpeg.trim', return_value=mock_video_stream):
        ffmpeg_instance.concatinate_video()
        ffmpeg_instance.trim_video(end=30)
        assert ffmpeg_instance.filter_stream == mock_video_stream

def test_execute():
    ffmpeg_instance = PyFFmpeg(
        video=[input_video],
        audio=input_audio,
        output_location=output_location,
        aws_client=MagicMock(),
        aws_settings=aws_settings,
        subtitle=None
    )

    with patch('ffmpeg.output', return_value=mock_video_stream), \
         patch('ffmpeg.run', return_value=(b'', b'')), \
         patch('your_module.upload_file_to_s3', return_value=True), \
         patch('builtins.open', MagicMock()):
        output = ffmpeg_instance.execute()
        assert output == output_location
