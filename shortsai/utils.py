"""
Utility functions for common tasks in the project.
"""

from __future__ import annotations
import os
import time
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Self, List, Dict
from functools import lru_cache

from dataclasses import dataclass, field
from enum import StrEnum
from datetime import datetime
import requests

from ibm_botocore.exceptions import ClientError
import ffmpeg
from redis import Redis
import asyncpraw

from langchain.schema import Document
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain.vectorstores.redis import RedisVectorStoreRetriever, Redis
from langchain.text_splitter import RecursiveCharacterTextSplitter

from shortsai.session_manager import Session
from shortsai.constants import (
    DEFAULT_WIKIPEDIA_SEARCH_PARAMS,
    WIKI_API_SEARCH_URL,
    TTS_MAKER_URL,
)

if TYPE_CHECKING:
    from shortsai.settings import (
        BucketSettings,
        HuggingFaceHubSettings,
        RedditSettings,
        RedisSettings,
    )
    from wikipediaapi import Wikipedia


def get_base_url(path):
    return os.path.dirname(os.path.abspath(path=path))

def return_base_dir():
    return os.path.dirname(os.path.abspath(""))  # os.path.abspath(path=__file__))


class MediaFileType(StrEnum):
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
    file_type: MediaFileType
    size: int = None
    url: str = None
    author: str = None
    timestamp: int = field(
        default_factory=lambda: int(datetime.now().timestamp()), init=False
    )

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
            case MediaFileType.VIDEO:
                return f"video_{self.name}-{self.timestamp}.{self.file_type}"

            case MediaFileType.AUDIO:
                return f"audio_{self.name}-{self.timestamp}.{self.file_type}"

    def set_location(self, settings: BucketSettings) -> Self:
        """
        Sets the location URL of the media file by combining the fastly URL from the provided 'settings' with the file name.

        Args:
            settings (BucketSettings): The settings object containing the fastly URL.

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

    def get_s3_location(self, settings: BucketSettings) -> str:
        """
        Returns the S3 location of the media file based on the provided settings.

        Args:
            settings (BucketSettings): The settings object containing the S3 URL.

        Returns:
            str: The S3 location of the media file.
        """


class AWSS3Method(StrEnum):
    """
    Supported AWS S3 methods
    """

    PUT = "put_object"
    PGET = "get_object"


@dataclass
class WikiPage:
    """
    Dataclass for storing metadata about a Wikipedia page

    page_title: title of the Wikipedia page (in lower case)
    text: text content of the Wikipedia page
    """

    page_title: str
    wikipidea_client: Wikipedia
    text: str = field(default=None)

    def __post_init__(self):
        self.page_title = self.page_title.lower()

    def __str__(self):
        return self.page_title

    @classmethod
    def wiki_search(cls, query: str, params: Dict[str]):
        """
        Searches for a given query on the Wikipedia API and returns a list of page keys.

        Parameters:
            query (str): The search query to be used for the Wikipedia API.

        Returns:
            list: A list of page keys extracted from the response of the Wikipedia API.
        """

        params = params if params is not None else DEFAULT_WIKIPEDIA_SEARCH_PARAMS

        if params.get("q", None) is None:
            params["q"] = query

        # Send the API request
        response = Session(url=WIKI_API_SEARCH_URL).get_json(params=params)
        pages = response["pages"]
        return [page["key"].lower() for page in pages]

    async def get_page_content(self) -> Self:
        """
        A function that retrieves the content of a Wikipedia page based on the provided title.

        Args:
            title (str): The title of the Wikipedia page to retrieve.

        Returns:
            WikiPage: An instance of WikiPage containing the title and text of the Wikipedia page.
        """

        self.text = self.wikipidea_client.page(title=self.page_title)

        return self


def upload_file_to_s3(
    s3_client,
    *,
    media_file: MediaFile,
    file_location,
    bucket_settings: BucketSettings,
) -> bool:
    """
    Uploads a file to an S3 bucket.

    Args:
        s3_client: An S3 client object (aws or ibm cloud).
        media_file: A MediaFile object containing metadata about the file.
        file_location: The location of the file to upload.
        bucket_settings: BucketSettings object containing settings for AWS.

    Returns:
        bool: True if the file was successfully uploaded, False otherwise.
    """
    try:
        print("Uploading file to S3")

        with open(file_location, "rb") as file:
            s3_client.upload_fileobj(
                Fileobj=file,
                Bucket=bucket_settings.bucket_name,
                Key=media_file.name,
                ExtraArgs={"ContentType": media_file.file_type},
            )

    except ClientError:
        return False
    print("Done uploading file to S3")
    return True


def index_exists(client: Redis, index_name: str) -> bool:
    """Check if Redis index exists."""
    try:
        client.ft(index_name).info()
    except:
        # logger.debug("Index does not exist")
        print("Index does not exit")
        return False
    # logger.debug("Index already exmists")
    return True


@lru_cache
def load_embeddings_model(embeddings_settings: HuggingFaceHubSettings):
    """
    A description of the entire function, its parameters, and its return types.
    """
    return HuggingFaceBgeEmbeddings(model_name=embeddings_settings.embedding_model)


def get_reddit_client(reddit_settings: RedditSettings):
    return asyncpraw.Reddit(
        client_id=reddit_settings.reddit_client_id,
        client_secret=reddit_settings.reddit_client_secret,
        user_agent="vidoe_bot",
    )


def chunk_and_save(page: WikiPage, redis_settings: RedisSettings, embeddings):
    """
    Splits the text of each WikiPage in the given list into smaller chunks using the RecursiveCharacterTextSplitter.

    Args:
        pages (List[WikiPage]): A list of WikiPage objects containing the text to be split.

    Returns:
        List[Redis]: A list of Redis objects created from the split text of each WikiPage.
    """

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
    page_splits = (
        WikiPage(page_title=page.page_title, text=text_splitter.split_text(page.text))
        if page.text != ""
        else None
    )
    if page_splits is None:
        return False
    Redis.from_texts(
        page_splits.text,
        embeddings,
        redis_url=redis_settings.redis_url,
        index_name=page_splits.page_title,
    )
    return True


def return_documents(
    user_prompt: str,
    redis_settings: RedisSettings,
    *,
    index_names: List[str],
    embeddings,
) -> List[Document]:
    """
    Generates a list of Document objects by invoking the RedisVectorStoreRetriever with the given user prompt and index names.

    Args:
        user_prompt (str): The user prompt to be passed to the RedisVectorStoreRetriever.
        index_names (List[str]): The list of index names to be used by the RedisVectorStoreRetriever.

    Returns:
        List[Document]: The list of Document objects generated by the RedisVectorStoreRetriever.
    """
    return [
        RedisVectorStoreRetriever(
            vectorstore=Redis(
                redis_url=redis_settings.redis_url,
                embedding=embeddings,
                index_name=index_name.lower(),
            ),
            search_kwargs={"k": 5, "distance_threshold": None},
        ).invoke(user_prompt)
        for index_name in index_names
    ]


def extract_entities(text: str, ner_model, hf_hub_settings: HuggingFaceHubSettings):
    """
    A function that returns named entity recognition (NER) tokens from the given text.

    Parameters:
    - text (str): The input text for which NER tokens need to be extracted.

    Returns:
    - list: A list of NER tokens extracted from the input text.
    """
    result = ner_model.token_classification(
        text=text, model=hf_hub_settings.ner_repo_id
    )
    return [i["word"].strip() for i in result]


def convert_text_to_audio(
    client, name: str, text: str, bucket_settings: BucketSettings
) -> MediaFile | None:
    """
    Converts text to audio using the TTSMP3 API.

    Parameters:
        client (Client): The AWS client.
        name (str): The name of the audio file.
        text (Story): The text to be converted to audio.

    Returns:
        MediaFile | None: The converted audio file if successful, None otherwise.
    """
    try:
        # logger.info("Converting text to audio")
        media_file = MediaFile(name=name, file_type=MediaFileType.AUDIO)
        data = {"msg": text, "lang": "Matthew", "source": "ttsmp3"}
        response = requests.post(TTS_MAKER_URL, data=data, timeout=60)
        response.raise_for_status()
        audio_url = response.json().get("URL")

        if not audio_url:
            # logger.warning("No URL found in TTSMP3 response")
            return None

        media_file.url = audio_url

        with NamedTemporaryFile(delete=False) as tempfile:
            download_response = requests.get(audio_url, stream=True, timeout=60)
            download_response.raise_for_status()

            for chunk in download_response.iter_content(chunk_size=1024):
                tempfile.write(chunk)

        # logger.info("Uploading audio to S3")
        upload_success = upload_file_to_s3(
            client,
            media_file=media_file,
            file_location=tempfile.name,
            bucket_settings=bucket_settings,
        )

        if not upload_success:
            # logger.warning("Failed to upload audio to S3")
            return None

        # logger.info("Audio converted successfully")
        return media_file.set_duration()

    except requests.RequestException as e:
        # logger.error(f"HTTP request failed: {e}")
        return None
    except Exception as e:
        # logger.error(f"An unexpected error occurred: {e}")
        return None


def transcribe_audio(
    client, *, audio: MediaFile, bucket_settings: BucketSettings
) -> str | None:
    """
    Transcribes audio using AWS Transcribe.

    Parameters:
        client (boto3.client): The AWS Transcribe client.
        audio (MediaFile): The audio file to be transcribed.

    Returns:
        str | None: The URI of the transcript if successful, None otherwise.
    """
    try:
        transcription_job_name = audio.name
        s3_location = audio.get_s3_location(settings=bucket_settings)

        # logger.info(f"Starting transcription job for {transcription_job_name}.")
        client.start_transcription_job(
            TranscriptionJobName=transcription_job_name,
            Media={"MediaFileUri": s3_location},
            MediaFormat="wav",
            LanguageCode="en-US",
        )

        for _ in range(60):
            job = client.get_transcription_job(
                TranscriptionJobName=transcription_job_name
            )
            job_status = job["TranscriptionJob"]["TranscriptionJobStatus"]

            if job_status in ["COMPLETED", "FAILED"]:
                # logger.info(f"Job {transcription_job_name} is {job_status}.")
                if job_status == "COMPLETED":
                    transcript_uri = job["TranscriptionJob"]["Transcript"][
                        "TranscriptFileUri"
                    ]
                    return transcript_uri
                break
            else:
                pass
                # logger.info(f"Waiting for {transcription_job_name}. Current status: {job_status}.")
            time.sleep(10)

    except client.exceptions.BadRequestException as e:
        # logger.error(f"Bad request: {e}")

        pass
    except client.exceptions.LimitExceededException as e:
        # logger.error(f"Limit exceeded: {e}")
        pass
    except client.exceptions.InternalFailureException as e:
        # logger.error(f"Internal failure: {e}")
        pass
    except client.exceptions.ConflictException as e:
        # logger.error(f"Conflict: {e}")
        pass
    except client.exceptions.ServiceUnavailableException as e:
        # logger.error(f"Service unavailable: {e}")
        pass
    except Exception as e:
        # logger.error(f"An unexpected error occurred: {e}")
        pass

    return None


# TODO add interface for both AWS and IBM Cloud
