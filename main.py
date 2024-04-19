import re
import random
from typing import List
from dataclasses import dataclass
from functools import lru_cache
from enum import StrEnum

import requests
import wikipediaapi
import praw
from ffmpeg import FFmpeg
import boto3
from botocore.exceptions import NoCredentialsError

from huggingface_hub import InferenceClient

from langchain_community.llms.ai21 import AI21, AI21PenaltyData
from langchain_community.chat_models.deepinfra import ChatDeepInfra
from langchain_core.messages import HumanMessage, SystemMessage

from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain.vectorstores.redis import RedisVectorStoreRetriever, Redis

from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

from settings import (
    RedditSettings,
    RedisSettings,
    EmbeddingSettings,
    TxtSpeechsettings,
    LLMsettings,
    HuggingFaceHubSettings,
    AWSSettings,
)

reddit_settings = RedditSettings()
redis_settings = RedisSettings()
embeddings_settings = EmbeddingSettings()
txt_speech_settings = TxtSpeechsettings()
llm_settings = LLMsettings()
hf_hub_settings = HuggingFaceHubSettings()
aws_settings = AWSSettings()

WIKI_API_SEARCH_URL = "https://en.wikipedia.org/w/rest.php/v1/search/page?q={}&limit=4"
redis_url = f"redis://{redis_settings.redis_host}:{redis_settings.redis_port}"
wiki_wiki = wikipediaapi.Wikipedia("MyProjectName (merlin@example.com)", "en")
ner_model = InferenceClient(token=hf_hub_settings.HuggingFacehub_api_token)


@lru_cache
def load_embeddings_model():
    """
    A description of the entire function, its parameters, and its return types.
    """
    return HuggingFaceBgeEmbeddings(model_name=embeddings_settings.embedding_model)


embeddings = load_embeddings_model()


class SupportedMediaFileType(StrEnum):
    """
    Supported media file types for downloading and converting
    """

    MP4 = ".mp4"
    WAV = ".wav"


@dataclass
class BaseMedia:
    """
    Dataclass for storing metadata about a media file

    location: file location on disk
    name: name of the file
    size: size of the file in bytes
    duration: duration of the media file in seconds
    file_type: file type of the media file (e.g. "mp4", "wav")
    """

    location: str
    name: str
    size: int
    duration: int
    file_type: SupportedMediaFileType


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


@dataclass
class Audio(BaseMedia):
    def __str__(self) -> str:
        return f"audio_{self.name}.{self.file_type}"


@dataclass
class Video(BaseMedia):
    def __str__(self) -> str:
        return f"video_{self.name}.{self.file_type}"


def generate_presigned_url(object_key: str, expiration: int = 3600):
    """
    Generates a presigned URL for uploading an object to an S3 bucket.

    Parameters:
        object_key (str): The key of the object to be uploaded.
        expiration (int, optional): The expiration time of the presigned URL in seconds. Defaults to 3600.

    Returns:
        str: The generated presigned URL for uploading the object.
        None: If the AWS credentials are not available.
    """
    aws_client = boto3.client(
        "s3",
        aws_access_key_id=aws_settings.aws_access_key,
        aws_secret_access_key=aws_settings.aws_secret_key,
    )
    try:
        url = aws_client.generate_presigned_url(
            "put_object",
            Params={"Bucket": aws_settings.s3_bucket, "Key": object_key},
            ExpiresIn=expiration,
        )
        return url
    except NoCredentialsError:
        print("Credentials not available")
        return None


def upload_file_to_s3(url, file_content):
    """
    Uploads file content to a specified S3 URL using PUT request.

    Parameters:
    url (str): The S3 URL to upload the file to.
    file_content (bytes): The content of the file to upload.

    Returns:
    None
    """
    response = requests.put(url, data=file_content, timeout=60)

    # Check if the upload was successful
    if response.status_code == 200:
        print("File uploaded successfully.")
    else:
        print(f"Failed to upload file. Status code: {response.status_code}")
        print(response.content)


def get_videos_from_subreddit():
    """
    Get videos from a specific subreddit using the PRAW library.

    Returns:
    list: A list of dictionaries containing video information like title, URL, and author.
    """
    reddit = praw.Reddit(
        client_id=reddit_settings.reddit_client_id,
        client_secret=reddit_settings.reddit_client_secret,
        user_agent="vidoe_bot",
    )

    # Get the subreddit instance
    subreddit = reddit.subreddit("oddlysatisfying")

    # Get video submissions
    videos = []
    for submission in subreddit.hot(limit=40):
        if (
            submission.secure_media is not None
            and submission.secure_media.get("reddit_video") is not None
        ):
            video_data = submission.secure_media["reddit_video"]
            duration = video_data.get("duration", 0)
            height = video_data.get("height", 0)
            width = video_data.get("width", 0)
            scrubber_media_url = video_data.get("fallback_url")

            if (
                20 <= duration <= 60
                and height >= 1000
                and width >= 1000
                and scrubber_media_url
            ):
                videos.append(
                    {
                        "title": submission.title,
                        "url": scrubber_media_url,
                        "author": submission.author.name,
                    }
                )
    return random.choice(videos)


def convert_video(video_file: str, audio_file: str):

    ffmpeg = (
        FFmpeg()
        .option("y")
        .input(video_file, stream_loop=-1)
        .input(audio_file)
        .output(
            "output.mp4",
            options={"codec:a": "libmp3lame", "filter:v": "scale=-1:1080"},
            map=["0:v:0", "1:a:0"],
            shortest=None,
        )
    )

    ffmpeg.execute()


def convert_text_to_audio(text: str):
    """
    Converts the given text to audio using the IBM Watson Text to Speech service.

    Args:
        text (str): The text to be converted to audio.

    Returns:
        None

    Raises:
        None
    """
    authenticator = IAMAuthenticator(apikey=txt_speech_settings.ibm_api_key)

    txt_speech = TextToSpeechV1(authenticator=authenticator)
    txt_speech.set_service_url(service_url=txt_speech_settings.ibm_url)
    url = generate_presigned_url(object_key="hello_world.wav")
    print(url)
    # with open("hello.wav", "wb") as wav_file:
    # Read binary data from the WAV file
    upload_file_to_s3(
        url=url,
        file_content=txt_speech.synthesize(
            text,
            accept="audio/wav",
            voice="en-US_HenryV3Voice",
            pitch_percentage=25,
        )
        .get_result()
        .content,
    )


def open_prompt_txt(file: str) -> str:
    """
    Reads and returns the content of the file 'prompt.txt' as a string.
    """
    with open(file, "r", encoding="utf-8") as f:
        return f.read()


def extract_answer(llm_output):
    """
    A function that extracts an answer from the given LLM output. It searches for a pattern that matches '**True**\n\nThe'
    This function takes in the LLM output as a string and splits it by newline.
    It then searches for a pattern that matches 'True' or 'False' at the beginning of the first line.
    The function returns the first match found.
    """
    llm_output_list = llm_output.split("\n")
    print(llm_output_list)
    result_match = re.findall(r"(True|False)", llm_output_list[0])
    return result_match[0]


def check_user_prompt(text: str, valid_documents: List[Document]):
    """
    A function that checks a user prompt against a list of documents and returns a question generated based on the prompt and documents.

    Parameters:
    - prompt (str): The user prompt to be checked.
    - documents (List[Document]): A list of Document objects to compare against the user prompt.

    Returns:
    - str: The question generated based on the user prompt and documents.
    """

    model = ChatDeepInfra(
        deepinfra_api_token=llm_settings.deepinfra_api_key,
        model="google/gemma-1.1-7b-it",
        max_tokens=5,
        temperature=0.2,
        top_p=0.2,
        top_k=10,
    )
    system_message = open_prompt_txt("openai_prompt.txt")
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=f"user text: {text} \n docuemnts: {valid_documents}"),
    ]
    llm_output = model.invoke(messages).content
    return extract_answer(llm_output=llm_output)


def generate_story(user_prompt: str, context_documents: List[Document]):
    """
    Generates a story question based on the user prompt and context documents.

    Parameters:
        user_prompt (str): The user prompt for generating the story question.
        context_documents (List[Document]): A list of documents providing context for the story.

    Returns:
        str: The generated story question.
    """

    presence_penalty = AI21PenaltyData(scale=4.9)
    frequency_penalty = AI21PenaltyData(scale=4)
    llm = AI21(
        model="j2-mid",
        ai21_api_key=llm_settings.ai21_api_key,
        maxTokens=2000,
        presencePenalty=presence_penalty,
        minTokens=100,
        frequencyPenalty=frequency_penalty,
    )

    prompt_template = open_prompt_txt("prompt.txt")
    final_prompt = PromptTemplate(
        input_variables=["documents", "user"],
        template=prompt_template,
    )
    chain = LLMChain(llm=llm, prompt=final_prompt)
    question = chain.invoke({"user": user_prompt, "documents": context_documents})
    return question["text"]


def return_ner_tokens(text: str):
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


def wiki_search(query: str):
    """
    Searches for a given query on the Wikipedia API and returns a list of page keys.

    Parameters:
        query (str): The search query to be used for the Wikipedia API.

    Returns:
        list: A list of page keys retrieved from the Wikipedia API response.

    Raises:
        None

    Examples:
        >>> wiki_search("Python")
        ['/wiki/Python', '/wiki/Python_(programming_language)', '/wiki/Python_(film)']
    """
    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "titles": query,
    }

    # Send the API request
    response = requests.get(
        WIKI_API_SEARCH_URL.format(query), params=params, timeout=60
    )

    if response.status_code == 200:
        # Parse the JSON data
        data = response.json()

        # Extract the page content from the response
        pages = data["pages"]
        return [page["key"].lower() for page in pages]

    else:
        print("Failed to retrieve page content.")


def get_page_content(title: str) -> WikiPage:
    """
    A function that retrieves the content of a Wikipedia page based on the provided title.

    Args:
        title (str): The title of the Wikipedia page to retrieve.

    Returns:
        WikiPage: An instance of WikiPage containing the title and text of the Wikipedia page.
    """
    return WikiPage(page_title=title, text=wiki_wiki.page(title=title).text)


def chunk_and_save(pages: List[WikiPage]):
    """
    Splits the text of each WikiPage in the given list into smaller chunks using the RecursiveCharacterTextSplitter.

    Args:
        pages (List[WikiPage]): A list of WikiPage objects containing the text to be split.

    Returns:
        List[Redis]: A list of Redis objects created from the split text of each WikiPage.
    """

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=250)
    page_splits = [
        WikiPage(page_title=page.page_title, text=text_splitter.split_text(page.text))
        for page in pages
        if page.text != ""
    ]
    result = [
        Redis.from_texts(
            page.text, embeddings, redis_url=redis_url, index_name=page.page_title
        )
        for page in page_splits
    ]
    return len(result)


def return_documents(user_prompt: str, *, index_names: List[str]) -> List[Document]:
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
                redis_url=redis_url, embedding=embeddings, index_name=index_name.lower()
            ),
            search_kwargs={"k": 5, "distance_threshold": None},
        ).invoke(user_prompt)
        for index_name in index_names
    ]


# def main():
#     prompt = "Write on how JFK's father actuallly wanted his brother to become president and not him"
#     documents = return_documents(
#         prompt,
#         index_names=[
#             "john_f._kennedy",
#             "assassination_of_john_f._kennedy",
#             "jfk_(film)",
#         ],
#     )
#     story = get_story(user_prompt=prompt, context_documents=documents)
#     audio = convert_text_to_audio(text=story)


# tokens = return_ner_tokens(prompt)
# print(tokens)
# search = [wiki_search(token) for token in tokens]
# print(search)
# contents = [get_page_content(title) for titles in search for title in titles]
# # print([i for i in content])
# chunk_and_save(contents)

# #


# # print(documents)

# print(check_user_prompt(text=prompt, valid_documents=documents))
# # print(documents)

# print(convert_text_to_audio("he"))
print(convert_text_to_audio(text="hello"))
