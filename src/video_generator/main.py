import re
import random
import logging
import time
from typing import List, Dict
from tempfile import NamedTemporaryFile
import itertools


import requests
import wikipediaapi
import praw
import ibm_boto3

from huggingface_hub import InferenceClient
from redisvl.index import SearchIndex

from langchain_community.llms.ai21 import AI21, AI21PenaltyData
from langchain_community.chat_models.deepinfra import ChatDeepInfra
from langchain_core.messages import HumanMessage, SystemMessage

from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores.redis import RedisVectorStoreRetriever, Redis

from video_generator.settings import (
    RedditSettings,
    RedisSettings,
    EmbeddingSettings,
    LLMSettings,
    HuggingFaceHubSettings,
    BucketSettings,
)

from video_generator.utils import MediaFile, Story, WikiPage, MediaFileType, upload_file_to_s3

from video_generator.py_ffmpeg.main import PyFFmpeg, InputFile
from video_generator.ass_parser.main import Transcript, Style, Dialogue, Section, PyAss, Format


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("logs/app.log")
file_handler.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)


reddit_settings = RedditSettings()
redis_settings = RedisSettings()
embeddings_settings = EmbeddingSettings()
llm_settings = LLMSettings()
hf_hub_settings = HuggingFaceHubSettings()
# bucket_settings = AWSSettings()

WIKI_API_SEARCH_URL = "https://en.wikipedia.org/w/rest.php/v1/search/page?q={}&limit=4"
redis_url = f"redis://{redis_settings.redis_host}:{redis_settings.redis_port}"
wiki_wiki = wikipediaapi.Wikipedia("MyProjectName (merlin@example.com)", "en")
ner_model = InferenceClient(token=hf_hub_settings.HuggingFacehub_api_token)


# aws_s3_client = boto3.client(
#     "s3",
#     aws_access_key_id=bucket_settings.aws_access_key,
#     aws_secret_access_key=bucket_settings.aws_secret_key,
# )

# aws_polly_client = boto3.client(
#     "polly",
#     aws_access_key_id=bucket_settings.aws_access_key,
#     aws_secret_access_key=bucket_settings.aws_secret_key,
#     region_name="us-west-2",
# )
# aws_transcribe_client = boto3.client(
#     "transcribe",
#     aws_access_key_id=bucket_settings.aws_access_key,
#     aws_secret_access_key=bucket_settings.aws_secret_key,
#     region_name="us-east-1",
# )




# embeddings = load_embeddings_model()


# def get_videos_from_subreddit(number_of_videos: int = 4):
#     """
#     Get videos from a specific subreddit using the PRAW library.

#     Returns:
#     list: A list of dictionaries containing video information like title, URL, and author.
#     """

#     logger.info("Getting videos from subreddit")
#     reddit = praw.Reddit(
#         client_id=reddit_settings.reddit_client_id,
#         client_secret=reddit_settings.reddit_client_secret,
#         user_agent="vidoe_bot",
#     )

#     # Get the subreddit instance
#     subreddit = reddit.subreddit("oddlysatisfying")

#     # Get video submissions
#     videos: List[Dict[str, str]] = []
#     for submission in subreddit.hot(limit=500):
#         if (
#             submission.secure_media is not None
#             and submission.secure_media.get("reddit_video") is not None
#         ):
#             video_data = submission.secure_media["reddit_video"]
#             duration = video_data.get("duration", 0)
#             height = video_data.get("height", 0)
#             width = video_data.get("width", 0)
#             scrubber_media_url = video_data.get("fallback_url")

#             if (
#                 5 <= duration <= 12
#                 and height >= 1000
#                 and width >= 1000
#                 and scrubber_media_url
#             ):
#                 videos.append(
#                     {
#                         "title": submission.title,
#                         "url": scrubber_media_url,
#                         "author": submission.author.name,
#                     }
#                 )
#     logger.info("Videos retrieved successfully")
#     videos = random.sample(videos, k=number_of_videos + 3)
#     return [
#         MediaFile(
#             name=video.get("title"),
#             file_type=MediaFileType.VIDEO,
#             url=video.get("url"),
#             author=video.get("author"),
#         )
#         for video in videos
#     ]


def combine_video_and_audio(
    input_video_files: List[MediaFile],
    input_audio_file: MediaFile,
    output_file: MediaFile,
    transcript: str = None,
) -> MediaFile:
    """
    Combines multiple video files with a single audio file into a single MP4 using FFmpeg.

    Args:
        input_video_files: A list of dictionaries, where each dictionary represents a video file
            with a "url" key containing the video's location.
        output_file The desired file for the combined MP4 output.

    Returns:
        None
    """
    logger.info("Combining video and audio")
    input_videos = [InputFile(media_file=video) for video in input_video_files]
    input_audio = InputFile(media_file=input_audio_file)
    process = PyFFmpeg(
        video=input_videos,
        audio=input_audio,
        output_location=output_file,
        s3_client=aws_s3_client,
        bucket_settings=bucket_settings,
    )

    process = process.concatinate_video().trim_video().execute()
    logger.info("Successfully Combining video and audio")
    return process


def convert_text_to_audio(client, name: str, text: Story) -> MediaFile | None:
    """
    Converts text to audio using the TTSMP3 API.
    Parameters:
        client (Client): The AWS client.
        name (str): The name of the audio file.
        text (Story): The text to be converted to audio.
    Returns:
        MediaFile | None: The converted audio file if successful, None otherwise.
    """

    logger.info("Converting text to audio")
    media_file = MediaFile(name=name, file_type=MediaFileType.AUDIO)
    data = {"msg": text.text, "lang": "Matthew", "source": "ttsmp3"}
    audio_url = requests.post(
        "https://ttsmp3.com/makemp3_new.php", data=data, timeout=60
    ).json()["URL"]
    media_file.url = audio_url
    with NamedTemporaryFile() as tempfile:
        r = requests.get(audio_url, stream=True, timeout=60)
        if r.status_code == 200:
            with open(tempfile.name, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024):
                    f.write(chunk)
        else:
            logger.warning("Failed to download audio from TTSMP3")
            return None
        logger.info("Uploading audio to S3")
        upload = upload_file_to_s3(
            client,
            media_file=media_file,
            file_location=tempfile.name,
            bucket_settings=bucket_settings,
        )
        if not upload:
            logger.warning("Failed to upload audio to S3")
            return None
    logger.info("Audio converted successfully")
    return media_file.set_duration()


def transcribe_audio(client, *, audio: MediaFile):
    client.start_transcription_job(
        TranscriptionJobName=audio.name,
        Media={"MediaFileUri": audio.get_s3_location(settings=bucket_settings)},
        MediaFormat="wav",
        LanguageCode="en-US",
    )
    for _ in range(60):
        job = client.get_transcription_job(TranscriptionJobName=audio.name)
        job_status = job["TranscriptionJob"]["TranscriptionJobStatus"]
        if job_status in ["COMPLETED", "FAILED"]:
            print(f"Job {audio.name} is {job_status}.")
            if job_status == "COMPLETED":
                return job["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
            break
        else:
            print(f"Waiting for {audio.name}. Current status is {job_status}.")
        time.sleep(10)


# def open_prompt_txt(prompt_txt: str) -> str:
#     """
#     Reads and returns the content of the file 'prompt.txt' as a string.
#     """
#     with open(prompt_txt, "r", encoding="utf-8") as f:
#         return f.read()


# def extract_answer(llm_output):
#     """
#     A function that extracts an answer from the given LLM output. It searches for a pattern that matches '**True**\n\nThe'
#     This function takes in the LLM output as a string and splits it by newline.
#     It then searches for a pattern that matches 'True' or 'False' at the beginning of the first line.
#     The function returns the first match found.
#     """
#     llm_output_list = llm_output.split("\n")
#     result_match = re.findall(r"(True|False)", llm_output_list[0])
#     if result_match[0] == "True":
#         return True
#     elif result_match[0] == "False":
#         return False
#     else:
#         return None


# def check_user_prompt(text: str, valid_documents: List[Document]) -> bool:
#     """
#     A function that checks a user prompt against a list of documents and returns a question generated based on the prompt and documents.

#     Parameters:
#     - prompt (str): The user prompt to be checked.
#     - documents (List[Document]): A list of Document objects to compare against the user prompt.

#     Returns:
#     - str: The question generated based on the user prompt and documents.
#     """
#     logger.info("Checking user prompt against documents")
#     model = ChatDeepInfra(
#         deepinfra_api_token=llm_settings.deepinfra_api_key,
#         model="google/gemma-1.1-7b-it",
#         max_tokens=5,
#         temperature=0.2,
#         top_p=0.2,
#         top_k=10,
#     )
#     system_message = open_prompt_txt("../prompts/validation_prompt.txt")
#     messages = [
#         SystemMessage(content=system_message),
#         HumanMessage(content=f"user text: {text} \n docuemnts: {valid_documents}"),
#     ]
#     llm_output = model.invoke(messages).content
#     return extract_answer(llm_output=llm_output)


# def generate_story(user_prompt: str, context_documents: List[Document]) -> Story:
#     """
#     Generates a story based on the user prompt and context documents.

#     Parameters:
#         user_prompt (str): The user prompt for generating the story.
#         context_documents (List[Document]): A list of documents providing context for the story.

#     Returns:
#         str: The generated story.
#     """
#     logger.info("Generating story")
#     presence_penalty = AI21PenaltyData(scale=4.9)
#     frequency_penalty = AI21PenaltyData(scale=4)
#     llm = AI21(
#         model="j2-mid",
#         ai21_api_key=llm_settings.ai21_api_key,
#         maxTokens=120,
#         presencePenalty=presence_penalty,
#         minTokens=80,
#         frequencyPenalty=frequency_penalty,
#         temperature=0.5,
#         topP=0.2,
#     )

#     prompt_template = open_prompt_txt("../prompts/prompt.txt")
#     final_prompt = PromptTemplate(
#         input_variables=["documents", "user"],
#         template=prompt_template,
#     )
#     chain = LLMChain(llm=llm, prompt=final_prompt)
#     question = chain.invoke({"user": user_prompt, "documents": context_documents})
#     logger.info("Story generated successfully")
#     return Story(prompt=user_prompt, text=question["text"])


# def extract_entities(text: str):
#     """
#     A function that returns named entity recognition (NER) tokens from the given text.

#     Parameters:
#     - text (str): The input text for which NER tokens need to be extracted.

#     Returns:
#     - list: A list of NER tokens extracted from the input text.
#     """
#     result = ner_model.token_classification(
#         text=text, model=hf_hub_settings.ner_repo_id
#     )
#     return [i["word"].strip() for i in result]


# def wiki_search(query: str):
#     """
#     Searches for a given query on the Wikipedia API and returns a list of page keys.

#     Parameters:
#         query (str): The search query to be used for the Wikipedia API.

#     Returns:
#         list: A list of page keys extracted from the response of the Wikipedia API.
#     """

#     params = {
#         "action": "query",
#         "format": "json",
#         "prop": "revisions",
#         "rvprop": "content",
#         "rvslots": "main",
#         "titles": query,
#     }

#     # Send the API request
#     response = requests.get(
#         WIKI_API_SEARCH_URL.format(query), params=params, timeout=60
#     )

#     if response.status_code == 200:
#         # Parse the JSON data
#         data = response.json()

#         # Extract the page content from the response
#         pages = data["pages"]
#         return [page["key"].lower() for page in pages]

#     else:
#         print("Failed to retrieve page content.")


# def get_page_content(title: str) -> WikiPage:
#     """
#     A function that retrieves the content of a Wikipedia page based on the provided title.

#     Args:
#         title (str): The title of the Wikipedia page to retrieve.

#     Returns:
#         WikiPage: An instance of WikiPage containing the title and text of the Wikipedia page.
#     """
#     return WikiPage(page_title=title, text=wiki_wiki.page(title=title).text)


# def chunk_and_save(page: WikiPage):
#     """
#     Splits the text of each WikiPage in the given list into smaller chunks using the RecursiveCharacterTextSplitter.

#     Args:
#         pages (List[WikiPage]): A list of WikiPage objects containing the text to be split.

#     Returns:
#         List[Redis]: A list of Redis objects created from the split text of each WikiPage.
#     """

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
#     page_splits = (
#         WikiPage(page_title=page.page_title, text=text_splitter.split_text(page.text))
#         if page.text != ""
#         else None
#     )
#     if page_splits is None:
#         return False
#     Redis.from_texts(
#         page_splits.text,
#         embeddings,
#         redis_url=redis_url,
#         index_name=page_splits.page_title,
#     )
#     return True


# def return_documents(user_prompt: str, *, index_names: List[str]) -> List[Document]:
#     """
#     Generates a list of Document objects by invoking the RedisVectorStoreRetriever with the given user prompt and index names.

#     Args:
#         user_prompt (str): The user prompt to be passed to the RedisVectorStoreRetriever.
#         index_names (List[str]): The list of index names to be used by the RedisVectorStoreRetriever.

#     Returns:
#         List[Document]: The list of Document objects generated by the RedisVectorStoreRetriever.
#     """
#     return [
#         RedisVectorStoreRetriever(
#             vectorstore=Redis(
#                 redis_url=redis_url, embedding=embeddings, index_name=index_name.lower()
#             ),
#             search_kwargs={"k": 5, "distance_threshold": None},
#         ).invoke(user_prompt)
#         for index_name in index_names
#     ]


# def check_existing_redis_index(index_name: str) -> bool:
#     """
#     Checks if an index with the given name exists in Redis.

#     Args:
#         index_name (str): The name of the index to check.

#     Returns:
#         bool: True if the index exists, False otherwise.
#     """
#     index = SearchIndex.from_dict(
#         {
#             "index": {"name": index_name, "prefix": "docs", "storage_type": "hash"},
#             "fields": [
#                 {
#                     "name": "content_vector",
#                     "type": "vector",
#                     "attrs": {
#                         "dims": 1536,
#                         "algorithm": "FLAT",
#                         "datatype": "FLOAT32",
#                         "distance_metric": "COSINE",
#                     },
#                 }
#             ],
#         }
#     )

#     return index.connect(redis_url=redis_url).exists()


def main():
    prompt = "Write on how nigeria got her name"
    entities = extract_entities(text=prompt)
    tokens = [wiki_search(entity) for entity in entities]
    tokens = list(itertools.chain(*tokens))
    indexes = []
    for token in tokens:
        logger.info("Checking existing redis index")
        if not check_existing_redis_index(token):
            logger.info("Creating new redis index: %s", token)
            logger.info("Getting page content")
            page = get_page_content(token)
            logger.info("Chunking and saving text")
            chunk = chunk_and_save(page)
            if not chunk:
                continue
        indexes.append(token)
    logger.info("Done checking and creating redis indexes")

    documents = return_documents(
        prompt,
        index_names=indexes,
    )
    checked_prompt = check_user_prompt(text=prompt, valid_documents=documents)
    if not checked_prompt:
        print("mate, this never happened or I am to old to remember 🥲")
        return
    story = generate_story(user_prompt=prompt, context_documents=documents)
    print(story.text)
    # audio_file = convert_text_to_audio(aws_s3_client, text=story, name=prompt
    # )
    # number_of_videos = int(audio_file.duration // 10)
    # videos = get_videos_from_subreddit(number_of_videos=number_of_videos)
    # video_file = MediaFile(name=prompt, file_type=MediaFileType.VIDEO)
    # trascript = transcribe_audio(aws_transcribe_client, audio=audio_file)

    # transcripts = Transcript.open_transcript_json(file_path=trascript)
    # ordering_format = Format(
    #     fields=[
    #         "Name",
    #         "Fontname",
    #         "Fontsize",
    #         "PrimaryColour",
    #         "SecondaryColour",
    #         "OutlineColour",
    #         "BackgroundColor",
    #     ]
    # )

    # style = Style(order_format=ordering_format)
    # dialogue_format = Format(
    #     fields=[
    #         "Layer",
    #         "Start",
    #         "End",
    #         "Style",
    #         "MarginL",
    #         "MarginR",
    #         "MarginV",
    #         "Text",
    #     ]
    # )

    # dialogues = Dialogue.from_list(
    #     transcripts,
    #     dialogue_style=style,
    #     order_format=dialogue_format,
    #     focus_style=r"{\xbord20}{\ybord10}{\3c&HD4AF37&\1c&HFFFFFF&}",
    #     MarginL=1,
    #     MarginR=1,
    #     MarginV=1
    # )

    # event_fields = [["Format", dialogue_format.return_fields_str()]]

    # for dialogue in dialogues:
    #     event_fields.append(dialogue)
    # events = Section(
    #     title="Events",
    #     fields=event_fields,
    # )
    # with NamedTemporaryFile(suffix=".ass") as subtitle:
    #     print("writing to file")
    #     with PyAss(subtitle.name, "w", sections=[events]) as ass:
    #         ass.write()

    #     combine_video_and_audio(
    #         input_audio_file=audio_file,
    #         input_video_files=videos,
    #         output_file=video_file,
    #         transcript=subtitle.name,
    #     )


if __name__ == "__main__":
    # main()
    from ibm_botocore.client import Config, ClientError

    ibm_cloud_writer_client = ibm_boto3.resource(
        service_name="s3",
        ibm_api_key_id="kKa3P-xl9gjjECYCAF0872FnvW03wIiH9fN7htuybP4o",
        # ibm_service_instance_id="crn:v1:bluemix:public:cloud-object-storage:global:a/0558f2d90ab4478793533d5202ce1691:acd3ab04-4d7c-4916-8937-6aeef9765913:bucket:shortsai",
        config=Config(signature_version="oauth"),
        endpoint_url="https://s3.us-south.cloud-object-storage.appdomain.cloud",
    )
    # print(ibm_cloud_writer_client)
    # # bucket = ibm_cloud_writer_client.Bucket("shortsai")
    # print(bucket.Object("hello.txt").get())
    # s3.Object('mybucket', 'hello.txt').download_file('/tmp/hello.txt')
