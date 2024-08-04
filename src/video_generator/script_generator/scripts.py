import re
import random
import asyncio
import itertools
from math import ceil
from typing import List, Protocol, Dict, Optional

from pydantic import BaseModel, model_validator, Field, ConfigDict
from langchain.schema import Document
from langchain_core.messages import HumanMessage, SystemMessage

from video_generator.settings import LLMSettings, RedditSettings
from video_generator.prompts_manager import TextReader
from video_generator.utils import get_reddit_client, MediaFile, MediaFileType


VALIDATION_PROMPT = "validation_prompt.txt"

MIN_VIDEO_LENGTH = 5
MAX_VIDEO_LENGTH = 12
MIN_VIDEO_HEIGHT = 1000
MIN_VIDEO_WIDTH = 1000
DEFAULT_SUBREDDITS = ["oddlysatisfying", "PerfectTiming", "satisfying"]
DEFAULT_NUMBER_OF_VIDEOS = 4
BOOL_DICT = {"True": True, "False": False}


def extract_answer(llm_output):
    """
    A function that extracts an answer from the given LLM output. It searches for a pattern that matches '**True**\n\nThe'
    This function takes in the LLM output as a string and splits it by newline.
    It then searches for a pattern that matches 'True' or 'False' at the beginning of the first line.
    The function returns the first match found.
    """
    llm_output_list = llm_output.split("\n")
    result_match = re.findall(r"(True|False)", llm_output_list[0])
    return BOOL_DICT.get(result_match[0], None)


class Model(Protocol):

    def __init__(self, **kwargs) -> None: ...

    def invoke(self): ...


class LLm:
    def __init__(self, validation_model: Model, gen_model: Model, use_same: bool):
        self.validation_model = validation_model
        self.gen_model = gen_model
        self.use_same: bool = use_same
        if use_same:
            self.validation_model = self.gen_model

    def validate(
        self,
        documents: List[Document],
        prompt: str,
        system_message: str,
        use_chat_model: bool = True,
    ) -> bool:
        if use_chat_model:
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=f"user text: {prompt} \n documents: {documents}"),
            ]
            result = extract_answer(self.validation_model.invoke(messages).content)
        else:
            result = extract_answer(
                self.validation_model.invoke({"user": prompt, "documents": documents})
            )

        if not result:
            return False
        return True

    def generate(self, prompt: str, documents: List[Document]):
        return self.gen_model.invoke({"user": prompt, "documents": documents})


class Script(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True)
    prompt: str
    documents: List[Document]
    llm_settings: LLMSettings
    indexs: List[str]
    model: LLm
    number_of_videos: int = Field(default=4, gt=0.0)

    @model_validator(mode="after")
    def validate_user_prompt(self):
        """
        A function that checks a user prompt against a list of documents and returns a question generated based on the prompt and documents.

        Parameters:
        - prompt (str): The user prompt to be checked.
        - documents (List[Document]): A list of Document objects to compare against the user prompt.

        Returns:
        - str: The question generated based on the user prompt and documents.
        """
        # logger.info("Checking user prompt against documents")
        text_reader = TextReader(file_path=self.prompt)  # prompt name
        return self.model.validate(
            documents=self.ocuments,
            prompt=self.prompt,
            system_message=text_reader.read(),
        )

    def generate_story(self):
        """
        Generates a story based on the user prompt and context documents.

        Parameters:
            user_prompt (str): The user prompt for generating the story.
            context_documents (List[Document]): A list of documents providing context for the story.

        Returns:
            str: The generated story.
        """
        # logger.info("Generating story")

        return self.model.generate(prompt=self.prompt, documents=self.documents)

    @classmethod
    async def _get_reddit_videos(
        cls, subreddit: str, praw_client, number_of_videos: int
    ) -> List[MediaFile]:

        def is_valid_video(
            video_data: Dict[str, Optional[float]],
            duration: float,
            height: float,
            width: float,
        ) -> bool:
            return (
                MIN_VIDEO_LENGTH <= duration <= MAX_VIDEO_LENGTH
                and height >= MIN_VIDEO_HEIGHT
                and width >= MIN_VIDEO_WIDTH
                and video_data.get("fallback_url") is not None
            )

        subreddit_instance = await praw_client.subreddit(subreddit)
        videos: List[Dict[str, str]] = []

        async for submission in subreddit_instance.hot(limit=500):
            if submission.secure_media and "reddit_video" in submission.secure_media:
                video_data = submission.secure_media["reddit_video"]
                duration = video_data.get("duration", 0)
                height = video_data.get("height", 0)
                width = video_data.get("width", 0)

                if is_valid_video(video_data, duration, height, width):
                    videos.append(
                        {
                            "title": submission.title,
                            "url": video_data.get("fallback_url"),
                            "author": submission.author,
                        }
                    )

        if not videos:
            # Handle empty list scenario, logging if necessary
            # logger.info("No valid videos found")
            return []

        # Randomly sample videos if more are available than requested
        sampled_videos = random.sample(videos, min(number_of_videos, len(videos)))

        return [
            MediaFile(
                name=video["title"],
                file_type=MediaFileType.VIDEO,
                url=video["url"],
                author=video["author"],
            )
            for video in sampled_videos
        ]

    @classmethod
    async def get_videos_from_subreddit(
        cls,
        reddit_settings: RedditSettings,
        number_of_videos: Optional[int] = None,
        subreddits: Optional[List[str]] = None,
    ):
        """
        Get videos from a specific subreddit using the PRAW library.

        Returns:
        list: A list of dictionaries containing video information like title, URL, and author.
        """

        if subreddits is None:
            subreddits = DEFAULT_SUBREDDITS

        if number_of_videos is None:
            number_of_videos = DEFAULT_NUMBER_OF_VIDEOS
        if number_of_videos <= 0:
            raise ValueError("Number of videos must be greater than zero")

        average_numeber_of_videos = (
            ceil(len(subreddits) / number_of_videos)
            if len(subreddits) != 1
            else number_of_videos
        )

        # logger.info("Getting videos from subreddit")
        praw_client = get_reddit_client(reddit_settings=reddit_settings)
        tasks = [
            cls._get_reddit_videos(
                praw_client=praw_client,
                number_of_videos=average_numeber_of_videos,
                subreddit=subreddit,
            )
            for subreddit in subreddits
        ]

        results = await asyncio.gather(*tasks)
        results = list(itertools.chain(*results))
        return random.sample(results, number_of_videos)


if __name__ == "__main__":

    settings = RedditSettings()

    asyncio.run(
        Script.get_videos_from_subreddit(reddit_settings=settings, number_of_videos=4)
    )
