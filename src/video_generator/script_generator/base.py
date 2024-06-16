from typing import List, Protocol

from langchain.cache import RedisCache
from pydantic import BaseModel, Field, model_validator
from langchain.schema import Document
from langchain_core.messages import HumanMessage, SystemMessage

from video_generator.settings import LLMSettings
from video_generator.prompts_manager import get_prompt_manager, TextReader

def extract_answer(llm_output):
    """
    A function that extracts an answer from the given LLM output. It searches for a pattern that matches '**True**\n\nThe'
    This function takes in the LLM output as a string and splits it by newline.
    It then searches for a pattern that matches 'True' or 'False' at the beginning of the first line.
    The function returns the first match found.
    """
    llm_output_list = llm_output.split("\n")
    result_match = re.findall(r"(True|False)", llm_output_list[0])
    if result_match[0] == "True":
        return True
    elif result_match[0] == "False":
        return False
    else:
        return None


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
                HumanMessage(content=f"user text: {prompt} \n docuemnts: {documents}"),
            ]
            result = extract_answer(self.validation_model.invoke(messages).content)
        else:
            result = extract_answer(
                self.validation_model.invoke({"user": prompt, "documents": documents})
            )

        if not result:
            return False
        return True

    def generate(self):
        pass


def validate_user_prompt(
    text: str,
    valid_documents: List[Document],
    model: LLm,
) -> bool:
    """
    A function that checks a user prompt against a list of documents and returns a question generated based on the prompt and documents.

    Parameters:
    - prompt (str): The user prompt to be checked.
    - documents (List[Document]): A list of Document objects to compare against the user prompt.

    Returns:
    - str: The question generated based on the user prompt and documents.
    """
    logger.info("Checking user prompt against documents")
    text_reader = TextReader(
        file_path="<PUT FILE PATH HERE>"
    )
    return model.validate(documents=valid_documents, prompt=text, system_message=text_reader.read())


class ScriptBase(BaseModel):

    prompt: str
    documents: List[Document]
    llm_settings: LLMSettings
    indexs: List[str]
    model: LLm

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
        logger.info("Checking user prompt against documents")
        text_reader = TextReader(
            file_path="<PUT FILE PATH HERE>"
        )
        return self.model.validate(documents=self.ocuments, prompt=self.prompt, system_message=text_reader.read())

    def generate_story(
        self, user_prompt: str, context_documents: List[Document]
    ) -> Story:
        """
        Generates a story based on the user prompt and context documents.

        Parameters:
            user_prompt (str): The user prompt for generating the story.
            context_documents (List[Document]): A list of documents providing context for the story.

        Returns:
            str: The generated story.
        """
        logger.info("Generating story")
        presence_penalty = AI21PenaltyData(scale=4.9)
        frequency_penalty = AI21PenaltyData(scale=4)
        llm = AI21(
            model="j2-mid",
            ai21_api_key=llm_settings.ai21_api_key,
            maxTokens=120,
            presencePenalty=presence_penalty,
            minTokens=80,
            frequencyPenalty=frequency_penalty,
            temperature=0.5,
            topP=0.2,
        )

        prompt_template = open_prompt_txt("../prompts/prompt.txt")
        final_prompt = PromptTemplate(
            input_variables=["documents", "user"],
            template=prompt_template,
        )
        chain = LLMChain(llm=llm, prompt=final_prompt)
        question = chain.invoke({"user": user_prompt, "documents": context_documents})
        logger.info("Story generated successfully")
        return Story(prompt=user_prompt, text=question["text"])
