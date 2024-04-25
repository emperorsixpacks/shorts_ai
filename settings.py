"""
Settings for api credentials and logins.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseConfig(BaseSettings):
    """
    Base class for settings to inherit from.

    This class loads environment variables from a .env file and ignores any
    extra variables that are not defined in the pydantic model.
    """

    model_config = SettingsConfigDict(
        env_file=(".env.prod", ".env"), env_file_encoding="utf-8", extra="ignore"
    )


class RedditSettings(BaseConfig):
    """
    Settings for Reddit API credentials.

    Attributes:
        reddit_client_id (str): Client ID of the Reddit app.
        reddit_client_secret (str): Client secret of the Reddit app.
    """

    reddit_client_id: str = None
    reddit_client_secret: str = None


class RedisSettings(BaseConfig):
    """
    Settings for Redis connection.

    Attributes:
        redis_host (str): Host of Redis server. Defaults to localhost.
        redis_port (int): Port of Redis server.
    """

    redis_host: str = "localhost"
    redis_port: int = None


class HuggingFaceHubSettings(BaseConfig):
    """
    Settings for Hugging Face Hub API credentials.

    Attributes:
        HuggingFacehub_api_token (str): API token for the Hugging Face Hub.
        NER_REPO_ID (str): The ID of the pre-trained NER model to be used with the Hugging Face Hub.
    """

    HuggingFacehub_api_token: str = None
    ner_repo_id: str = "jean-baptiste/roberta-large-ner-english"


class EmbeddingSettings(BaseConfig):
    """
    Settings for the pre-trained embedding model to be used with the LLM.

    Attributes:
        embedding_model (str): The name of the pre-trained embedding model.
    """

    embedding_model: str = None


class LLMSettings(BaseConfig):
    """
    Settings for the ASSISTANT Language Model (LLM) API credentials.

    Attributes:
        ai21_api_key (str): API key for the ASSISTANT LLM API.
        deepinfra_api_key (str): API key for the Deep Infra Chat Model API.
    """

    ai21_api_key: str = None
    deepinfra_api_key: str = None


class TxtSpeechSettings(BaseConfig):
    """
    Settings for IBM Watson Text to Speech API credentials.

    Attributes:
        ibm_api_key (str): API key for the IBM Watson Text to Speech API.
        ibm_link (str): Link to the IBM Watson Text to Speech API.
    """

    ibm_api_key: str = None
    ibm_url: str = None
    ibm_service_name: str = None


class AWSSettings(BaseConfig):
    """
    Settings for AWS credentials.

    Attributes:
        aws_secret_key (str): AWS secret key.
        aws_access_key (str): AWS access key.
    """
    fastly_url: str = None
    aws_secret_key: str = None
    aws_access_key: str = None
    s3_bucket: str = None


