import requests
from enum import StrEnum
from typing import TypeVar
from pydantic import BaseModel, AnyUrl, Field

from video_generator.exceptions.sessionExceptions import (
    ServerTimeOutError,
    ResourceNotFoundError,
    ServerError
)

RequestSession = TypeVar("RequestSession", "requests.Session", None)



class RequestMethods(StrEnum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"


def create_session() -> requests.Session:
    """
    Creates a new instance of the `requests.Session` class.

    Returns:
        requests.Session: A new instance of the `requests.Session` class.
    """
    return requests.Session()


class SessionManager(BaseModel):
    url: AnyUrl
    session: RequestSession = Field(default_factory=create_session)

    def send_requst(
        self, request_method: RequestMethods = RequestMethods.DELETE
    ) -> requests.Response:
        """
        Sends a request to the specified URL using the provided request method.

        Args:
            request_method (RequestMethods): The HTTP method to use for the request. Defaults to "GET".

        Returns:
            int: The HTTP response code.

        Raises:
            ServerTimeOutError: If the request times out.

        """
        try:
            request = requests.Request(request_method.value, self.url)
            prepped = self.session.prepare_request(request)
            responce = self.session.send(prepped, timeout=30)
            responce.raise_for_status()
        except TimeoutError as e:
            raise ServerTimeOutError(location=self.url) from e

        return responce
    
    def close(self):
        """
        Closes the session by calling the `close()` method of the `session` object.

        This method is used to release any resources that the session may be holding. It is typically called when the session is no longer needed or when the program is exiting.

        Parameters:
            self (Session): The current instance of the `Session` class.

        Returns:
            None
        """
        self.session.close()


class Session(SessionManager):

    def ping(self) -> int:
        """
        Sends a HEAD request to the specified URL to check if it is valid.

        Returns:
            int: The HTTP response code.

        Raises:
            ServerTimeOutError: If the request times out.
        """
        return self.send_requst(request_method=RequestMethods.HEAD).status_code

    def get_content(self) -> str:
        """
        Sends a GET request to the specified URL and returns the response content.

        Returns:
            str: The response content.

        Raises:
            ServerTimeOutError: If the request times out.
        """
        response = self.send_requst(request_method=RequestMethods.GET)

        if response.status_code == 404:
            raise ResourceNotFoundError(location=self.url)
        if response.status_code not in (200, 404):
            raise ServerError(location=self.location, status_code=response.status_code)

        return response.text
