import requests
from enum import StrEnum
from pydantic import BaseModel, AnyUrl, Field

def create_session() -> requests.Session:
    return requests.Session()

class Session(BaseModel):
    url: AnyUrl
    session: requests.Session = Field(default_factory=create_session)

    def send_requst(self, request_method="GET") -> int:
    
        request_method = request_method.upper()
        try:
            request = requests.Request(request_method, self.url)
            prepped = self.session.prepare_request(request)
            responce = self.session.send(prepped, timeout=30)
            responce.raise_for_status()
        except TimeoutError as e:
            raise ServerTimeOutError(location=self.url) from e
        else:
            return responce

    # def is_valid_url(self):