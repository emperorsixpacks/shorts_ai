from typing import Protocol


class FileReader(Protocol):

    files_name: str
    file_type: str

    def read_from_url(self, session) -> str: ...

    def read_from_path(self) -> str: ...
