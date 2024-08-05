from typing import Protocol


class FileReader(Protocol):

    file_path: str
    files_name: str
    file_type: str

    def read(self) -> str: ...

