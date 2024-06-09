import re
from dataclasses import dataclass, field


@dataclass
class BaseReader:

    file_path: str
    file_name: str = field(init=False)
    file_type: str = field(init=False)

    def __post_init__(self):
        self.file_name, self.file_type = self.return_name_and_type()

    def return_name_and_type(self):
        """
        Returns the name and type of a file based on its file path.

        Parameters:
            self (object): The current instance of the class.

        Returns:
            tuple: A tuple containing the file name (str) and file extension (str).

        Raises:
            ValueError: If the file path is invalid.

        """
        pattern = re.compile(r"([^\/\\\s]+)\.([a-zA-Z0-9]+)$")
        match = pattern.search(self.file_path)
        if not match:
            raise ValueError(f"Invalid file path {self.file_path}")

        file_name = match.group(1)
        file_extension = match.group(2)
        return file_name, file_extension


# class TextReader(PromptsBase):
#      def read_from_url(self):
#         try:
#             responce = self.session.get_content()
#         except TimeoutError as e:
#             raise ServerTimeOutError(location=self.location) from e
#         else:
#             self.contents = responce.text

#     def read_from_path(self):
#         with open(self.location, "r", encoding="utf-8") as f:
#             self.contents = f.read()


