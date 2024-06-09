from dataclasses import dataclass, field

class BaseReader:
    file_name = field(init=False)
    file_type = field(init=False)

    def 

class TextReader(PromptsBase):
     def read_from_url(self):
        try:
            responce = self.session.get_content()
        except TimeoutError as e:
            raise ServerTimeOutError(location=self.location) from e
        else:
            self.contents = responce.text

    def read_from_path(self):
        with open(self.location, "r", encoding="utf-8") as f:
            self.contents = f.read()