from __future__ import annotations
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Self
from dataclasses import dataclass, field
from exceptions import UnsupportedFileFormat

from pydantic import BaseModel, Field, model_validator, ConfigDict
import numpy as np


# # Define your list
# my_list = [1, 2, 3, 4, 5]

# # Convert the list to a NumPy array
# arr = np.array(my_list)

# # Split the array into two parts
# split_array = np.array_split(arr, 2)

# # Convert the split arrays back to lists
# split_list = [sub_arr.tolist() for sub_arr in split_array]


@dataclass
class Transcript:
    content: str = field(default=None)
    _start_time: float = field(default=None)  # in seconds
    _end_time: float = field(default=None)  # in seconds

    @property
    def start_time(self) -> int:
        """
        Returns the start time of the object in milliseconds.

        :return: The start time of the object as an integer.
        :rtype: int
        """
        return int(self._start_time)

    @start_time.setter
    def start_time(self, value: int):
        """
        Set the start time of the object.

        Args:
            value (int): The start time value to be set.

        Returns:
            None
        """
        self._start_time = value * 1000

    @property
    def end_time(self) -> int:
        """
        Returns the end time of the object in milliseconds.

        :return: The end time of the object as an integer.
        :rtype: int
        """
        return int(self._end_time)

    @end_time.setter
    def end_time(self, value: int):
        """
        Set the end time of the object.

        Args:
            value (int): The end time value to be set.

        Returns:
            None
        """
        self._end_time = value * 1000

    def __str__(self):
        return self.content

    def __dict__(self):
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "content": self.content,
        }

    @classmethod
    def open_transcript_json(cls, file_path: str) -> List[Transcript]:
        """
        Opens the transcript file and returns a list of Transcript objects.

        Args:
            file_path (str): The path to the transcript file.

        Returns:
            List[Transcript]: A list of Transcript objects.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                transcripts = []
                for item in data["results"]["items"]:
                    cleaned_data = {
                        "_start_time": float(item.get("start_time", 0)),
                        "_end_time": float(item.get("end_time", 0)),
                        "content": item["alternatives"][0]["content"],
                    }
                    transcripts.append(Transcript(**cleaned_data))
                return transcripts
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {file_path}") from e


class Format(BaseModel):
    name: str = Field(default="Format", init=False, frozen=True)
    fields: List[str] = Field(default=None)

    def return_fields_str(self):
        """
        Returns a string containing the values of all the fields in the current object, joined by commas.

        :return: A string containing the values of all the fields in the current object, joined by commas.
        :rtype: str
        """
        fields = ",".join(self.fields)
        return fields


class Entry(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    title: str = Field(default="Default", exclude=True, frozen=True, init=False)
    ordering_format: Format = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def validate_fields(self) -> Self:
        """
        Validates the fields of the model after they have been set.

        This function is a model validator decorated with `@model_validator(mode="after")`.
        It is called after the fields of the model have been set.

        Parameters:
            self (Self): The instance of the model.

        Returns:
            Self: The instance of the model with validated fields.

        Raises:
            ValueError: If the field at a specific index does not match the corresponding key in `self.ordering_format.fields`.
        """
        key_extras = self.model_dump(by_alias=True).keys()
        print(self.model_dump(by_alias=True))
        for i, key in enumerate(key_extras):
            if self.ordering_format.fields[i] != key:
                raise ValueError(
                    f"{key} doesn't match index in {self.ordering_format.fields}"
                )

        return self

    def return_entry_fields(self):
        """
        Returns a string containing the values of all the fields in the current object, joined by commas.

        :return: A string containing the values of all the fields in the current object, joined by commas.
        :rtype: str
        """

        return ",".join(self.model_dump().values())


class Style(Entry):
    name: str = Field(default="Default", alias="Name")
    font_name: str = Field(default="Arial", alias="Fontname")
    font_size: int = Field(default=24, alias="Fontsize")
    primary_colour: str = Field(default="#FFFFFF", alias="PrimaryColour")
    secondary_colour: str = Field(default="#000000", alias="SecondaryColour")
    outline_colour: str = Field(default="#000000", alias="OutlineColour")
    background_colour: str = Field(default="#000000", alias="BackgroundColor")


class Dialogue(Entry):
    layer: str = Field(default="0", alias="Layer")
    start_time: str = Field(default="00:00:00.00", alias="Start")
    stop_time: str = Field(default="00:00:00.00", alias="End")
    style: str = Field(default=None, alias="Style")
    name: str = Field(default="Default", alias="Name")
    text: str = Field(default="This is my text", alias="Text")

    @model_validator(mode="before")
    def check_style(self) -> Self:
        """
        A model validator function that checks if the style attribute of the current object is an instance of the Style class.

        This function is decorated with `@model_validator(mode="before")` to indicate that it should be executed before other model validators.

        Parameters:
            self (object): The current object.

        Returns:
            object: The current object with the style attribute updated to its name.

        Raises:
            ValueError: If the style attribute is not an instance of the Style class.
        """
        if not isinstance(self["style"], Style):
            raise ValueError("style must be an instance of Style")
        self["style"] = self["style"].name
        return self


class Section(BaseModel):
    title: str = Field(default=None)
    fields: Dict[str, str] = Field(default=None)

    def to_ass_format(self):
        """
        Converts the object to a string representation in the ASS format.

        Returns:
            str: The object converted to the ASS format. Each key-value pair in the `fields` dictionary is converted to a line in the format "{key.capitalize()}: {value}". The lines are joined with a newline character and a trailing newline character is added.
        """
        lines = [f"[{self.title}]"]
        for key, value in self.fields.items():
            line = f"{key.capitalize()}: {value}"
            lines.append(line)

        return "\n".join(lines) + "\n\n"
        # return lines


class PyAss:

    def __init__(self, file_name, mode, *, sections: List[Section]) -> None:
        self.file_name = file_name
        self.mode = mode
        self.sections = sections
        self._io = None

    def write(self):
        sections = [section.to_ass_format() for section in self.sections]
        self._io.writelines(sections)
        return self

    def __enter__(self):
        _, extention = os.path.splitext(self.file_name)
        if extention != ".ass":
            raise UnsupportedFileFormat(file=self.file_name)
        self._io = open(self.file_name, mode="w+", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._io:
            self._io.close()


if __name__ == "__main__":
    transcripts = Transcript.open_transcript_json(
        file_path="/home/emperorsixpacks/Downloads/asrOutput(1).json"
    )
    print(transcripts)
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
    # style = Style(ordering_format=ordering_format)

    # dialogue_format = Format(fields=["Layer", "Start", "End", "Style", "Name", "Text"])
    # dialogue = Dialogue(ordering_format=dialogue_format, style=style)

    # sript_info = Section(title="Script Info", fields={"title": "Sample project"})
    # events = Section(
    #     title="Events",
    #     fields={
    #         "Format": dialogue_format.return_fields_str(),
    #         "Dialogue": dialogue.return_entry_fields(),
    #     },
    # )

    # with PyAss("test.ass", "w", sections=[sript_info, events]) as ass:
    #     ass.write()
