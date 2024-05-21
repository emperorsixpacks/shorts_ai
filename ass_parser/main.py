from __future__ import annotations
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Self
from dataclasses import dataclass, field
import string
from exceptions import UnsupportedFileFormat

import num2words
from pydantic import BaseModel, Field, model_validator, ConfigDict
import numpy as np


@dataclass
class Transcript:
    """
    A class for storing caption or subtitle information.

    Attributes:
        text (str): The text of the caption or subtitle.
        start_time (float | int): The start time of the caption or subtitle in seconds.
        end_time (float | int): The end time of the caption or subtitle in seconds.
    """

    text: str = field(default=None)
    start_time: float | int = field(default=None)  # in seconds
    end_time: float | int = field(default=None)  # in seconds

    def __post_init__(self):
        """
        Sets the start_time and end_time attributes to their respective
        millisecond values.
        """
        self._start_time = self.start_time
        self._end_time = self.end_time

    @property
    def start_time(self) -> int:
        """
        Returns the start time of the object in milliseconds.

        :return: The start time of the object as an integer.
        :rtype: int
        """
        return self._start_time

    @start_time.setter
    def start_time(self, value: float):
        """
        Set the start time of the object.

        Args:
            value (int): The start time value to be set.

        Returns:
            None
        """
        self._start_time = int(value * 1000)

    @property
    def end_time(self) -> int:
        """
        Returns the end time of the object in milliseconds.

        :return: The end time of the object as an integer.
        :rtype: int
        """
        return self._end_time

    @end_time.setter
    def end_time(self, value: float):
        """
        Set the end time of the object.

        Args:
            value (int): The end time value to be set.

        Returns:
            None
        """
        self._end_time = int(value * 1000)

    @classmethod
    def open_transcript_json(cls, file_path: str) -> List[Dict[str, str]]:
        """
        Opens the transcript file and returns a list of Transcript objects.

        Args:
            file_path (str): The path to the transcript file.

        Returns:
            List[Transcript]: A list of Transcript objects.

        Raises:
            FileNotFoundError: If the transcript file does not exist.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                _transcripts = []
                for item in data["results"]["items"]:
                    text: str = item["alternatives"][0]["content"]
                    if text.isnumeric():
                        text = num2words.num2words(text)
                    elif text in string.punctuation:
                        continue
                    cleaned_data = {
                        "text": text.lower().strip(),
                        "start_time": float(item.get("start_time", 0)),
                        "end_time": float(item.get("end_time", 0)),
                    }
                    _transcripts.append(Transcript(**cleaned_data))
                num_sections = int(len(_transcripts) / 5)
                np_array = np.array_split(_transcripts, num_sections)
                return [array.tolist() for array in np_array]
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {file_path}") from e


class Format(BaseModel):
    """
    A class for storing the format of a SubRip file.

    The class stores the name of the format and a list of field names that are used in the format.
    """

    name: str = Field(default="Format", init=False, frozen=True)
    """The name of the format."""

    fields: List[str] = Field(default=None)
    """A list of field names that are used in the format."""

    def return_fields_str(self) -> str:
        """
        Returns a string containing the values of all the fields in the current object, joined by commas.

        :return: A string containing the values of all the fields in the current object, joined by commas.
        :rtype: str
        """
        fields = ",".join(self.fields)
        return fields


class Entry(BaseModel):
    """
    A class for storing the fields of a SubRip file entry.

    The class stores the fields of the entry in a dictionary and provides functionality for accessing and manipulating the fields.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)
    """The configuration of the model."""

    title: str = Field(default="Default", exclude=True, frozen=True, init=False)
    """The title of the entry."""

    order_format: Format
    """The format of the entry."""

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
        fields = self.model_dump(by_alias=True).keys()
        if self.order_format is None:
            raise ValueError("order_format cannot be None")
        for _, key in enumerate(fields):
            if key not in self.order_format.fields:
                raise ValueError(
                    f"{key} doesn't exist in {self.order_format.fields}"
                )

        return self

    def _reorder_and_return_fields(self) -> Dict:
        fields = self.model_dump(by_alias=True)

        return {key: fields[key] for key in self.order_format.fields}

    def return_entry_str(self) -> str:
        """
        Returns a string containing the values of all the fields in the current object, joined by commas.

        :return: A string containing the values of all the fields in the current object, joined by commas.
        :rtype: str
        """

        return ",".join(self.model_dump().values())


class Style(Entry):
    """
    A class for storing the style of a SubRip file.

    The class stores the name of the style and the values of its attributes.
    """

    name: str = Field(default="Default", alias="Name")
    """The name of the style."""

    font_name: str = Field(default="Arial", alias="Fontname")
    """The name of the font used in the style."""

    font_size: int = Field(default=24, alias="Fontsize")
    """The size of the font used in the style."""

    primary_colour: str = Field(default="#FFFFFF", alias="PrimaryColour")
    """The primary colour of the style."""

    secondary_colour: str = Field(default="#000000", alias="SecondaryColour")
    """The secondary colour of the style."""

    outline_colour: str = Field(default="#000000", alias="OutlineColour")
    """The outline colour of the style."""

    background_colour: str = Field(default="#000000", alias="BackgroundColor")
    """The background colour of the style."""

    def apply_style(self, text: str) -> str:
        """
        Applies the style to the given text.

        Args:
            text (str): The text to apply the style to.

        Returns:
            str: The text with the style applied.
        """

        return f"[{self.name}]{text}[/{self.name}]"


class Dialogue(Entry):
    """
    A class for storing the dialogue of a SubRip file.

    The class stores the start and end time of the dialogue, the name of the speaker,
    and the text of the dialogue.
    """

    layer: str = Field(default="0", alias="Layer")
    """The layer of the dialogue."""

    start_time: float | int | str = Field(default="00:00:00.00", alias="Start")
    """The start time of the dialogue in milliseconds."""

    end_time: float | int | str = Field(default="00:00:00.00", alias="End")
    """The end time of the dialogue in milliseconds."""

    style: str = Field(default=None, alias="Style")
    """The name of the style to be applied to the dialogue."""

    text: str = Field(default="This is my text", alias="Text")
    """The text of the dialogue."""

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
        if self["style"] is None:
            raise ValueError("style cannot be None")
        elif not isinstance(self["style"], Style):
            raise ValueError("style must be an instance of Style")
        self["style"] = self["style"].name
        return self

    @model_validator(mode="after")
    def return_time_formarts(self) -> Self:
        """
        A model validator function that returns the start and stop time in the format "HH:MM:SS.FFF".

        This function is decorated with `@model_validator(mode="after")` to indicate that it should be executed after other model validators.

        Parameters:
            self (object): The current object.

        Returns:
            object: The current object with the start and stop time in the format "HH:MM:SS.FFF".
        """

        if not isinstance(self.start_time, (int, float)) or not isinstance(
            self.end_time, (int, float)
        ):
            return self

        time_format = "%H:%M:%S.%f"

        start_time_delta = timedelta(milliseconds=self.start_time)
        stop_time_delta = timedelta(milliseconds=self.end_time)

        referemce_time = datetime(2024, 1, 1)

        self.start_time = (
            (referemce_time + start_time_delta).strftime(time_format).strip()[:-4]
        )
        self.end_time = (
            (referemce_time + stop_time_delta).strftime(time_format).strip()[:-4]
        )

        return self

    @classmethod
    def from_list(
        cls,
        data: List[List[Transcript]],
        style: Style = None,
        order_format: Format = None,
        focus_style: str = None,
    ) -> List[Self]:
        """
        Creates a list of Dialogue objects from a list of lists of dictionaries.

        Each item in the outer list represents a dialogue. Each item in the inner list represents a transcript.
        The function joins the text of all the transcripts in a dialogue together separated by a space.

        Args:
            data (List[List[Transcript]]): The data to create the Dialogue objects from.
            style (Style, optional): The style to be applied to all the Dialogue objects.
                Defaults to None.
            format (Format, optional): The format of the ordering of the dialogues.
                Defaults to None.

        Returns:
            List[Dialogue]: A list of Dialogue objects.
        """
        dialogues = []
        for item in data:
            text = []
            for transcript in item:

                start_time = transcript.start_time
                end_time = transcript.end_time

                focus_text = (
                    f"{focus_style} {transcript.text.upper()} {focus_style}"
                    if focus_style is not None
                    else transcript.text.upper()
                )
                dialogues.append(
                    [
                        "Dialogue",
                        cls(
                            start_time=start_time,
                            end_time=end_time,
                            text=f'{" ".join(text)} {focus_text}',
                            style=style,
                            ordering_format=order_format,
                        ).return_entry_str(),
                    ]
                )
                text.append(transcript.text)
        return dialogues


class Section(BaseModel):
    """
    Represents a section of an ASS file.

    Attributes:
        title (str): The title of the section.
        fields (Dict[str, str]): The fields in the section.
    """

    title: str = Field(default=None, description="The title of the section.")
    fields: List[List[str]] = Field(
        default=None, description="The fields in the section."
    )

    def to_ass_format(self) -> str:
        """
        Converts the object to a string representation in the ASS format.

        Returns:
            str: The object converted to the ASS format. Each key-value pair in the `fields` dictionary is converted to a line in the format "{key.capitalize()}: {value}". The lines are joined with a newline character and a trailing newline character is added.
        """
        lines = [f"[{self.title}]"]
        for item in self.fields:
            lines.append(f"{item[0].capitalize()}: {item[1]}")
        return "\n".join(lines) + "\n\n"


class PyAss:
    """
    Class for writing ASS files.

    Attributes:
        file_name (str): The file name of the ASS file.
        mode (str): The file mode of the ASS file.
        sections (List[Section]): The list of sections in the ASS file.
        _io (TextIOWrapper): The file handle for the ASS file.
    """

    def __init__(self, file_name, mode, *, sections: List[Section]) -> None:
        """
        Initializes the PyAss class.

        Args:
            file_name (str): The file name of the ASS file.
            mode (str): The file mode of the ASS file.
            sections (List[Section]): The list of sections in the ASS file.
        """
        self.file_name = file_name
        self.mode = mode
        self.sections = sections
        self._io = None

    def write(self) -> "PyAss":
        """
        Writes the ASS file to disk.

        Returns:
            PyAss: The PyAss object.
        """
        sections = [section.to_ass_format() for section in self.sections]
        self._io.writelines(sections)
        return self

    def __enter__(self):
        """
        Opens the ASS file in write mode and returns the PyAss object.

        Returns:
            PyAss: The PyAss object.

        Raises:
            UnsupportedFileFormat: If the file name does not have the ".ass" extension.
        """
        _, extention = os.path.splitext(self.file_name)
        if extention != ".ass":
            raise UnsupportedFileFormat(file=self.file_name)
        self._io = open(self.file_name, mode="w+", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Closes the ASS file.
        """
        if self._io:
            self._io.close()


if __name__ == "__main__":
    transcripts = Transcript.open_transcript_json(
        file_path="/home/emperorsixpacks/Downloads/asrOutput(2).json"
    )
    ordering_format = Format(
        fields=[
            "Name",
            "Fontname",
            "Fontsize",
            "PrimaryColour",
            "SecondaryColour",
            "OutlineColour",
            "BackgroundColor",
        ]
    )
    # print(transcripts)

    style = Style(order_format=ordering_format)
    dialogue_format = Format(fields=["Layer", "Start", "End", "Style", "Text"])
    # print(Dialogue(Text="Hello", style=style, Start=1111, End=222, ordering_format=dialogue_format))
    dialogues = Dialogue.from_list(
        transcripts,
        style=style,
        order_format=dialogue_format,
        focus_style=r"{\xbord20}{\ybord10}{\3c&HD4AF37&\1c&HFFFFFF&}",
    )
    # dialogue = Dialogue(ordering_format=dialogue_format, style=style)

    # sript_info = Section(title="Script Info", fields=(["title", "Sample project"]))
    event_fields = [["Format", dialogue_format.return_fields_str()]]

    for dialogue in dialogues:
        event_fields.append(dialogue)
    events = Section(
        title="Events",
        fields=event_fields,
    )
    with PyAss("test.ass", "w", sections=[events]) as ass:
        ass.write()
