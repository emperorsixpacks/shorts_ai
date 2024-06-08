import os
import requests
from exceptions import InvalidLocationError

class PromptsBase:
    def __init__(self, location) -> None:
        self.location = location

    @staticmethod
    def check_url(url:str) -> int:
        """
        A static method to check the availability of a URL by sending a HEAD request.
        
        Parameters:
            url (str): The URL to check.
        
        Returns:
            int: The status code of the HEAD request.
        """
        return requests.head(url, timeout=60)
    
    @staticmethod
    def check_path(path:str) -> bool:
        """
        A method to check if the specified path exists.

        Parameters:
            path (str): The path to check.

        Returns:
            bool: True if the path exists, False otherwise.
        """
        return os.path.exists(path)

    @property
    def location(self):
        """
        Getter method for the 'location' property.
        Returns the value of the '_location' attribute.
        """
        return self._location
    
    @location.setter
    def set_location(self, location:str):
        """
        Sets the location of the prompt. Validates the location based on URL or path. 
        Raises InvalidLocationError if the location is invalid. 

        Parameters:
            location (str): The location to be set for the prompt.

        Returns:
            None
        """
        if location.startswith(("https://", "http://")):
            if self.check_url(location) != 200:
                raise InvalidLocationError(location=location)

        if not self.check_path(location):
            raise InvalidLocationError(location=location)
        
        self._location = location