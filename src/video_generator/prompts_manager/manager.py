import os
from video_generator.exceptions.promptexceptions import InvalidLocationError, ServerTimeOutError







class PromptsBase:
    def __init__(self, location) -> None:
        self.location = location
        self.contents: str = None
        self.session = requests.Session()
    
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
    
    
    def check_url(self, url:str) -> int:
        return send_requst(session=self.session, method="HEAD", url=url)
    
    @staticmethod
    def path_is_url(url) -> bool:
        """
        A static method to check if a given path is a URL.
        
        Parameters:
            url (str): The path to check.
        
        Returns:
            bool: True if the path is a URL, False otherwise.
        """
        if url.startswith(("https://", "http://", "www.")):
            return True
        return False
    
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
        if self.path_is_url(location) and self.check_url(location) != 200:
            raise InvalidLocationError(location=location)

        if not self.check_path(location):
            raise InvalidLocationError(location=location)
        
        self._location = location
    

    def _read_from_url(self):
        try:
            responce  = requests.get(self.location, timeout=30)
        except TimeoutError as e:
            raise ServerTimeOutError(location=self.location) from e
        else:
            self.contents = responce.text

    def _read_from_path(self):
        pass

    def read_prompt(self):
        if self.path_is_url(self.location):
            self._read_from_url()
        else:
            self._read_from_path()
    
    def __del__(self):
        self.session.close()
        
