DEFAULT_WIKIPEDIA_SEARCH_LIMIT = "4"

DEFAULT_WIKIPEDIA_SEARCH_PARAMS = {
    "action": "query",
    "format": "json",
    "prop": "revisions",
    "rvprop": "content",
    "rvslots": "main",
    "limit": DEFAULT_WIKIPEDIA_SEARCH_LIMIT,
}


WIKI_API_SEARCH_URL = "https://en.wikipedia.org/w/rest.php/v1/search/page"
TTS_MAKER_URL = "https://ttsmp3.com/makemp3_new.php"

VALIDATION_PROMPT = "validation_prompt.txt"

MIN_VIDEO_LENGTH = 5
MAX_VIDEO_LENGTH = 12
MIN_VIDEO_HEIGHT = 1000
MIN_VIDEO_WIDTH = 1000
DEFAULT_SUBREDDITS = ["oddlysatisfying", "PerfectTiming", "satisfying"]
DEFAULT_NUMBER_OF_VIDEOS = 4
BOOL_DICT = {"True": True, "False": False}
