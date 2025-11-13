from functools import wraps

# ! Key rule: ds and adts can import utils, but utils should never import from ds or adts.
# todo create helper function - that takes an input text from file. and counts all the words in the text. then takes the words (keys) and appends them to a list via a counter object. useful for generating random data for testing


class Ansi:
    """ANSI Color Code Wrapper for a text input. """
    BLUE = "\033[1;36m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    RESET = "\033[0m"

    @staticmethod
    def color(text: str, color_code: str) -> str:
        return f"{color_code}{text}{Ansi.RESET}"



class RandomClass:
    """dummy class for testing - move to test folder later."""
    def __init__(self, name=None) -> None:
        self.name = name
    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}: {self.name}"
    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}: {self.name}"

