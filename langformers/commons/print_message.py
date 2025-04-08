from colorama import Fore, Style
import logging

_logger = logging.getLogger('langformers')
_logger.addHandler(logging.NullHandler())


def print_message(message: str, level: str = "info"):
    """
    Custom print message function for Langformers that's logging-compatible.

    Args:
        message (str): The message to print/log.
        level (str): Log level ("debug", "info", "warning", "error", "critical").
                     Defaults to "info".
    """

    colored_message = f"{Fore.CYAN}[Langformers]{Style.RESET_ALL} {message}"
    print(colored_message)

    log_level = getattr(logging, level.upper(), logging.INFO)
    _logger.log(log_level, message)

