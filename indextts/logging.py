import logging
import os
import signal
import sys
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any, Callable, Dict, Optional

# Default log level
DEFAULT_LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ANSI color codes for terminal output
COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "cyan": "\033[36m",
    "red": "\033[31m",
    "magenta": "\033[35m",
    "white": "\033[37m",
}

# Box-drawing characters for sections
BOX = {
    "top_left": "┌",
    "top_right": "┐",
    "bottom_left": "└",
    "bottom_right": "┘",
    "horizontal": "─",
    "vertical": "│",
    "t_right": "├",
    "t_left": "┤",
}

# Status symbols
STATUS_SYMBOLS = {
    "progress": f"{COLORS['yellow']}[...]{COLORS['reset']}",
    "complete": f"{COLORS['green']}[OK]{COLORS['reset']}",
    "failed": f"{COLORS['red']}[FAIL]{COLORS['reset']}",
    "info": f"{COLORS['blue']}[i]{COLORS['reset']}",
    "request": f"{COLORS['magenta']}[REQ]{COLORS['reset']}",
    "response": f"{COLORS['cyan']}[RES]{COLORS['reset']}",
}


def print_stage(
    message: str,
    status: str = "progress",
    elapsed: Optional[float] = None,
    message_extra: Optional[str] = None,
) -> None:
    """Print a formatted stage message with status indicator.
    
    Args:
        message: The main message to display
        status: One of 'progress', 'complete', 'failed', 'info'
        elapsed: Optional elapsed time in seconds to display
        message_extra: Optional extra message to append
    """
    symbol = STATUS_SYMBOLS.get(status, STATUS_SYMBOLS["info"])
    
    parts = [symbol, message]
    
    if elapsed is not None:
        parts.append(f"{COLORS['dim']}({elapsed:.2f}s){COLORS['reset']}")
    
    if message_extra:
        parts.append(f"{COLORS['dim']}{message_extra}{COLORS['reset']}")
    
    print(" ".join(parts))

def setup_logging(name: str = "indextts", log_level: int | str = DEFAULT_LOG_LEVEL, log_dir: str = "logs"):
    """
    Sets up the logging configuration for the application.

    Args:
        name (str): The name of the root logger to configure.
        log_level (int | str): The logging level (e.g., logging.INFO, "DEBUG").
        log_dir (str): The directory where log files will be stored.
    """
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), DEFAULT_LOG_LEVEL)

    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{name}.log")

    # Get the logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Prevent duplicate handlers if setup is called multiple times
    if logger.handlers:
        return logger

    # Create formatters
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (Rotating)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logging initialized. Level: {logging.getLevelName(log_level)}, Log file: {log_file}")
    return logger

def get_logger(name: str = "indextts"):
    """
    Returns a configured logger. If not set up, it defaults to a basic configuration.
    
    Args:
        name (str): The name of the logger to retrieve.
    """
    logger = logging.getLogger(name)
    # If the logger (or its parents) hasn't been configured, standard logging might not output anything
    # if basicConfig hasn't been called. However, we rely on setup_logging being called at entry points.
    return logger


def print_section_header(title: str, width: int = 60) -> None:
    """Print a section header with box-drawing characters.
    
    Args:
        title: The title to display
        width: Total width of the header line
    """
    title_with_padding = f" {title} "
    remaining = width - len(title_with_padding) - 2
    left_pad = remaining // 2
    right_pad = remaining - left_pad
    
    line = (
        f"{COLORS['cyan']}{BOX['top_left']}"
        f"{BOX['horizontal'] * left_pad}"
        f"{COLORS['bold']}{title_with_padding}{COLORS['reset']}{COLORS['cyan']}"
        f"{BOX['horizontal'] * right_pad}"
        f"{BOX['top_right']}{COLORS['reset']}"
    )
    print(line)


def print_section_footer(width: int = 60) -> None:
    """Print a section footer with box-drawing characters."""
    line = (
        f"{COLORS['cyan']}{BOX['bottom_left']}"
        f"{BOX['horizontal'] * (width - 2)}"
        f"{BOX['bottom_right']}{COLORS['reset']}"
    )
    print(line)


def print_section_item(key: str, value: Any, indent: int = 2) -> None:
    """Print a key-value pair within a section.
    
    Args:
        key: The label/key
        value: The value to display
        indent: Number of spaces to indent
    """
    spaces = " " * indent
    print(f"{COLORS['cyan']}{BOX['vertical']}{COLORS['reset']}{spaces}"
          f"{COLORS['dim']}{key}:{COLORS['reset']} {COLORS['white']}{value}{COLORS['reset']}")


def print_config_section(title: str, config: Dict[str, Any], width: int = 60) -> None:
    """Print a configuration section with all key-value pairs.
    
    Args:
        title: Section title
        config: Dictionary of configuration values
        width: Width of the section
    """
    print_section_header(title, width)
    for key, value in config.items():
        # Format boolean values nicely
        if isinstance(value, bool):
            value = f"{COLORS['green']}enabled{COLORS['reset']}" if value else f"{COLORS['red']}disabled{COLORS['reset']}"
        print_section_item(key, value)
    print_section_footer(width)


def print_request_start(request_id: str, text: str, max_text_len: int = 50) -> None:
    """Print the start of a TTS request.
    
    Args:
        request_id: Unique identifier for the request
        text: The text being synthesized
        max_text_len: Maximum text length to display
    """
    truncated = text[:max_text_len] + "..." if len(text) > max_text_len else text
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{STATUS_SYMBOLS['request']} {COLORS['dim']}[{timestamp}]{COLORS['reset']} "
          f"{COLORS['magenta']}#{request_id}{COLORS['reset']} "
          f"\"{truncated}\"")


def print_request_complete(
    request_id: str,
    duration: float,
    audio_length: float,
    rtf: float,
    cached: bool = False
) -> None:
    """Print the completion of a TTS request.
    
    Args:
        request_id: Unique identifier for the request
        duration: Time taken for inference in seconds
        audio_length: Length of generated audio in seconds
        rtf: Real-time factor
        cached: Whether the result was from cache
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    cache_indicator = f" {COLORS['cyan']}(cached){COLORS['reset']}" if cached else ""
    print(f"{STATUS_SYMBOLS['response']} {COLORS['dim']}[{timestamp}]{COLORS['reset']} "
          f"{COLORS['cyan']}#{request_id}{COLORS['reset']}{cache_indicator} "
          f"{COLORS['dim']}duration:{COLORS['reset']}{duration:.2f}s "
          f"{COLORS['dim']}audio:{COLORS['reset']}{audio_length:.2f}s "
          f"{COLORS['dim']}RTF:{COLORS['reset']}{rtf:.2f}")


class GracefulShutdown:
    """Context manager for handling graceful shutdown on SIGINT/SIGTERM.
    
    Usage:
        with GracefulShutdown(cleanup_func):
            # Your server code here
            pass
    """
    
    def __init__(self, cleanup: Optional[Callable[[], None]] = None):
        self.cleanup = cleanup
        self.shutdown_requested = False
        self._original_sigint = None
        self._original_sigterm = None
    
    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals gracefully."""
        sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        if not self.shutdown_requested:
            self.shutdown_requested = True
            print()  # New line after ^C
            print_section_header("Shutdown", 40)
            print_section_item("Signal", sig_name)
            print_section_item("Status", "Cleaning up...")
            
            if self.cleanup:
                try:
                    self.cleanup()
                    print_section_item("Cleanup", f"{COLORS['green']}complete{COLORS['reset']}")
                except Exception as e:
                    print_section_item("Cleanup", f"{COLORS['red']}failed: {e}{COLORS['reset']}")
            
            print_section_footer(40)
            sys.exit(0)
    
    def __enter__(self) -> "GracefulShutdown":
        self._original_sigint = signal.signal(signal.SIGINT, self._signal_handler)
        self._original_sigterm = signal.signal(signal.SIGTERM, self._signal_handler)
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # Restore original handlers
        if self._original_sigint:
            signal.signal(signal.SIGINT, self._original_sigint)
        if self._original_sigterm:
            signal.signal(signal.SIGTERM, self._original_sigterm)
