"""
Helper functions for the application.
"""

import time
import requests
from .logger import logger

def format_time(seconds):
    """
    Format time in seconds to a human-readable string.
    
    Args:
        seconds (float): Time in seconds
        
    Returns:
        str: Formatted time string
    """
    if seconds < 0.001:
        return f"{seconds * 1000000:.2f} µs"
    elif seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    else:
        return f"{seconds:.2f} s"

def check_ollama_status(host):
    """
    Check if Ollama server is running and accessible.
    
    Args:
        host (str): Ollama server host URL
        
    Returns:
        bool: True if Ollama is accessible, False otherwise
    """
    try:
        response = requests.get(f"{host}/api/tags", timeout=2)
        return response.status_code == 200
    except requests.RequestException as e:
        logger.warning(f"Ollama server not accessible: {str(e)}")
        return False

def time_function(func):
    """
    Decorator to measure and log the execution time of a function.
    
    Args:
        func (callable): Function to be timed
        
    Returns:
        callable: Wrapped function with timing
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.debug(f"Function {func.__name__} executed in {format_time(execution_time)}")
        return result
    return wrapper