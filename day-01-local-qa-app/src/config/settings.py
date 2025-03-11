"""
Configuration settings for the LocalLLM Q&A application.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Ollama settings
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.environ.get("DEFAULT_OLLAMA_MODEL", "mistral")
AVAILABLE_OLLAMA_MODELS = [
    "mistral",
    "llama2",
    "codellama",
    "phi",
    "gemma"
]

# HuggingFace fallback models (CPU-friendly)
DEFAULT_HF_MODEL = os.environ.get("DEFAULT_HF_MODEL", "google/flan-t5-small")
AVAILABLE_HF_MODELS = [
    "google/flan-t5-small",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
]

# Generation settings
DEFAULT_MAX_LENGTH = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 40

# Logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FILE = os.path.join(BASE_DIR, "app.log")

# UI settings
APP_TITLE = "LocalLLM Q&A Assistant"
APP_ICON = "🤖"
