"""
LLM loader for different model backends (Ollama and HuggingFace).
"""

import sys
import json
import requests
from pathlib import Path
from transformers import pipeline

# Add src directory to path for imports
src_dir = str(Path(__file__).resolve().parent.parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from utils.logger import logger
from utils.helpers import time_function, check_ollama_status
from config import settings

class LLMManager:
    """
    Manager for loading and interacting with different LLM backends.
    """
    
    def __init__(self):
        """Initialize the LLM Manager."""
        self.ollama_host = settings.OLLAMA_HOST
        self.default_ollama_model = settings.DEFAULT_OLLAMA_MODEL
        self.default_hf_model = settings.DEFAULT_HF_MODEL
        
        # Check if Ollama is available
        self.ollama_available = check_ollama_status(self.ollama_host)
        logger.info(f"Ollama available: {self.ollama_available}")
        
        # Initialize HuggingFace model if needed
        self.hf_pipeline = None
        if not self.ollama_available:
            logger.info(f"Initializing HuggingFace model: {self.default_hf_model}")
            self._initialize_hf_model(self.default_hf_model)
    
    def _initialize_hf_model(self, model_name):
        """Initialize a HuggingFace model pipeline."""
        try:
            self.hf_pipeline = pipeline(
                "text2text-generation",
                model=model_name,
                max_length=settings.DEFAULT_MAX_LENGTH,
                device=-1,  # Use CPU
            )
            logger.info(f"Successfully loaded HuggingFace model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading HuggingFace model: {str(e)}")
            self.hf_pipeline = None
    
    @time_function
    def generate_with_ollama(self, prompt, model=None, temperature=None, max_tokens=None):
        """
        Generate text using Ollama API.
        
        Args:
            prompt (str): Input prompt
            model (str, optional): Model name
            temperature (float, optional): Sampling temperature
            max_tokens (int, optional): Maximum tokens to generate
            
        Returns:
            str: Generated text
        """
        if not self.ollama_available:
            logger.warning("Ollama not available, falling back to HuggingFace")
            return self.generate_with_hf(prompt)
        
        model = model or self.default_ollama_model
        temperature = temperature or settings.DEFAULT_TEMPERATURE
        max_tokens = max_tokens or settings.DEFAULT_MAX_LENGTH
        
        try:
            # Updated: Use 'completion' endpoint for newer Ollama versions
            request_data = {
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            # Try the newer completion endpoint first
            response = requests.post(
                f"{self.ollama_host}/api/chat",
                json={"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("message", {}).get("content", "")
            
            # Fall back to completion endpoint
            response = requests.post(
                f"{self.ollama_host}/api/completion",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            
            # Fall back to the older generate endpoint
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return self.generate_with_hf(prompt)
        
        except Exception as e:
            logger.error(f"Error generating with Ollama: {str(e)}")
            return self.generate_with_hf(prompt)
    
    @time_function
    def generate_with_hf(self, prompt, model=None, temperature=None, max_length=None):
        """
        Generate text using HuggingFace pipeline.
        
        Args:
            prompt (str): Input prompt
            model (str, optional): Model name
            temperature (float, optional): Sampling temperature
            max_length (int, optional): Maximum length to generate
            
        Returns:
            str: Generated text
        """
        model = model or self.default_hf_model
        temperature = temperature or settings.DEFAULT_TEMPERATURE
        max_length = max_length or settings.DEFAULT_MAX_LENGTH
        
        # Initialize model if not done yet or if model changed
        if self.hf_pipeline is None or self.hf_pipeline.model.name_or_path != model:
            self._initialize_hf_model(model)
        
        if self.hf_pipeline is None:
            return "Sorry, the model is not available at the moment."
        
        try:
            result = self.hf_pipeline(
                prompt,
                temperature=temperature,
                max_length=max_length
            )
            return result[0]["generated_text"]
        
        except Exception as e:
            logger.error(f"Error generating with HuggingFace: {str(e)}")
            return "Sorry, an error occurred during text generation."
    
    def generate(self, prompt, use_ollama=True, **kwargs):
        """
        Generate text using the preferred backend.
        
        Args:
            prompt (str): Input prompt
            use_ollama (bool): Whether to use Ollama if available
            **kwargs: Additional generation parameters
            
        Returns:
            str: Generated text
        """
        if use_ollama and self.ollama_available:
            return self.generate_with_ollama(prompt, **kwargs)
        else:
            return self.generate_with_hf(prompt, **kwargs)
    
    def get_available_models(self):
        """
        Get a list of available models from both backends.
        
        Returns:
            dict: Dictionary with available models
        """
        models = {
            "ollama": [],
            "huggingface": settings.AVAILABLE_HF_MODELS
        }
        
        # Get Ollama models if available
        if self.ollama_available:
            try:
                response = requests.get(f"{self.ollama_host}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models["ollama"] = [model["name"] for model in data.get("models", [])]
                else:
                    models["ollama"] = settings.AVAILABLE_OLLAMA_MODELS
            except:
                models["ollama"] = settings.AVAILABLE_OLLAMA_MODELS
        
        return models