# Day 1: Building a Local LLM Q&A Assistant

# Folder Structure

llm-python-learning-journey/
├── README.md
└── day-01-local-qa-app/
    ├── README.md
    ├── requirements.txt
    ├── .gitignore
    └── src/
        ├── __init__.py
        ├── app.py
        ├── config/
        │   ├── __init__.py
        │   └── settings.py
        ├── models/
        │   ├── __init__.py
        │   ├── llm_loader.py
        │   └── prompt_templates.py
        └── utils/
            ├── __init__.py
            ├── helpers.py
            └── logger.py
            

This project involves building a Q&A application that leverages Ollama to run open-source Large Language Models (LLMs) locally on a CPU. The application features a Streamlit-based web interface and follows software engineering best practices.

## Setup Instructions

### Prerequisites
- Python 3.8+
- Git
- Ollama installed and running (https://ollama.ai/)

### Installation
1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate