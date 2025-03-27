"""
Text cleaning utilities.
"""

def clean_text(text):
    """
    Basic text cleaning function.
    
    Args:
        text (str): Input text to clean
    
    Returns:
        str: Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove leading and trailing whitespace
    text = text.strip()
    
    return text

def remove_special_chars(text):
    """
    Remove special characters from text.
    
    Args:
        text (str): Input text
    
    Returns:
        str: Text with special characters removed
    """
    import re
    
    # Remove punctuation while keeping spaces
    text = re.sub(r'[^\w\s]', '', text)
    
    return text