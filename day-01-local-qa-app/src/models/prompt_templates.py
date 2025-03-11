"""
Prompt templates for different use cases.
"""

class PromptTemplate:
    """
    Class to handle prompt templates and formatting.
    """
    
    @staticmethod
    def qa_template(question, conversation_history=None):
        """
        Format a question-answering prompt.
        
        Args:
            question (str): User question
            conversation_history (list, optional): List of previous conversation turns
            
        Returns:
            str: Formatted prompt
        """
        if not conversation_history:
            return f"""
You are a helpful assistant. Answer the following question:

Question: {question}

Answer:
""".strip()
        
        # Format conversation history
        history_text = ""
        for turn in conversation_history:
            role = turn.get("role", "")
            content = turn.get("content", "")
            if role.lower() == "user":
                history_text += f"Human: {content}\n"
            elif role.lower() == "assistant":
                history_text += f"Assistant: {content}\n"
        
        # Add the current question
        history_text += f"Human: {question}\nAssistant:"
        
        return f"""
You are a helpful assistant. Here's the conversation so far:

{history_text}
""".strip()
    
    @staticmethod
    def coding_template(question, language=None):
        """
        Format a prompt for coding questions.
        
        Args:
            question (str): User's coding question
            language (str, optional): Programming language
            
        Returns:
            str: Formatted prompt
        """
        lang_context = f"using {language}" if language else ""
        
        return f"""
You are an expert programming assistant {lang_context}. Answer the following coding question with clear explanations and example code:

Question: {question}

Answer:
""".strip()
    
    @staticmethod
    def educational_template(question, topic=None, level="beginner"):
        """
        Format a prompt for educational explanations.
        
        Args:
            question (str): User's question
            topic (str, optional): The topic area
            level (str): Knowledge level (beginner, intermediate, advanced)
            
        Returns:
            str: Formatted prompt
        """
        topic_context = f"about {topic}" if topic else ""
        
        return f"""
You are an educational assistant helping a {level} learner {topic_context}. Provide a clear and helpful explanation for the following question:

Question: {question}

Explanation:
""".strip()