"""
Main voice assistant class for DAISY.
"""
import requests
from typing import Optional, List, Dict
from src.core.personality import PersonalityManager
from src.data.chat_history import ChatHistory
from src.rag.retriever import Retriever

class VoiceAssistant:
    def __init__(self, model_name="llama3.2:latest", use_rag=True):
        self.chat_history = ChatHistory()
        self.personality = PersonalityManager()
        self.model_name = model_name
        self.use_rag = use_rag
        
        # Initialize retriever for RAG if enabled
        self.retriever = Retriever() if use_rag else None
        
    def get_ai_response(self, user_input):
        # Add user message to history
        self.chat_history.add_message("user", user_input)
        
        # Get personality as system message
        system_prompt = self.personality.get_personality()
        
        # Prepare messages
        if self.use_rag and self.retriever:
            # Use RAG-enhanced prompting
            messages = self.retriever.get_rag_prompt(
                query=user_input,
                system_prompt=system_prompt
            )
            
            # Add relevant chat history
            formatted_history = self.chat_history.get_formatted_history()
            # Remove the system message and the last user message (which is already in the RAG prompt)
            history_messages = [msg for msg in formatted_history[:-1] 
                              if msg["role"] != "system"]
            
            # Insert history between system and user message
            if history_messages:
                messages = [messages[0]] + history_messages + [messages[-1]]
        else:
            # Standard prompting without RAG
            messages = [
                {"role": "system", "content": system_prompt}
            ] + self.chat_history.get_formatted_history()
        
        # Get response from Ollama
        try: 
            print("Attempting to connect to Ollama...")
            response = requests.post(
                'http://localhost:11434/api/chat',
                json={
                    'model': self.model_name,
                    'messages': messages,
                    'stream': False
                },
                timeout=10  # Add timeout to avoid hanging
            )
            print(f"Response status code: {response.status_code}")
            response_json = response.json()
            ai_response = response_json['message']['content']
            self.chat_history.add_message("assistant", ai_response)
            return ai_response
            
        except requests.exceptions.ConnectionError:
            error_msg = "Could not connect to Ollama. Is it running?"
            print(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error getting AI response: {str(e)}"
            print(error_msg)
            return error_msg
    
    def process_documents(self, force_reprocess=False):
        """
        Process and index documents for RAG.
        
        Args:
            force_reprocess: If True, reprocess all documents regardless of tracking status
            
        Returns:
            Message indicating the result of document processing
        """
        if not self.use_rag or not self.retriever:
            return "RAG is not enabled."
        
        try:
            chunk_count = self.retriever.process_documents(force_reprocess=force_reprocess)
            
            if chunk_count == 0:
                return "No new documents to process. All documents are up-to-date."
                
            return f"Successfully processed and indexed {chunk_count} document chunks."
        except Exception as e:
            error_msg = f"Error processing documents: {str(e)}"
            print(error_msg)
            return error_msg 