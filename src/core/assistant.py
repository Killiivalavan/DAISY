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
        self.ollama_available = False
        self.connection_error = None
        
        # Initialize retriever for RAG if enabled
        self.retriever = Retriever() if use_rag else None
        
        # Pre-initialize connection to Ollama
        self.initialize_ollama_connection()
        
    def initialize_ollama_connection(self):
        """Initialize connection to Ollama and verify it's working."""
        try:
            print("Pre-initializing connection to Ollama...")
            # Use a lightweight API call to test connection
            response = requests.get(
                'http://localhost:11434/api/tags',
                timeout=5
            )
            
            if response.status_code == 200:
                # First check if the model is in the list of models from the tags response
                try:
                    tags_data = response.json()
                    available_models = [model["name"] for model in tags_data.get("models", [])]
                    
                    if self.model_name in available_models:
                        print(f"Successfully connected to Ollama with model: {self.model_name}")
                        self.ollama_available = True
                        self.connection_error = None
                        return True
                    
                    # If we didn't find the exact match, try a secondary check using /api/show
                    # This is a fallback in case the model name format is different
                    model_response = requests.get(
                        f'http://localhost:11434/api/show?name={self.model_name}',
                        timeout=5
                    )
                    
                    if model_response.status_code == 200:
                        print(f"Successfully connected to Ollama with model: {self.model_name}")
                        self.ollama_available = True
                        self.connection_error = None
                        return True
                    else:
                        # List available models for troubleshooting
                        print(f"Ollama is running, but model '{self.model_name}' is not available")
                        print(f"Available models: {', '.join(available_models)}")
                        
                        # Suggest a similar model if one exists
                        similar_models = [m for m in available_models if self.model_name.split(':')[0].lower() in m.lower()]
                        if similar_models:
                            suggestion = similar_models[0]
                            self.connection_error = f"Model '{self.model_name}' not found. Did you mean '{suggestion}'?"
                            print(f"Did you mean: {suggestion}?")
                        else:
                            self.connection_error = f"Model '{self.model_name}' not found"
                except Exception as e:
                    print(f"Error parsing model data: {e}")
                    self.connection_error = f"Error verifying model: {str(e)}"
            else:
                print(f"Ollama connection test failed with status code: {response.status_code}")
                self.connection_error = f"Ollama API returned status code {response.status_code}"
            
        except requests.exceptions.ConnectionError:
            print("Could not connect to Ollama. Is it running?")
            self.connection_error = "Could not connect to Ollama server"
        except Exception as e:
            print(f"Error initializing Ollama connection: {str(e)}")
            self.connection_error = str(e)
            
        self.ollama_available = False
        return False
        
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
            # Check if connection was already initialized
            if not self.ollama_available:
                # Try to initialize again
                if not self.initialize_ollama_connection():
                    if self.connection_error:
                        return f"I'm having trouble connecting to my brain: {self.connection_error}"
                    else:
                        return "Could not connect to Ollama. Is it running?"
            
            # We've verified Ollama is available, make the chat request
            response = requests.post(
                'http://localhost:11434/api/chat',
                json={
                    'model': self.model_name,
                    'messages': messages,
                    'stream': False
                },
                timeout=30  # Increased timeout for potentially longer responses
            )
            
            if response.status_code != 200:
                print(f"Ollama API error with status code: {response.status_code}")
                error_msg = f"Error from Ollama API: Status code {response.status_code}"
                self.ollama_available = False
                return error_msg
                
            response_json = response.json()
            ai_response = response_json['message']['content']
            self.chat_history.add_message("assistant", ai_response)
            return ai_response
            
        except requests.exceptions.ConnectionError:
            error_msg = "Could not connect to Ollama. Is it running?"
            print(error_msg)
            self.ollama_available = False
            return error_msg
        except Exception as e:
            error_msg = f"Error getting AI response: {str(e)}"
            print(error_msg)
            self.ollama_available = False
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