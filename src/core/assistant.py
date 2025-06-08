"""
Main voice assistant class for DAISY.
"""
import requests
from typing import Optional, List, Dict
from src.core.personality import PersonalityManager
from src.data.chat_history import ChatHistory
from src.rag.retriever import Retriever
from src.utils.connection_manager import OllamaConnectionManager
import logging
import time

logger = logging.getLogger(__name__)

class VoiceAssistant:
    def __init__(self, model_name="llama3.2:latest", use_rag=True):
        # Initialize chat history with reasonable limits
        self.chat_history = ChatHistory(max_messages=100, max_age_days=7)  # Keep last 100 messages for 7 days
        self.personality = PersonalityManager()
        self.model_name = model_name
        self.use_rag = use_rag
        
        # Initialize connection manager
        self.connection_manager = OllamaConnectionManager()
        
        # Initialize retriever for RAG if enabled
        self.retriever = Retriever() if use_rag else None
        
        # Pre-initialize connection to Ollama
        self.initialize_ollama_connection()
    
    @property
    def ollama_available(self) -> bool:
        """Check if Ollama is available."""
        return self.connection_manager.is_connected
    
    @property
    def connection_error(self) -> Optional[str]:
        """Get connection error message."""
        return self.connection_manager.connection_error
        
    def initialize_ollama_connection(self):
        """Initialize connection to Ollama and verify it's working."""
        logger.info("Pre-initializing connection to Ollama...")
        return self.connection_manager.connect(self.model_name)
        
    def get_ai_response(self, user_input):
        """Get response from AI model with enhanced error handling and debugging."""
        logger.info(f"Getting AI response for input: '{user_input}'")
        
        # Validate input
        if not user_input or not user_input.strip():
            logger.warning("Empty or whitespace-only input received")
            return "I didn't catch what you said. Could you please try again?"
        
        user_input = user_input.strip()
        logger.debug(f"Processed input: '{user_input}'")
        
        # Add user message to history
        self.chat_history.add_message("user", user_input)
        logger.debug(f"Added user message to history. History length: {len(self.chat_history.messages)}")
        
        # Get personality as system message
        system_prompt = self.personality.get_personality()
        logger.debug(f"System prompt length: {len(system_prompt)} chars")
        
        # Prepare messages with enhanced logic
        try:
            if self.use_rag and self.retriever:
                # Use RAG-enhanced prompting
                logger.debug("Preparing RAG-enhanced prompting")
                messages = self.retriever.get_rag_prompt(
                    query=user_input,
                    system_prompt=system_prompt
                )
                logger.debug(f"RAG messages prepared: {len(messages)} messages")
                
                # Add relevant chat history but limit it to prevent context overflow
                formatted_history = self.chat_history.get_formatted_history()
                # Remove the system message and the last user message (which is already in the RAG prompt)
                history_messages = [msg for msg in formatted_history[:-1] 
                                  if msg["role"] != "system"]
                
                # Limit history to last 10 messages to prevent context overflow
                if len(history_messages) > 10:
                    history_messages = history_messages[-10:]
                    logger.debug(f"Limited history to last 10 messages")
                
                # Insert history between system and user message
                if history_messages:
                    messages = [messages[0]] + history_messages + [messages[-1]]
                    logger.debug(f"Added {len(history_messages)} history messages")
            else:
                # Standard prompting without RAG
                logger.debug("Preparing standard prompting (no RAG)")
                
                # Add explicit instruction to avoid placeholder responses
                enhanced_system_prompt = f"{system_prompt}\n\nIMPORTANT: Always provide a proper, helpful response. Never respond with placeholders like '*No response*', '*Silence*', or similar non-responses."
                
                # Get formatted history and limit it to prevent context overflow
                formatted_history = self.chat_history.get_formatted_history()
                
                # Limit history to last 15 messages to prevent context overflow
                if len(formatted_history) > 15:
                    formatted_history = formatted_history[-15:]
                    logger.debug(f"Limited history to last 15 messages")
                
                messages = [
                    {"role": "system", "content": enhanced_system_prompt}
                ] + formatted_history
                
                logger.debug(f"Standard messages prepared: {len(messages)} messages")
            
            # Log message details for debugging
            total_chars = sum(len(msg.get('content', '')) for msg in messages)
            logger.debug(f"Total message content length: {total_chars} characters")
            
            # Estimate token count (rough approximation: 1 token ≈ 4 characters)
            estimated_tokens = total_chars // 4
            logger.debug(f"Estimated token count: {estimated_tokens}")
            
            # Warn if context might be too large
            if estimated_tokens > 3000:
                logger.warning(f"Context size may be too large ({estimated_tokens} estimated tokens). This might cause issues.")
        
        except Exception as e:
            logger.error(f"Error preparing messages: {e}", exc_info=True)
            return "I encountered an error while preparing my response. Please try again."
        
        # Get response from Ollama using connection manager
        try:
            # Set up options for better response quality
            options = {
                'temperature': 0.7,
                'top_p': 0.9,
                'num_predict': 256,  # Reduced from 512 to avoid timeout issues
                'repeat_penalty': 1.1,
                'top_k': 40
            }
            
            # Remove problematic options that might cause issues
            # Removed: stop, frequency_penalty, presence_penalty
            
            logger.info(f"Making request to Ollama with model: {self.model_name}")
            logger.debug(f"Request options: {options}")
            
            # Use connection manager for chat completion
            response_json = self.connection_manager.chat_completion(
                model=self.model_name,
                messages=messages,
                **options
            )
            
            logger.debug(f"Raw Ollama response: {response_json}")
            
            # Enhanced response extraction and validation
            if not response_json:
                logger.error("Received None/empty response from Ollama")
                return "I received an empty response from my language model. Please try again."
            
            if 'message' not in response_json:
                logger.error(f"Response missing 'message' field: {list(response_json.keys())}")
                return "I received an invalid response format from my language model. Please try again."
            
            if 'content' not in response_json['message']:
                logger.error(f"Response message missing 'content' field: {list(response_json['message'].keys())}")
                return "I received a response without content from my language model. Please try again."
            
            ai_response = response_json['message']['content']
            logger.debug(f"Extracted AI response: '{ai_response}' (length: {len(ai_response)})")
            
            # Enhanced response validation
            if ai_response is None:
                logger.warning("AI response is None")
                ai_response = ""
            
            ai_response_stripped = ai_response.strip()
            
            # Check for various invalid response patterns
            invalid_responses = [
                "*No response*", "*The end*", "*Nothing*", "*Silence*", 
                "*I'm gone*", "*Nothing remains*", "*NO RESPONSE*",
                "", "None", "null", "undefined"
            ]
            
            is_invalid = False
            invalid_reason = None
            
            if not ai_response_stripped:
                is_invalid = True
                invalid_reason = "empty response"
            elif ai_response_stripped.lower() in [r.lower() for r in invalid_responses]:
                is_invalid = True
                invalid_reason = f"known invalid response pattern: '{ai_response_stripped}'"
            elif ai_response_stripped.startswith("*") and ai_response_stripped.endswith("*"):
                is_invalid = True
                invalid_reason = f"asterisk-wrapped response: '{ai_response_stripped}'"
            elif len(ai_response_stripped) < 3:
                is_invalid = True
                invalid_reason = f"suspiciously short response: '{ai_response_stripped}'"
            
            if is_invalid:
                logger.warning(f"Invalid response detected: {invalid_reason}")
                
                # Generate contextual fallback response
                fallback_response = "I apologize, but I'm having trouble generating a proper response. "
                
                # Add context-specific fallback based on the user's query
                user_lower = user_input.lower()
                if any(word in user_lower for word in ["how are you", "how do you feel", "what's up", "how's it going"]):
                    fallback_response += "I'm functioning well and ready to assist you with any questions or tasks."
                elif any(word in user_lower for word in ["help", "can you", "would you", "please"]):
                    fallback_response += "I'm here to help. Could you please rephrase your request?"
                elif any(word in user_lower for word in ["what", "who", "where", "when", "why", "how"]):
                    fallback_response += "I'd be happy to answer your question. Could you please provide more details?"
                elif any(word in user_lower for word in ["hello", "hi", "hey", "good morning", "good afternoon"]):
                    fallback_response += "Hello! How may I assist you today?"
                else:
                    fallback_response += "Could you please try rephrasing your question?"
                
                # Add to history to maintain conversation context
                self.chat_history.add_message("assistant", fallback_response)
                return fallback_response
            
            logger.info(f"Successfully generated response (length: {len(ai_response_stripped)})")
            
            # Add to history
            self.chat_history.add_message("assistant", ai_response_stripped)
            return ai_response_stripped
            
        except ConnectionError as e:
            error_msg = f"I'm having trouble connecting to my language model: {str(e)}"
            logger.error(error_msg)
            return error_msg
        except ValueError as e:
            error_msg = f"There's an issue with the language model: {str(e)}"
            logger.error(error_msg)
            return error_msg
        except KeyError as e:
            error_msg = f"I received an unexpected response format: missing {str(e)}"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"I encountered an unexpected error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return "I apologize, but I encountered an unexpected error. Please try again."
    
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
    
    def cleanup(self):
        """Clean up resources used by the assistant."""
        logger.info("Cleaning up VoiceAssistant resources...")
        self.connection_manager.disconnect()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()