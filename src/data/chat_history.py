"""
Manages chat history for the DAISY assistant with enhanced features.
"""
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from src.utils.config import CHAT_HISTORY_FILE
import logging

logger = logging.getLogger(__name__)

class ChatHistory:
    def __init__(self, history_file=None, max_messages=1000, max_age_days=30):
        """
        Initialize chat history manager.
        
        Args:
            history_file: Path to chat history file
            max_messages: Maximum number of messages to keep
            max_age_days: Maximum age of messages in days
        """
        self.history_file = history_file or CHAT_HISTORY_FILE
        self.max_messages = max_messages
        self.max_age_days = max_age_days
        self.messages = self.load_history()
        
        # Clean up old messages on initialization
        self.cleanup_old_messages()
    
    def load_history(self) -> List[Dict]:
        """Load chat history from file with error handling."""
        try:
            if not os.path.exists(self.history_file):
                logger.info("No existing chat history file found, starting fresh")
                return []
                
            with open(self.history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if not isinstance(data, list):
                logger.warning("Invalid chat history format, starting fresh")
                return []
                
            logger.info(f"Loaded {len(data)} messages from chat history")
            return data
            
        except (FileNotFoundError, json.JSONDecodeError, PermissionError) as e:
            logger.warning(f"Error loading chat history: {e}, starting fresh")
            return []
        except Exception as e:
            logger.error(f"Unexpected error loading chat history: {e}", exc_info=True)
            return []
    
    def save_history(self) -> bool:
        """Save chat history to file with error handling."""
        try:
            # Make sure the directory exists
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            
            # Create a backup if the file already exists
            if os.path.exists(self.history_file):
                backup_file = f"{self.history_file}.backup"
                try:
                    import shutil
                    shutil.copy2(self.history_file, backup_file)
                except Exception as e:
                    logger.warning(f"Could not create backup: {e}")
            
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.messages, f, indent=2, ensure_ascii=False)
                
            logger.debug(f"Saved {len(self.messages)} messages to chat history")
            return True
            
        except (PermissionError, OSError) as e:
            logger.error(f"Error saving chat history: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error saving chat history: {e}", exc_info=True)
            return False
    
    def add_message(self, role: str, content: str) -> bool:
        """
        Add a message to chat history with validation.
        
        Args:
            role: Message role ('user' or 'assistant')
            content: Message content
            
        Returns:
            bool: True if message was added successfully
        """
        # Validate inputs
        if not role or role not in ['user', 'assistant']:
            logger.warning(f"Invalid role: {role}")
            return False
            
        if not content or not content.strip():
            logger.warning("Empty or whitespace-only content")
            return False
        
        content = content.strip()
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        self.messages.append(message)
        logger.debug(f"Added {role} message (length: {len(content)})")
        
        # Clean up if we exceed limits
        self._enforce_limits()
        
        # Save to file
        return self.save_history()
    
    def get_formatted_history(self, include_timestamps=False) -> List[Dict]:
        """
        Get formatted history for API calls.
        
        Args:
            include_timestamps: Whether to include timestamps
            
        Returns:
            List of formatted messages
        """
        if include_timestamps:
            return [{"role": msg["role"], "content": msg["content"], "timestamp": msg.get("timestamp")} 
                    for msg in self.messages]
        else:
            return [{"role": msg["role"], "content": msg["content"]} 
                    for msg in self.messages]
    
    def get_recent_history(self, max_messages: int = 20) -> List[Dict]:
        """
        Get recent chat history limited to specified number of messages.
        
        Args:
            max_messages: Maximum number of recent messages to return
            
        Returns:
            List of recent formatted messages
        """
        recent_messages = self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages
        return [{"role": msg["role"], "content": msg["content"]} for msg in recent_messages]
    
    def cleanup_old_messages(self) -> int:
        """
        Remove messages older than max_age_days.
        
        Returns:
            Number of messages removed
        """
        if not self.max_age_days:
            return 0
            
        cutoff_date = datetime.now() - timedelta(days=self.max_age_days)
        initial_count = len(self.messages)
        
        self.messages = [
            msg for msg in self.messages
            if self._get_message_date(msg) > cutoff_date
        ]
        
        removed_count = initial_count - len(self.messages)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} old messages (older than {self.max_age_days} days)")
            self.save_history()
            
        return removed_count
    
    def _enforce_limits(self):
        """Enforce message count limits."""
        if len(self.messages) > self.max_messages:
            overflow = len(self.messages) - self.max_messages
            self.messages = self.messages[overflow:]
            logger.info(f"Removed {overflow} old messages to enforce limit of {self.max_messages}")
    
    def _get_message_date(self, message: Dict) -> datetime:
        """Get datetime from message timestamp."""
        try:
            timestamp = message.get("timestamp")
            if timestamp:
                return datetime.fromisoformat(timestamp)
            else:
                # If no timestamp, assume it's recent
                return datetime.now()
        except (ValueError, TypeError):
            # If timestamp is invalid, assume it's recent
            return datetime.now()
    
    def clear_history(self) -> bool:
        """Clear all chat history."""
        self.messages = []
        logger.info("Cleared all chat history")
        return self.save_history()
    
    def get_statistics(self) -> Dict:
        """Get chat history statistics."""
        if not self.messages:
            return {
                "total_messages": 0,
                "user_messages": 0,
                "assistant_messages": 0,
                "oldest_message": None,
                "newest_message": None,
                "total_characters": 0
            }
        
        user_count = sum(1 for msg in self.messages if msg["role"] == "user")
        assistant_count = sum(1 for msg in self.messages if msg["role"] == "assistant")
        total_chars = sum(len(msg["content"]) for msg in self.messages)
        
        timestamps = [self._get_message_date(msg) for msg in self.messages]
        oldest = min(timestamps) if timestamps else None
        newest = max(timestamps) if timestamps else None
        
        return {
            "total_messages": len(self.messages),
            "user_messages": user_count,
            "assistant_messages": assistant_count,
            "oldest_message": oldest.isoformat() if oldest else None,
            "newest_message": newest.isoformat() if newest else None,
            "total_characters": total_chars
        } 