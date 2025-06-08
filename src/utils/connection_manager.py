"""
Ollama connection manager for robust connection handling.
"""
import requests
import time
import logging
import threading
from typing import Optional, Dict, Any, List
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class OllamaConnectionManager:
    """Manages Ollama connections with automatic reconnection and health monitoring."""
    
    def __init__(self, base_url="http://localhost:11434", timeout=30, max_retries=3, retry_delay=2):
        """
        Initialize connection manager.
        
        Args:
            base_url: Base URL for Ollama API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Connection state
        self._is_connected = False
        self._available_models = []
        self._last_health_check = 0
        self._health_check_interval = 30  # seconds
        self._connection_error = None
        self._lock = threading.RLock()
        
        # Session for connection pooling
        self._session = requests.Session()
        self._session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    @property
    def is_connected(self) -> bool:
        """Check if currently connected to Ollama."""
        with self._lock:
            return self._is_connected
    
    @property
    def connection_error(self) -> Optional[str]:
        """Get the last connection error message."""
        with self._lock:
            return self._connection_error
    
    @property
    def available_models(self) -> List[str]:
        """Get list of available models."""
        with self._lock:
            return self._available_models.copy()
    
    def _should_refresh_health_check(self) -> bool:
        """Check if health check should be refreshed."""
        return time.time() - self._last_health_check > self._health_check_interval
    
    def health_check(self, force=False) -> bool:
        """
        Perform health check on Ollama connection.
        
        Args:
            force: Force health check even if recently performed
            
        Returns:
            True if healthy, False otherwise
        """
        with self._lock:
            if not force and not self._should_refresh_health_check():
                return self._is_connected
            
            try:
                logger.debug("Performing Ollama health check...")
                response = self._session.get(
                    f"{self.base_url}/api/tags",
                    timeout=5  # Shorter timeout for health checks
                )
                
                if response.status_code == 200:
                    # Update available models
                    try:
                        data = response.json()
                        self._available_models = [model["name"] for model in data.get("models", [])]
                        logger.debug(f"Available models: {self._available_models}")
                    except Exception as e:
                        logger.warning(f"Could not parse models list: {e}")
                        self._available_models = []
                    
                    self._is_connected = True
                    self._connection_error = None
                    self._last_health_check = time.time()
                    logger.debug("Health check passed")
                    return True
                else:
                    self._handle_connection_error(f"Health check failed with status {response.status_code}")
                    return False
                    
            except requests.exceptions.ConnectionError as e:
                self._handle_connection_error(f"Connection error: {str(e)}")
                return False
            except requests.exceptions.Timeout as e:
                self._handle_connection_error(f"Timeout error: {str(e)}")
                return False
            except Exception as e:
                self._handle_connection_error(f"Unexpected error: {str(e)}")
                return False
    
    def _handle_connection_error(self, error_msg: str):
        """Handle connection errors consistently."""
        self._is_connected = False
        self._connection_error = error_msg
        self._available_models = []
        logger.warning(f"Ollama connection error: {error_msg}")
    
    def connect(self, model_name: Optional[str] = None) -> bool:
        """
        Establish connection to Ollama and optionally verify model availability.
        
        Args:
            model_name: Model name to verify (optional)
            
        Returns:
            True if connected successfully, False otherwise
        """
        logger.info("Connecting to Ollama...")
        
        if not self.health_check(force=True):
            return False
        
        # If model name is specified, verify it's available
        if model_name:
            return self.verify_model(model_name)
        
        return True
    
    def verify_model(self, model_name: str) -> bool:
        """
        Verify that a specific model is available.
        
        Args:
            model_name: Name of the model to verify
            
        Returns:
            True if model is available, False otherwise
        """
        with self._lock:
            if not self._is_connected:
                if not self.health_check(force=True):
                    return False
            
            if model_name in self._available_models:
                logger.info(f"Model '{model_name}' is available")
                return True
            
            # Try direct model check
            try:
                logger.debug(f"Verifying model '{model_name}' with direct API call...")
                response = self._session.get(
                    f"{self.base_url}/api/show",
                    params={"name": model_name},
                    timeout=5
                )
                
                if response.status_code == 200:
                    logger.info(f"Model '{model_name}' verified successfully")
                    # Add to available models list if not already there
                    if model_name not in self._available_models:
                        self._available_models.append(model_name)
                    return True
                else:
                    self._suggest_similar_model(model_name)
                    return False
                    
            except Exception as e:
                logger.error(f"Error verifying model '{model_name}': {e}")
                return False
    
    def _suggest_similar_model(self, model_name: str):
        """Suggest similar models if exact match not found."""
        base_name = model_name.split(':')[0].lower()
        similar_models = [
            m for m in self._available_models 
            if base_name in m.lower()
        ]
        
        if similar_models:
            suggestion = similar_models[0]
            error_msg = f"Model '{model_name}' not found. Did you mean '{suggestion}'?"
            logger.warning(error_msg)
            self._connection_error = error_msg
        else:
            error_msg = f"Model '{model_name}' not found. Available models: {', '.join(self._available_models) if self._available_models else 'None'}"
            logger.warning(error_msg)
            self._connection_error = error_msg
    
    @contextmanager
    def request_session(self):
        """Context manager for making requests with automatic reconnection."""
        if not self.is_connected:
            if not self.connect():
                raise ConnectionError(f"Could not connect to Ollama: {self.connection_error}")
        
        try:
            yield self._session
        except requests.exceptions.ConnectionError:
            # Try to reconnect once
            logger.warning("Connection lost, attempting to reconnect...")
            self._handle_connection_error("Connection lost")
            if self.connect():
                yield self._session
            else:
                raise ConnectionError(f"Could not reconnect to Ollama: {self.connection_error}")
    
    def make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """
        Make a request to Ollama API with automatic retry and reconnection.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (without base URL)
            **kwargs: Additional arguments for requests
            
        Returns:
            Response object
            
        Raises:
            ConnectionError: If unable to connect after retries
            requests.exceptions.RequestException: For other request errors
        """
        endpoint = endpoint.lstrip('/')
        url = f"{self.base_url}/{endpoint}"
        
        # Set default timeout if not provided
        kwargs.setdefault('timeout', self.timeout)
        
        for attempt in range(self.max_retries):
            try:
                with self.request_session() as session:
                    response = session.request(method, url, **kwargs)
                    return response
                    
            except requests.exceptions.ConnectionError as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(self.retry_delay)
                    continue
                else:
                    raise ConnectionError(f"Failed to connect after {self.max_retries} attempts: {e}")
            
            except requests.exceptions.Timeout as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Request timeout (attempt {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(self.retry_delay)
                    continue
                else:
                    raise
    
    def chat_completion(self, model: str, messages: List[Dict], **options) -> Dict[str, Any]:
        """
        Perform chat completion with automatic retry.
        
        Args:
            model: Model name to use
            messages: List of message dictionaries
            **options: Additional options for the chat completion
            
        Returns:
            Response dictionary from Ollama
            
        Raises:
            ConnectionError: If unable to connect
            ValueError: If model is not available
            requests.exceptions.RequestException: For API errors
        """
        # Verify model availability
        if not self.verify_model(model):
            raise ValueError(f"Model '{model}' is not available. {self.connection_error}")
        
        payload = {
            'model': model,
            'messages': messages,
            'stream': False,
            **options
        }
        
        logger.debug(f"Making chat completion request with model: {model}")
        response = self.make_request('POST', 'api/chat', json=payload)
        
        if response.status_code != 200:
            error_msg = f"Chat completion failed with status {response.status_code}: {response.text}"
            logger.error(error_msg)
            raise requests.exceptions.RequestException(error_msg)
        
        return response.json()
    
    def disconnect(self):
        """Clean up connection resources."""
        with self._lock:
            logger.info("Disconnecting from Ollama...")
            self._session.close()
            self._is_connected = False
            self._connection_error = None
            self._available_models = []
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect() 