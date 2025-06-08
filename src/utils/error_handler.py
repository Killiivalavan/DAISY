"""
Comprehensive error handling utilities for DAISY.
"""
import logging
import traceback
import functools
import threading
import time
from typing import Any, Callable, Optional, Type, Union, Dict
from contextlib import contextmanager
from enum import Enum

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class DaisyError(Exception):
    """Base exception for DAISY-specific errors."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.context = context or {}
        self.timestamp = time.time()

class ConnectionError(DaisyError):
    """Connection-related errors."""
    pass

class AudioError(DaisyError):
    """Audio processing errors."""
    pass

class ModelError(DaisyError):
    """AI model related errors."""
    pass

class ResourceError(DaisyError):
    """Resource management errors."""
    pass

class ConfigurationError(DaisyError):
    """Configuration errors."""
    pass

class ErrorCollector:
    """Collects and manages errors throughout the application."""
    
    def __init__(self, max_errors: int = 100):
        self.max_errors = max_errors
        self._errors = []
        self._lock = threading.RLock()
        self._error_counts = {}
    
    def add_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Add an error to the collection."""
        with self._lock:
            error_info = {
                'error': error,
                'context': context or {},
                'timestamp': time.time(),
                'thread': threading.current_thread().name,
                'traceback': traceback.format_exc()
            }
            
            self._errors.append(error_info)
            
            # Keep only the most recent errors
            if len(self._errors) > self.max_errors:
                self._errors.pop(0)
            
            # Count error types
            error_type = type(error).__name__
            self._error_counts[error_type] = self._error_counts.get(error_type, 0) + 1
            
            # Log the error
            severity = ErrorSeverity.MEDIUM
            if isinstance(error, DaisyError):
                severity = error.severity
            
            self._log_error(error, severity, context)
    
    def _log_error(self, error: Exception, severity: ErrorSeverity, context: Optional[Dict[str, Any]]):
        """Log error with appropriate level."""
        message = f"{type(error).__name__}: {str(error)}"
        if context:
            message += f" | Context: {context}"
        
        if severity == ErrorSeverity.LOW:
            logger.debug(message)
        elif severity == ErrorSeverity.MEDIUM:
            logger.warning(message)
        elif severity == ErrorSeverity.HIGH:
            logger.error(message)
        else:  # CRITICAL
            logger.critical(message)
    
    def get_recent_errors(self, count: int = 10) -> list:
        """Get recent errors."""
        with self._lock:
            return self._errors[-count:]
    
    def get_error_counts(self) -> Dict[str, int]:
        """Get error counts by type."""
        with self._lock:
            return self._error_counts.copy()
    
    def clear_errors(self):
        """Clear all collected errors."""
        with self._lock:
            self._errors.clear()
            self._error_counts.clear()

# Global error collector
_error_collector = ErrorCollector()

def get_error_collector() -> ErrorCollector:
    """Get the global error collector."""
    return _error_collector

def handle_error(error: Exception, context: Optional[Dict[str, Any]] = None, 
                severity: Optional[ErrorSeverity] = None):
    """Handle an error by logging and collecting it."""
    if isinstance(error, DaisyError) and severity is None:
        severity = error.severity
    elif severity is None:
        severity = ErrorSeverity.MEDIUM
    
    # If it's not already a DaisyError, convert it
    if not isinstance(error, DaisyError):
        error = DaisyError(str(error), severity, context)
    
    _error_collector.add_error(error, context)

@contextmanager
def error_context(operation_name: str, reraise: bool = True, 
                 default_return: Any = None, context: Optional[Dict[str, Any]] = None):
    """
    Context manager for handling errors in operations.
    
    Args:
        operation_name: Name of the operation for logging
        reraise: Whether to reraise the exception
        default_return: Default return value if error occurs and not reraising
        context: Additional context for error logging
    """
    try:
        yield
    except Exception as e:
        error_context_info = {'operation': operation_name}
        if context:
            error_context_info.update(context)
        
        handle_error(e, error_context_info)
        
        if reraise:
            raise
        else:
            return default_return

def retry_on_error(max_retries: int = 3, delay: float = 1.0, 
                  backoff_factor: float = 2.0, exceptions: tuple = (Exception,)):
    """
    Decorator for retrying functions on specific exceptions.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries
        backoff_factor: Multiplier for delay on each retry
        exceptions: Tuple of exception types to retry on
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        # Final attempt failed
                        handle_error(e, {
                            'function': func.__name__,
                            'attempt': attempt + 1,
                            'max_retries': max_retries
                        })
                        raise
                    
                    # Log retry attempt
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                                 f"Retrying in {current_delay:.1f}s...")
                    
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
                except Exception as e:
                    # Different exception type, don't retry
                    handle_error(e, {
                        'function': func.__name__,
                        'attempt': attempt + 1
                    })
                    raise
            
            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator

def safe_execute(func: Callable, *args, default_return: Any = None, 
                error_message: str = "Operation failed", **kwargs) -> Any:
    """
    Safely execute a function, handling any errors.
    
    Args:
        func: Function to execute
        *args: Arguments for the function
        default_return: Value to return if function fails
        error_message: Custom error message for logging
        **kwargs: Keyword arguments for the function
        
    Returns:
        Function result or default_return if error occurs
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        handle_error(e, {
            'function': func.__name__ if hasattr(func, '__name__') else str(func),
            'custom_message': error_message
        })
        logger.error(f"{error_message}: {e}")
        return default_return

class CircuitBreaker:
    """Circuit breaker pattern for handling recurring failures."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0,
                 expected_exception: Type[Exception] = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.RLock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Call function through circuit breaker."""
        with self._lock:
            if self.state == 'OPEN':
                if self._should_attempt_reset():
                    self.state = 'HALF_OPEN'
                else:
                    raise ConnectionError(
                        f"Circuit breaker is OPEN. Too many failures.",
                        ErrorSeverity.HIGH,
                        {
                            'failure_count': self.failure_count,
                            'last_failure': self.last_failure_time
                        }
                    )
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker."""
        return (self.last_failure_time is not None and
                time.time() - self.last_failure_time >= self.recovery_timeout)
    
    def _on_success(self):
        """Handle successful operation."""
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

@contextmanager
def graceful_degradation(fallback_func: Callable = None, 
                        fallback_return: Any = None,
                        operation_name: str = "operation"):
    """
    Context manager that provides graceful degradation on errors.
    
    Args:
        fallback_func: Function to call as fallback
        fallback_return: Value to return as fallback
        operation_name: Name of the operation for logging
    """
    try:
        yield
    except Exception as e:
        logger.warning(f"Graceful degradation triggered for {operation_name}: {e}")
        handle_error(e, {'operation': operation_name, 'degraded': True})
        
        if fallback_func:
            try:
                return fallback_func()
            except Exception as fallback_error:
                logger.error(f"Fallback function also failed: {fallback_error}")
                handle_error(fallback_error, {'operation': f"{operation_name}_fallback"})
                return fallback_return
        else:
            return fallback_return

def log_performance(func: Callable) -> Callable:
    """Decorator to log function performance and errors."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} completed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            handle_error(e, {
                'function': func.__name__,
                'execution_time': execution_time,
                'args_count': len(args),
                'kwargs_count': len(kwargs)
            })
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    return wrapper 