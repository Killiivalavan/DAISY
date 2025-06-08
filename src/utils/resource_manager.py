"""
Resource manager for proper cleanup of system resources.
"""
import os
import threading
import logging
import weakref
from contextlib import contextmanager
from typing import Any, Dict, Set, Optional, IO
import tempfile

logger = logging.getLogger(__name__)

class ResourceTracker:
    """Tracks and manages system resources to ensure proper cleanup."""
    
    def __init__(self):
        self._resources: Dict[str, Any] = {}
        self._temp_files: Set[str] = set()
        self._file_handles: Set[IO] = set()
        self._lock = threading.RLock()
        self._cleanup_registered = False
    
    def register_resource(self, resource_id: str, resource: Any, cleanup_func=None):
        """
        Register a resource for tracking.
        
        Args:
            resource_id: Unique identifier for the resource
            resource: The resource object
            cleanup_func: Optional cleanup function to call when resource is released
        """
        with self._lock:
            if resource_id in self._resources:
                logger.warning(f"Resource {resource_id} is already registered")
                return
            
            self._resources[resource_id] = {
                'resource': resource,
                'cleanup_func': cleanup_func,
                'created_at': threading.current_thread().name
            }
            logger.debug(f"Registered resource: {resource_id}")
    
    def unregister_resource(self, resource_id: str) -> bool:
        """
        Unregister and cleanup a resource.
        
        Args:
            resource_id: Resource identifier
            
        Returns:
            True if resource was found and cleaned up, False otherwise
        """
        with self._lock:
            if resource_id not in self._resources:
                return False
            
            resource_info = self._resources.pop(resource_id)
            self._cleanup_resource(resource_id, resource_info)
            return True
    
    def _cleanup_resource(self, resource_id: str, resource_info: Dict[str, Any]):
        """Clean up a single resource."""
        try:
            cleanup_func = resource_info.get('cleanup_func')
            if cleanup_func:
                cleanup_func(resource_info['resource'])
            logger.debug(f"Cleaned up resource: {resource_id}")
        except Exception as e:
            logger.error(f"Error cleaning up resource {resource_id}: {e}")
    
    def cleanup_all(self):
        """Clean up all registered resources."""
        with self._lock:
            logger.info(f"Cleaning up {len(self._resources)} registered resources...")
            
            for resource_id, resource_info in list(self._resources.items()):
                self._cleanup_resource(resource_id, resource_info)
            
            self._resources.clear()
            
            # Clean up temporary files
            self._cleanup_temp_files()
            
            # Close file handles
            self._cleanup_file_handles()
    
    def register_temp_file(self, file_path: str):
        """Register a temporary file for cleanup."""
        with self._lock:
            self._temp_files.add(file_path)
            logger.debug(f"Registered temp file: {file_path}")
    
    def unregister_temp_file(self, file_path: str):
        """Unregister a temporary file."""
        with self._lock:
            self._temp_files.discard(file_path)
    
    def _cleanup_temp_files(self):
        """Clean up all temporary files."""
        for file_path in list(self._temp_files):
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug(f"Removed temp file: {file_path}")
            except Exception as e:
                logger.warning(f"Could not remove temp file {file_path}: {e}")
        self._temp_files.clear()
    
    def register_file_handle(self, file_handle: IO):
        """Register a file handle for cleanup."""
        with self._lock:
            self._file_handles.add(file_handle)
    
    def unregister_file_handle(self, file_handle: IO):
        """Unregister a file handle."""
        with self._lock:
            self._file_handles.discard(file_handle)
    
    def _cleanup_file_handles(self):
        """Close all registered file handles."""
        for handle in list(self._file_handles):
            try:
                if not handle.closed:
                    handle.close()
                    logger.debug("Closed file handle")
            except Exception as e:
                logger.warning(f"Could not close file handle: {e}")
        self._file_handles.clear()
    
    def get_resource_count(self) -> int:
        """Get the number of tracked resources."""
        with self._lock:
            return len(self._resources)
    
    def get_temp_file_count(self) -> int:
        """Get the number of tracked temporary files."""
        with self._lock:
            return len(self._temp_files)

# Global resource tracker instance
_global_tracker = ResourceTracker()

def get_resource_tracker() -> ResourceTracker:
    """Get the global resource tracker instance."""
    return _global_tracker

@contextmanager
def managed_temp_file(suffix=None, prefix=None, dir=None, delete=True):
    """
    Context manager for temporary files with automatic cleanup.
    
    Args:
        suffix: File suffix
        prefix: File prefix  
        dir: Directory to create file in
        delete: Whether to delete file on exit
        
    Yields:
        Temporary file path
    """
    temp_file = None
    try:
        temp_file = tempfile.NamedTemporaryFile(
            suffix=suffix, 
            prefix=prefix, 
            dir=dir, 
            delete=False
        )
        file_path = temp_file.name
        temp_file.close()
        
        if delete:
            _global_tracker.register_temp_file(file_path)
        
        yield file_path
        
    finally:
        if delete and temp_file:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                _global_tracker.unregister_temp_file(file_path)
            except Exception as e:
                logger.warning(f"Could not clean up temp file {file_path}: {e}")

@contextmanager
def managed_file(file_path: str, mode: str = 'r', **kwargs):
    """
    Context manager for file handles with automatic cleanup tracking.
    
    Args:
        file_path: Path to file
        mode: File open mode
        **kwargs: Additional arguments for open()
        
    Yields:
        File handle
    """
    file_handle = None
    try:
        file_handle = open(file_path, mode, **kwargs)
        _global_tracker.register_file_handle(file_handle)
        yield file_handle
        
    finally:
        if file_handle:
            try:
                if not file_handle.closed:
                    file_handle.close()
                _global_tracker.unregister_file_handle(file_handle)
            except Exception as e:
                logger.warning(f"Error closing file {file_path}: {e}")

@contextmanager
def managed_resource(resource_id: str, resource: Any, cleanup_func=None):
    """
    Context manager for general resource management.
    
    Args:
        resource_id: Unique identifier for resource
        resource: The resource object
        cleanup_func: Optional cleanup function
        
    Yields:
        The resource object
    """
    try:
        _global_tracker.register_resource(resource_id, resource, cleanup_func)
        yield resource
        
    finally:
        _global_tracker.unregister_resource(resource_id)

class AudioStreamManager:
    """Manages audio streams with proper cleanup."""
    
    def __init__(self):
        self._active_streams = {}
        self._lock = threading.RLock()
    
    @contextmanager
    def get_input_stream(self, **kwargs):
        """
        Get an input audio stream with automatic cleanup.
        
        Args:
            **kwargs: Arguments for sounddevice.InputStream
            
        Yields:
            Audio stream object
        """
        import sounddevice as sd
        
        stream_id = f"input_{threading.get_ident()}_{id(kwargs)}"
        stream = None
        
        try:
            with self._lock:
                # Create stream
                stream = sd.InputStream(**kwargs)
                self._active_streams[stream_id] = stream
                stream.start()
            
            yield stream
            
        finally:
            with self._lock:
                if stream_id in self._active_streams:
                    try:
                        stream = self._active_streams.pop(stream_id)
                        if stream.active:
                            stream.stop()
                        stream.close()
                        logger.debug(f"Closed audio stream: {stream_id}")
                    except Exception as e:
                        logger.warning(f"Error closing audio stream {stream_id}: {e}")
    
    @contextmanager
    def get_output_stream(self, **kwargs):
        """
        Get an output audio stream with automatic cleanup.
        
        Args:
            **kwargs: Arguments for sounddevice.OutputStream
            
        Yields:
            Audio stream object
        """
        import sounddevice as sd
        
        stream_id = f"output_{threading.get_ident()}_{id(kwargs)}"
        stream = None
        
        try:
            with self._lock:
                # Create stream
                stream = sd.OutputStream(**kwargs)
                self._active_streams[stream_id] = stream
                stream.start()
            
            yield stream
            
        finally:
            with self._lock:
                if stream_id in self._active_streams:
                    try:
                        stream = self._active_streams.pop(stream_id)
                        if stream.active:
                            stream.stop()
                        stream.close()
                        logger.debug(f"Closed audio stream: {stream_id}")
                    except Exception as e:
                        logger.warning(f"Error closing audio stream {stream_id}: {e}")
    
    def cleanup_all_streams(self):
        """Close all active streams."""
        with self._lock:
            for stream_id, stream in list(self._active_streams.items()):
                try:
                    if stream.active:
                        stream.stop()
                    stream.close()
                    logger.debug(f"Closed audio stream: {stream_id}")
                except Exception as e:
                    logger.warning(f"Error closing audio stream {stream_id}: {e}")
            self._active_streams.clear()

# Global audio stream manager
_audio_manager = AudioStreamManager()

def get_audio_manager() -> AudioStreamManager:
    """Get the global audio stream manager."""
    return _audio_manager

def cleanup_all_resources():
    """Clean up all tracked resources."""
    logger.info("Performing global resource cleanup...")
    _global_tracker.cleanup_all()
    _audio_manager.cleanup_all_streams()

# Register cleanup function to be called on exit
import atexit
atexit.register(cleanup_all_resources) 