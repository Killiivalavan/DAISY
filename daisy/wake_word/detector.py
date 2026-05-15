import asyncio
import logging
import numpy as np
from openwakeword.model import Model

logger = logging.getLogger(__name__)

class WakeWordDetector:
    def __init__(self, config, event_bus):
        self.config = config.wake_word
        self.event_bus = event_bus
        self.is_listening = False
        self._task = None
        
        logger.info(f"Initializing openWakeWord model from: {self.config.model}...")
        # Load the custom model file directly
        self.oww_model = Model(wakeword_models=[self.config.model])

    async def listen_loop(self, audio_stream):
        """
        Continuously reads from the async audio stream and checks for the wake word.
        """
        self.is_listening = True
        
        # Reset the model's internal feature buffer to prevent phantom triggers
        # caused by old audio from the previous conversation turn.
        self.oww_model.reset()
        
        # Flush the old audio queue so we don't process audio from 
        # while D.A.I.S.Y. was speaking or processing.
        audio_stream.flush()
                
        logger.info("Wake Word Detector listening...")
        
        while self.is_listening:
            try:
                # Read raw audio chunk (np.ndarray)
                chunk = await audio_stream.read()
                
                # Convert to 16-bit PCM if needed
                if chunk.dtype == np.float32:
                    chunk = (chunk * 32768).astype(np.int16)
                elif chunk.dtype != np.int16:
                    chunk = chunk.astype(np.int16)
                    
                # Ensure it's a 1D array
                if len(chunk.shape) > 1:
                    chunk = chunk.flatten()

                # Predict
                prediction = self.oww_model.predict(chunk)
                
                # --- Diagnostic Logging ---
                rms = np.sqrt(np.mean(chunk.astype(np.float32)**2))
                max_score = max(prediction.values()) if prediction else 0
                
                if not hasattr(self, '_log_counter'): self._log_counter = 0
                self._log_counter += 1
                if self._log_counter % 25 == 0:
                    logger.info(f"[WW Debug] Vol: {rms:.1f} | Max Score: {max_score:.3f}")
                # --------------------------

                for mdl_name, score in prediction.items():
                    if score >= self.config.threshold:
                        logger.info(f"Wake word '{mdl_name}' detected! (score: {score:.3f})")
                        self.is_listening = False # Stop self
                        await self.event_bus.publish("WAKE")
                        return # Exit loop completely
                            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in wake word loop: {e}")
                await asyncio.sleep(0.1)

        logger.debug("Wake Word Detector stopped.")

    def start(self, audio_stream):
        """Start the background listening loop."""
        self._task = asyncio.create_task(self.listen_loop(audio_stream))

    def stop(self):
        """Stop the background listening loop."""
        self.is_listening = False
        if self._task:
            self._task.cancel()
