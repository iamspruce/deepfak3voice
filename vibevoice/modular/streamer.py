from __future__ import annotations

import torch
import asyncio
import threading
from queue import Queue, Empty
from typing import TYPE_CHECKING, Optional, Any

from transformers.generation import BaseStreamer


class AudioStreamer(BaseStreamer):
    """
    Audio streamer that stores audio chunks in queues for each sample in the batch.
    Thread-safe implementation for synchronous contexts.
    
    Parameters:
        batch_size (`int`):
            The batch size for generation
        stop_signal (`any`, *optional*):
            The signal to put in the queue when generation ends. Defaults to None.
        timeout (`float`, *optional*):
            The timeout for the audio queue. If `None`, the queue will block indefinitely.
    """
    
    def __init__(
        self, 
        batch_size: int,
        stop_signal: Optional[Any] = None,
        timeout: Optional[float] = None,
    ):
        self.batch_size = batch_size
        self.stop_signal = stop_signal
        self.timeout = timeout
        
        # Create a queue for each sample in the batch
        self.audio_queues = [Queue() for _ in range(batch_size)]
        self.finished_flags = [False for _ in range(batch_size)]
        
    def put(self, audio_chunks: torch.Tensor, sample_indices: torch.Tensor):
        """
        Receives audio chunks and puts them in the appropriate queues.
        
        Args:
            audio_chunks: Tensor of shape (num_samples, ...) containing audio chunks
            sample_indices: Tensor indicating which samples these chunks belong to
        """
        for i, sample_idx in enumerate(sample_indices):
            idx = sample_idx.item()
            if idx < self.batch_size and not self.finished_flags[idx]:
                audio_chunk = audio_chunks[i].detach().cpu()
                self.audio_queues[idx].put(audio_chunk, timeout=self.timeout)
    
    def end(self, sample_indices: Optional[torch.Tensor] = None):
        """
        Signals the end of generation for specified samples or all samples.
        
        Args:
            sample_indices: Optional tensor of sample indices to end. If None, ends all.
        """
        if sample_indices is None:
            for idx in range(self.batch_size):
                if not self.finished_flags[idx]:
                    self.audio_queues[idx].put(self.stop_signal, timeout=self.timeout)
                    self.finished_flags[idx] = True
        else:
            for sample_idx in sample_indices:
                idx = sample_idx.item() if torch.is_tensor(sample_idx) else sample_idx
                if idx < self.batch_size and not self.finished_flags[idx]:
                    self.audio_queues[idx].put(self.stop_signal, timeout=self.timeout)
                    self.finished_flags[idx] = True
    
    def get_stream(self, sample_idx: int):
        """Get the audio stream for a specific sample."""
        if sample_idx >= self.batch_size:
            raise ValueError(f"Sample index {sample_idx} exceeds batch size {self.batch_size}")
        return AudioSampleIterator(self, sample_idx)


class AudioSampleIterator:
    """Iterator for a single audio stream from the batch."""
    
    def __init__(self, streamer: AudioStreamer, sample_idx: int):
        self.streamer = streamer
        self.sample_idx = sample_idx
        
    def __iter__(self):
        return self
    
    def __next__(self):
        value = self.streamer.audio_queues[self.sample_idx].get(timeout=self.streamer.timeout)
        if value == self.streamer.stop_signal:
            raise StopIteration()
        return value


class AsyncAudioStreamer(AudioStreamer):
    """
    Async version of AudioStreamer for use in async contexts.
    FIXED: Handles event loop lazily and supports thread-safe queue operations.
    """
    
    def __init__(
        self, 
        batch_size: int,
        stop_signal: Optional[Any] = None,
        timeout: Optional[float] = None,
    ):
        # Don't call super().__init__ to avoid creating sync queues
        self.batch_size = batch_size
        self.stop_signal = stop_signal
        self.timeout = timeout
        
        # Use thread-safe queues that work across threads
        self.audio_queues = [Queue(maxsize=100) for _ in range(batch_size)]
        self.finished_flags = [False for _ in range(batch_size)]
        
        # Event loop will be set when needed
        self._loop = None
        self._lock = threading.Lock()
        
    def _ensure_loop(self):
        """Lazily get or store the event loop reference."""
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                # No loop running, will be set later
                pass
        return self._loop
    
    def put(self, audio_chunks: torch.Tensor, sample_indices: torch.Tensor):
        """
        Put audio chunks in the appropriate queues (thread-safe).
        This is called from the model's generation thread.
        """
        for i, sample_idx in enumerate(sample_indices):
            idx = sample_idx.item()
            if idx < self.batch_size and not self.finished_flags[idx]:
                audio_chunk = audio_chunks[i].detach().cpu()
                
                try:
                    # Non-blocking put with timeout
                    self.audio_queues[idx].put(audio_chunk, block=True, timeout=1.0)
                except Exception as e:
                    print(f"Warning: Failed to put audio chunk in queue {idx}: {e}")
    
    def end(self, sample_indices: Optional[torch.Tensor] = None):
        """Signal the end of generation for specified samples (thread-safe)."""
        if sample_indices is None:
            indices_to_end = range(self.batch_size)
        else:
            indices_to_end = [s.item() if torch.is_tensor(s) else s for s in sample_indices]
            
        for idx in indices_to_end:
            if idx < self.batch_size and not self.finished_flags[idx]:
                try:
                    self.audio_queues[idx].put(self.stop_signal, block=False)
                except Exception as e:
                    print(f"Warning: Failed to put stop signal in queue {idx}: {e}")
                finally:
                    self.finished_flags[idx] = True
    
    async def get_stream(self, sample_idx: int):
        """
        Get async iterator for a specific sample's audio stream.
        FIXED: Uses asyncio.to_thread to safely read from Queue in async context.
        """
        if sample_idx >= self.batch_size:
            raise ValueError(f"Sample index {sample_idx} exceeds batch size {self.batch_size}")
        
        # Ensure we have the event loop reference
        self._ensure_loop()
        
        while True:
            try:
                # Use to_thread to safely get from Queue without blocking event loop
                value = await asyncio.to_thread(
                    self.audio_queues[sample_idx].get,
                    timeout=self.timeout or 30.0
                )
                
                if value == self.stop_signal:
                    print(f"Stream {sample_idx} ended")
                    break
                    
                yield value
                
            except Empty:
                # Queue is empty but not finished
                if self.finished_flags[sample_idx]:
                    break
                # Small delay to avoid tight loop
                await asyncio.sleep(0.01)
            except Exception as e:
                print(f"Error in get_stream for sample {sample_idx}: {e}")
                break
    
    def has_data(self, sample_idx: int) -> bool:
        """Check if a queue has data available."""
        return not self.audio_queues[sample_idx].empty()
    
    def clear(self, sample_idx: Optional[int] = None):
        """Clear queue(s) of any remaining data."""
        if sample_idx is not None:
            # Clear specific queue
            while not self.audio_queues[sample_idx].empty():
                try:
                    self.audio_queues[sample_idx].get_nowait()
                except Empty:
                    break
        else:
            # Clear all queues
            for idx in range(self.batch_size):
                while not self.audio_queues[idx].empty():
                    try:
                        self.audio_queues[idx].get_nowait()
                    except Empty:
                        break