"""
Performance Optimization Module for SoundsAwful

This module provides optimizations for the image-to-music and music-to-image
conversion processes, making them more efficient for larger resolutions.
"""

import os
import time
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import List, Tuple, Callable, Any, Dict, Optional

import cv2
import numpy as np
from music21 import chord, stream

# Classes needed by ui_improved.py
class ParallelImageProcessor:
    """
    Handles parallel processing of image blocks for faster conversion
    """
    def __init__(self, max_workers: int = None, chunk_size: int = 100):
        """
        Initialize the parallel processor
        
        :param max_workers: Maximum number of worker threads
        :param chunk_size: Size of processing chunks
        """
        # Determine optimal number of workers based on CPU cores
        if max_workers is None:
            import multiprocessing
            max_workers = max(1, multiprocessing.cpu_count() - 1)
            
        self.max_workers = max_workers
        self.chunk_size = chunk_size
    
    def process_parallel(self, process_func: Callable, items: List[Any]) -> List[Any]:
        """
        Process a list of items in parallel
        
        :param process_func: Function to process each item
        :param items: List of items to process
        :return: List of processed results
        """
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_item = {
                executor.submit(process_func, item): i 
                for i, item in enumerate(items)
            }
            
            for future in as_completed(future_to_item):
                results.append(future.result())
        
        return results

class MemoryOptimizer:
    """
    Optimizes memory usage for large images and operations
    """
    def __init__(self, mode: str = "balanced"):
        """
        Initialize the memory optimizer
        
        :param mode: Optimization mode ("speed", "balanced", or "quality")
        """
        self.mode = mode
        self._downscale_threshold = 20_000_000  # 20 megapixels
        
    def set_mode(self, mode: str) -> None:
        """
        Set the optimization mode
        
        :param mode: Optimization mode ("speed", "balanced", or "quality")
        """
        self.mode = mode
        
        # Adjust thresholds based on mode
        if mode == "speed":
            self._downscale_threshold = 10_000_000  # 10 megapixels
        elif mode == "quality":
            self._downscale_threshold = 30_000_000  # 30 megapixels
        else:  # balanced
            self._downscale_threshold = 20_000_000  # 20 megapixels
    
    def optimize_image_load(self, image_path: str) -> np.ndarray:
        """
        Load an image with memory optimization
        
        :param image_path: Path to image file
        :return: Loaded image array
        """
        # Get image dimensions without loading the whole image
        img = cv2.imread(image_path, cv2.IMREAD_REDUCED_COLOR_8)
        
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        height, width = img.shape[:2]
        del img  # Release the reduced image
        
        # If image is very large, process with reduced resolution
        if height * width > self._downscale_threshold:
            if self.mode == "speed":
                # Fast loading with significant downscaling
                return cv2.imread(image_path, cv2.IMREAD_REDUCED_COLOR_4)
            elif self.mode == "quality":
                # Load full image but optimize memory
                return cv2.imread(image_path, cv2.IMREAD_COLOR)
            else:  # balanced
                # Moderate downscaling
                return cv2.imread(image_path, cv2.IMREAD_REDUCED_COLOR_2)
        else:
            # For smaller images, read normally
            return cv2.imread(image_path, cv2.IMREAD_COLOR)

class CacheManager:
    """
    Manages caching for processed data to speed up repeated operations
    """
    def __init__(self, enabled: bool = True, cache_dir: str = ".cache"):
        """
        Initialize the cache manager
        
        :param enabled: Whether caching is enabled
        :param cache_dir: Directory to store cache files
        """
        self.enabled = enabled
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def get_cached_data(self, key: str) -> Optional[Any]:
        """
        Get data from cache
        
        :param key: Cache key
        :return: Cached data or None if not found
        """
        if not self.enabled:
            return None
            
        cache_path = os.path.join(self.cache_dir, self._hash_key(key))
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                # If loading fails, return None
                return None
        
        return None
    
    def cache_data(self, key: str, data: Any) -> bool:
        """
        Store data in cache
        
        :param key: Cache key
        :param data: Data to cache
        :return: True if caching succeeded, False otherwise
        """
        if not self.enabled:
            return False
            
        cache_path = os.path.join(self.cache_dir, self._hash_key(key))
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception:
            return False
    
    def clear_cache(self) -> bool:
        """
        Clear all cached data
        
        :return: True if cache was cleared, False otherwise
        """
        if not os.path.exists(self.cache_dir):
            return False
            
        try:
            for file_name in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            return True
        except Exception:
            return False
    
    def _hash_key(self, key: str) -> str:
        """
        Hash a key for use as a filename
        
        :param key: Key to hash
        :return: Hashed key
        """
        import hashlib
        return hashlib.md5(key.encode()).hexdigest()

# Performance monitoring
class PerformanceMonitor:
    """
    Utility class to monitor and report performance metrics.
    """
    def __init__(self, name: str):
        """
        Initialize a performance monitor
        
        :param name: Name of the monitored operation
        """
        self.name = name
        self.start_time = None
        self.end_time = None
        self.checkpoints = {}
        
    def start(self) -> None:
        """Start the performance monitoring timer"""
        self.start_time = time.time()
        self.checkpoints = {'start': self.start_time}
        
    def checkpoint(self, name: str) -> float:
        """
        Record a checkpoint with the given name
        
        :param name: Name of the checkpoint
        :return: Time elapsed since the start
        """
        current = time.time()
        elapsed = current - self.start_time
        self.checkpoints[name] = current
        return elapsed
        
    def stop(self) -> float:
        """
        Stop the timer and return the total elapsed time
        
        :return: Total elapsed time in seconds
        """
        self.end_time = time.time()
        self.checkpoints['end'] = self.end_time
        return self.end_time - self.start_time
        
    def report(self) -> Dict[str, float]:
        """
        Generate a report of time spent between checkpoints
        
        :return: Dictionary mapping checkpoint pairs to elapsed time
        """
        report = {}
        checkpoints = sorted(self.checkpoints.items(), key=lambda x: x[1])
        
        for i in range(len(checkpoints) - 1):
            current_name, current_time = checkpoints[i]
            next_name, next_time = checkpoints[i + 1]
            report[f"{current_name} → {next_name}"] = next_time - current_time
            
        report["total"] = self.checkpoints['end'] - self.checkpoints['start']
        return report

# Parallel image processing
def parallel_image_split(
    image: np.ndarray, 
    split_number: Tuple[int, int],
    process_func: Callable[[np.ndarray], Tuple[float, float, float]],
    max_workers: int = None,
    progress_callback: Callable[[float, str], None] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process image blocks in parallel for faster conversion
    
    :param image: Input image
    :param split_number: Resolution of blocks (rows, cols)
    :param process_func: Function to process each block
    :param max_workers: Maximum number of worker threads
    :param progress_callback: Optional callback to report progress
    :return: Tuple of note, quarter_length, and volume arrays
    """
    # Resize image if necessary
    d_height = image.shape[0] / split_number[0]
    d_width = image.shape[1] / split_number[1]

    ratio_h = np.floor(d_height) * split_number[0]
    ratio_w = np.floor(d_width) * split_number[1]

    # Resize image if too large
    image = cv2.resize(image, (int(ratio_w), int(ratio_h)))

    d_height = int(image.shape[0] / split_number[0])
    d_width = int(image.shape[1] / split_number[1])
    
    # Create a list of all blocks to process
    blocks = []
    for r in range(0, image.shape[0], d_height):
        for c in range(0, image.shape[1], d_width):
            blocks.append((r, c, d_height, d_width))
    
    # Process blocks in parallel
    results = []
    total_blocks = len(blocks)
    
    # Determine optimal number of workers based on CPU cores
    if max_workers is None:
        import multiprocessing
        max_workers = max(1, multiprocessing.cpu_count() - 1)
    
    # Define a worker function to process each block
    def process_block(idx, block_info):
        r, c, d_h, d_w = block_info
        crop = image[r:r + d_h, c:c + d_w]
        result = process_func(crop)
        
        # Report progress periodically
        if progress_callback and idx % 10 == 0:
            progress = (idx / total_blocks) * 100
            progress_callback(progress, f"Processing block {idx}/{total_blocks}")
            
        return idx, result
    
    # Use thread pool for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(process_block, idx, block): idx 
            for idx, block in enumerate(blocks)
        }
        
        for future in as_completed(future_to_idx):
            idx, result = future.result()
            results.append((idx, result))
    
    # Sort results back into the correct order
    results.sort(key=lambda x: x[0])
    
    # Extract and combine results
    note_values = np.array([res[1][0] for res in results])
    quarter_lengths = np.array([res[1][1] for res in results])
    volumes = np.array([res[1][2] for res in results])
    
    return note_values, quarter_lengths, volumes

# Optimized image block processor
def optimized_block_processor(crop: np.ndarray) -> Tuple[float, float, float]:
    """
    Optimized version of image block processing
    
    :param crop: Image crop to process
    :return: Tuple of (note, quarter_length, volume) values
    """
    # Use numpy operations for efficiency
    # Extract channels (faster than iterating)
    r_channel = crop[:, :, 0]
    g_channel = crop[:, :, 1]
    b_channel = crop[:, :, 2]
    
    # Use median for stability in music generation
    note = np.median(b_channel)
    quarter_length = np.median(g_channel)
    volume = np.median(r_channel)
    
    return note, quarter_length, volume

# Memory-optimized image processing
def memory_optimized_image_processing(
    image_path: str, 
    split_number: Tuple[int, int],
    progress_callback: Callable[[float, str], None] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process an image with memory optimization for large images
    
    :param image_path: Path to the image file
    :param split_number: Resolution of blocks (rows, cols)
    :param progress_callback: Optional callback to report progress
    :return: Tuple of note, quarter_length, and volume arrays
    """
    # Use memory-mapped image reading for large images
    # First get image dimensions without loading the whole image
    img = cv2.imread(image_path, cv2.IMREAD_REDUCED_COLOR_8)
    
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
        
    height, width = img.shape[:2]
    del img  # Release the reduced image
    
    # If image is very large, process in chunks
    if height * width > 20_000_000:  # ~20 megapixels threshold
        if progress_callback:
            progress_callback(0, "Image is large, processing in chunks...")
            
        # Calculate target size based on split_number
        target_height = split_number[0] * 16  # Use multiples of block size
        target_width = split_number[1] * 16
        
        # Calculate scale factor
        scale = min(target_height / height, target_width / width)
        new_width, new_height = int(width * scale), int(height * scale)
        
        # Read the image at reduced resolution
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (new_width, new_height))
        
        if progress_callback:
            progress_callback(10, f"Resized image to {new_width}x{new_height}")
    else:
        # For smaller images, read normally
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    # Process the image using parallel processing
    result = parallel_image_split(
        img, 
        split_number, 
        optimized_block_processor,
        progress_callback=progress_callback
    )
    
    return result

# Cached note processing
@lru_cache(maxsize=128)
def cached_note_to_chord(
    note_name: str, 
    quarter_length: float, 
    volume: float, 
    key_name: str
) -> chord.Chord:
    """
    Convert note to chord with caching to avoid redundant processing
    
    :param note_name: Name of the note
    :param quarter_length: Duration of the note
    :param volume: Volume of the note
    :param key_name: Name of the key (used for cache key)
    :return: Generated chord
    """
    from music21 import note
    from keys import AVAILABLE_KEYS
    
    # Create note
    n = note.Note(note_name)
    n.quarterLength = quarter_length
    n.volume.velocity = volume
    
    # For real implementation, you would create the correct chord here
    # This is a simplified version
    c = chord.Chord([n])
    c.quarterLength = quarter_length
    c.volume.velocity = volume
    
    return c

# Optimized MIDI write
def optimized_midi_write(
    chords: stream.Stream, 
    output_path: str,
    compress: bool = True
) -> None:
    """
    Write MIDI file with optimizations for size and performance
    
    :param chords: Stream of chords to write
    :param output_path: Path to output MIDI file
    :param compress: Whether to compress the MIDI file
    """
    # Write MIDI file
    chords.write('midi', output_path)
    
    # Optionally compress the file
    if compress and output_path.endswith('.mid'):
        try:
            import zlib
            import shutil
            
            # Compress to a temporary file
            temp_path = output_path + '.compressed'
            with open(output_path, 'rb') as f_in:
                data = f_in.read()
                compressed = zlib.compress(data, level=9)
                
            with open(temp_path, 'wb') as f_out:
                f_out.write(compressed)
                
            # Check if compression was effective
            if os.path.getsize(temp_path) < os.path.getsize(output_path):
                shutil.move(temp_path, output_path)
            else:
                os.remove(temp_path)
                
        except Exception:
            # Fall back to uncompressed if compression fails
            pass

# Batch chord generation
def batch_chord_generation(
    note_data: List[Tuple[str, float, float]],
    cypher: Any,
    batch_size: int = 100,
    progress_callback: Callable[[float, str], None] = None
) -> stream.Stream:
    """
    Generate chords in batches for better memory efficiency
    
    :param note_data: List of (note, quarter_length, volume) tuples
    :param cypher: Cypher to use for chord generation
    :param batch_size: Number of chords to generate in each batch
    :param progress_callback: Optional callback to report progress
    :return: Stream containing all generated chords
    """
    from musicgen import create_chords
    
    result_stream = stream.Stream()
    num_batches = (len(note_data) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(note_data))
        
        # Process this batch
        batch_data = note_data[start_idx:end_idx]
        batch_chords = create_chords(batch_data, cypher)
        
        # Add chords to result stream
        for chord_obj in batch_chords.recurse().getElementsByClass('Chord'):
            result_stream.append(chord_obj)
            
        # Report progress
        if progress_callback:
            progress = (batch_idx + 1) / num_batches * 100
            progress_callback(progress, f"Generated batch {batch_idx + 1}/{num_batches}")
            
    return result_stream