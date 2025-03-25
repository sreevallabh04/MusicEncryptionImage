import os
import platform
import subprocess
import threading
import tkinter as tk
import multiprocessing
import time
from io import TextIOWrapper
from tkinter import ttk, messagebox
from tkinter.filedialog import askopenfile, asksaveasfile
from typing import List, Tuple, Optional, Dict, Any, Union

import cv2
import numpy as np
import pygubu

import musicgen
from keys import *
# Import our new modules
import steganography
from performance import ParallelImageProcessor, MemoryOptimizer, CacheManager

# Default settings
DEFAULT_KEY = A_MINOR
DEFAULT_RESOLUTION = (64, 64)
DEFAULT_STEG_METHOD = "bass_note"
CACHE_ENABLED = True
AVAILABLE_KEYS = {
    "A Minor": A_MINOR,
    "C Major": C_MAJOR,
    "G Major": G_MAJOR,
    "D Minor": D_MINOR,
    "F Major": F_MAJOR,
    "E Minor": E_MINOR
}

# Steganography methods
STEGANOGRAPHY_METHODS = {
    "Bass Note": "bass_note",
    "Note Duration": "duration",
    "Note Volume": "volume",
    "Metadata": "metadata",
    "Pattern-based": "pattern"
}

# Constants for conversion parameters
MIN_QUARTER_LENGTH = 0.25
MAX_QUARTER_LENGTH = 2.0
MIN_VOLUME = 20
MAX_VOLUME = 127

# Performance settings
MAX_WORKERS = multiprocessing.cpu_count()
CHUNK_SIZE = 100  # For parallel processing

"""
Enhanced Python GUI for Image-Music Cryptography
"""

class ImageAudioConverter:
    def __init__(self, notes: List[str]) -> None:
        """
        Initialize the GUI with enhanced features
        
        :param notes: the available notes for the audio conversion to use
        """
        # Operating system type
        self.platform = platform.system()
        
        # Setup builder and variables
        self.notes = notes
        self.available_notes = notes
        self.builder = pygubu.Builder()
        
        # Create main window
        self.builder.add_from_file('converter.ui')
        self.mainwindow = self.builder.get_object('mainwindow')
        self.mainwindow.title("SoundsAwful - Image-Music Cryptography")
        
        # Initialize variables
        self.file = None
        self.selected_key = tk.StringVar(value=list(AVAILABLE_KEYS.keys())[0])
        self.selected_steg_method = tk.StringVar(value=list(STEGANOGRAPHY_METHODS.keys())[0])
        self.image_preview = None
        self.progress_var = tk.DoubleVar(value=0.0)
        self.steganography_enabled = tk.BooleanVar(value=False)
        self.hide_throughout = tk.BooleanVar(value=False)
        self.performance_mode = tk.StringVar(value="Balanced")
        self.use_caching = tk.BooleanVar(value=CACHE_ENABLED)
        
        # Initialize performance components
        self.processor = ParallelImageProcessor(max_workers=MAX_WORKERS)
        self.memory_optimizer = MemoryOptimizer()
        self.cache_manager = CacheManager()
        
        # Connect basic callbacks from UI file
        self.builder.connect_callbacks(self)
        
        # Load default values
        self.builder.tkvariables['row_size'].set(DEFAULT_RESOLUTION[0])
        self.builder.tkvariables['col_size'].set(DEFAULT_RESOLUTION[1])
        
        # Enhance the UI
        self._enhance_ui()

    def _enhance_ui(self) -> None:
        """
        Enhance the UI with additional features beyond the base converter.ui
        """
        # Add a frame for the image preview
        preview_frame = ttk.LabelFrame(self.mainwindow, text="Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add image preview area
        self.preview_label = ttk.Label(preview_frame)
        self.preview_label.pack(padx=10, pady=10)
        
        # Add advanced options frame
        options_frame = ttk.LabelFrame(self.mainwindow, text="Advanced Options")
        options_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Add key selection
        key_frame = ttk.Frame(options_frame)
        key_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(key_frame, text="Musical Key:").pack(side=tk.LEFT, padx=5)
        key_combobox = ttk.Combobox(key_frame, textvariable=self.selected_key, 
                                   values=list(AVAILABLE_KEYS.keys()),
                                   state="readonly")
        key_combobox.pack(side=tk.LEFT, padx=5)
        
        # Add steganography options with more detailed controls
        steg_frame = ttk.LabelFrame(options_frame, text="Steganography Options")
        steg_frame.pack(fill=tk.X, padx=5, pady=5)
        
        steg_enable_frame = ttk.Frame(steg_frame)
        steg_enable_frame.pack(fill=tk.X, pady=2)
        
        ttk.Checkbutton(steg_enable_frame, text="Enable Steganography", 
                      variable=self.steganography_enabled,
                      command=self._toggle_steg_options).pack(side=tk.LEFT, padx=5)
        
        ttk.Checkbutton(steg_enable_frame, text="Hide Throughout Music", 
                      variable=self.hide_throughout).pack(side=tk.LEFT, padx=5)
        
        # Method selection
        self.steg_method_frame = ttk.Frame(steg_frame)
        self.steg_method_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(self.steg_method_frame, text="Method:").pack(side=tk.LEFT, padx=5)
        method_combobox = ttk.Combobox(self.steg_method_frame, 
                                     textvariable=self.selected_steg_method,
                                     values=list(STEGANOGRAPHY_METHODS.keys()),
                                     state="readonly")
        method_combobox.pack(side=tk.LEFT, padx=5)
        
        # Add performance options
        perf_frame = ttk.LabelFrame(options_frame, text="Performance Settings")
        perf_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Performance mode selection
        perf_mode_frame = ttk.Frame(perf_frame)
        perf_mode_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(perf_mode_frame, text="Mode:").pack(side=tk.LEFT, padx=5)
        ttk.Combobox(perf_mode_frame, textvariable=self.performance_mode,
                   values=["Speed", "Balanced", "Quality"],
                   state="readonly").pack(side=tk.LEFT, padx=5)
        
        # Caching option
        cache_frame = ttk.Frame(perf_frame)
        cache_frame.pack(fill=tk.X, pady=2)
        
        ttk.Checkbutton(cache_frame, text="Enable Caching", 
                      variable=self.use_caching).pack(side=tk.LEFT, padx=5)
        
        # Add thread count slider
        thread_frame = ttk.Frame(perf_frame)
        thread_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(thread_frame, text=f"Threads (1-{MAX_WORKERS}):").pack(side=tk.LEFT, padx=5)
        self.thread_slider = ttk.Scale(thread_frame, from_=1, to=MAX_WORKERS, 
                                    orient=tk.HORIZONTAL, length=150)
        self.thread_slider.set(MAX_WORKERS // 2)  # Start with half the available threads
        self.thread_slider.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Thread count label
        self.thread_count_label = ttk.Label(thread_frame, text=str(MAX_WORKERS // 2))
        self.thread_count_label.pack(side=tk.LEFT, padx=5)
        
        # Update thread count label when slider is moved
        self.thread_slider.configure(command=self._update_thread_count)
        
        # Add progress bar
        progress_frame = ttk.Frame(self.mainwindow)
        progress_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                          length=300, mode='determinate')
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        # Add status label
        self.status_label = ttk.Label(progress_frame, text="Ready")
        self.status_label.pack(padx=5)
        
        # Theme configuration for better appearance
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat", background="#ccc")
        
        # Add button frame
        button_frame = ttk.Frame(self.mainwindow)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Add a help button
        help_button = ttk.Button(button_frame, text="Help", command=self.show_help)
        help_button.pack(side=tk.RIGHT, padx=5)
        
        # Add playback preview button (for MIDI files)
        self.preview_button = ttk.Button(button_frame, text="Preview", 
                                      command=self.preview_audio, state=tk.DISABLED)
        self.preview_button.pack(side=tk.RIGHT, padx=5)
        
        # Add clear cache button
        clear_cache_button = ttk.Button(button_frame, text="Clear Cache", 
                                      command=self.clear_cache)
        clear_cache_button.pack(side=tk.RIGHT, padx=5)
        
        # Disable steganography options initially
        self._toggle_steg_options()
    def _toggle_steg_options(self) -> None:
        """
        Enable or disable steganography options based on checkbox state
        """
        state = tk.NORMAL if self.steganography_enabled.get() else tk.DISABLED
        for child in self.steg_method_frame.winfo_children():
            child.configure(state=state)
    
    def _update_thread_count(self, value) -> None:
        """
        Update the thread count label when slider is moved
        
        :param value: Current slider value
        """
        thread_count = int(float(value))
        self.thread_count_label.config(text=str(thread_count))
        self.processor.max_workers = thread_count
    
    def clear_cache(self) -> None:
        """
        Clear the cache of processed images and conversion data
        """
        try:
            cleared = self.cache_manager.clear_cache()
            if cleared:
                messagebox.showinfo("Cache Cleared", "Image processing cache has been cleared.")
            else:
                messagebox.showinfo("Cache Empty", "No cache to clear.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not clear cache: {str(e)}")
    
    def preview_audio(self) -> None:
        """
        Preview the generated MIDI file
        """
        if not hasattr(self, 'last_midi_file') or not self.last_midi_file:
            messagebox.showinfo("Preview", "No MIDI file available for preview.")
            return
            
        try:
            # Implement a simple MIDI player
            from pygame import mixer
            import pygame
            
            pygame.init()
            mixer.init()
            
            try:
                mixer.music.load(self.last_midi_file)
                mixer.music.play()
                
                # Create a simple playback control window
                play_window = tk.Toplevel(self.mainwindow)
                play_window.title("Audio Preview")
                play_window.geometry("300x150")
                
                # Add playback controls
                control_frame = ttk.Frame(play_window)
                control_frame.pack(fill=tk.X, padx=10, pady=10)
                
                # Play/Pause button
                play_paused = [False]  # Use a list to make it mutable in nested function
                
                def toggle_play():
                    if play_paused[0]:
                        mixer.music.unpause()
                        play_button.config(text="Pause")
                    else:
                        mixer.music.pause()
                        play_button.config(text="Play")
                    play_paused[0] = not play_paused[0]
                
                play_button = ttk.Button(control_frame, text="Pause", command=toggle_play)
                play_button.pack(side=tk.LEFT, padx=5)
                
                # Stop button
                def stop_playback():
                    mixer.music.stop()
                    play_window.destroy()
                
                stop_button = ttk.Button(control_frame, text="Stop", command=stop_playback)
                stop_button.pack(side=tk.LEFT, padx=5)
                
                # File info
                info_frame = ttk.Frame(play_window)
                info_frame.pack(fill=tk.X, padx=10, pady=5)
                
                ttk.Label(info_frame, text=f"Playing: {os.path.basename(self.last_midi_file)}").pack()
                
                # Handle window close
                play_window.protocol("WM_DELETE_WINDOW", stop_playback)
                
            except Exception as e:
                mixer.quit()
                pygame.quit()
                raise e
                
        except Exception as e:
            messagebox.showerror("Preview Error", f"Could not play MIDI: {str(e)}")
    def run(self) -> None:
        """
        Starts the GUI main loop, keeps the GUI alive
        """
        self.mainwindow.mainloop()
    def select_file(self) -> None:
        """
        Asks the user to select a file and shows a preview if it's an image
        """
        current_dir = os.getcwd()

        self.file: TextIOWrapper = askopenfile(initialdir=current_dir,
                                            filetypes=(('Images', "*.jpg;*.png"), ("Audio", "*.mid")))

        if self.file is not None:
            self.builder.tkvariables['file_location_var'].set(os.path.basename(self.file.name))
            self.show_file_preview(self.file.name)
            self.status_label.config(text=f"File selected: {os.path.basename(self.file.name)}")

    def show_file_preview(self, file_path: str) -> None:
        """
        Display a preview of the selected file (image or audio waveform)
        
        :param file_path: Path to the selected file
        """
        if file_path.endswith('.jpg') or file_path.endswith('.png'):
            try:
                # Create a thumbnail for the image
                # Use memory optimizer for large images
                image = self.memory_optimizer.optimize_image_load(file_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                
                # Resize for preview (maintain aspect ratio)
                height, width = image.shape[:2]
                max_size = 300
                scale = min(max_size / width, max_size / height)
                new_width, new_height = int(width * scale), int(height * scale)
                
                image = cv2.resize(image, (new_width, new_height))
                
                # Convert to PhotoImage for tkinter
                from PIL import Image, ImageTk
                image_pil = Image.fromarray(image)
                self.image_preview = ImageTk.PhotoImage(image=image_pil)
                
                # Display in the preview label
                self.preview_label.config(image=self.image_preview)
                
            except Exception as e:
                messagebox.showerror("Preview Error", f"Could not preview image: {str(e)}")
                self.preview_label.config(image='')
        
        elif file_path.endswith('.mid'):
            try:
                # Generate a simple waveform or piano roll visualization
                from music21 import converter
                from PIL import Image, ImageDraw, ImageTk
                
                # Try to load MIDI and create a visualization
                try:
                    # Load the MIDI file
                    midi_data = converter.parse(file_path)
                    
                    # Create a blank image for the piano roll
                    width, height = 300, 200
                    image = Image.new("RGB", (width, height), color=(240, 240, 240))
                    draw = ImageDraw.Draw(image)
                    
                    # Extract notes and draw them as rectangles
                    notes = []
                    for part in midi_data.parts:
                        for note in part.flatten().notes:
                            offset = note.offset
                            duration = note.duration.quarterLength
                            pitch = note.pitch.midi
                            notes.append((offset, duration, pitch))
                    
                    # Normalize positions
                    if notes:
                        max_offset = max([n[0] + n[1] for n in notes])
                        min_pitch = min([n[2] for n in notes])
                        max_pitch = max([n[2] for n in notes])
                        pitch_range = max(1, max_pitch - min_pitch)
                        
                        # Draw notes
                        for offset, duration, pitch in notes:
                            x1 = int(offset * width / max_offset)
                            x2 = int((offset + duration) * width / max_offset)
                            y = height - int((pitch - min_pitch) * height / pitch_range)
                            
                            # Draw the note rectangle
                            draw.rectangle([x1, y - 2, x2, y + 2], 
                                        fill=(0, 100, 200), outline=(0, 0, 0))
                    
                    # Add grid lines
                    for i in range(0, width, 30):
                        draw.line([(i, 0), (i, height)], fill=(200, 200, 200))
                    
                    for i in range(0, height, 20):
                        draw.line([(0, i), (width, i)], fill=(200, 200, 200))
                    
                    # Display the visualization
                    self.image_preview = ImageTk.PhotoImage(image=image)
                    self.preview_label.config(image=self.image_preview, text="")
                    
                except Exception:
                    # If visualization fails, show placeholder
                    self.preview_label.config(image='', 
                                           text="MIDI File Selected\n(Click Preview to listen)")
            except ImportError:
                # Fallback if music21 is not available
                self.preview_label.config(image='', 
                                          text="MIDI File Selected\n(Click Preview to listen)")
    
    def get_resolution(self) -> Tuple[int, int]:
        """
        Get the current resolution settings
        
        :return: Current resolution as (rows, cols) tuple
        """
        return ImageAudioConverter.SPLIT_NUMBER
        
    def update_progress(self, value: float, message: str = None) -> None:
        """
        Update the progress bar and status message
        
        :param value: Progress value (0-100)
        :param message: Optional status message
        """
        self.progress_var.set(value)
        if message:
            self.status_label.config(text=message)
        self.mainwindow.update_idletasks()

    def convert(self) -> None:
        """
        Function that selects the appropriate transformation function based on file type
        """
        if self.file is None:
            messagebox.showerror("Error", "Please select a file first")
            return
            
        # Update resolution from spinboxes
        try:
            rows = self.builder.tkvariables['row_size'].get()
            cols = self.builder.tkvariables['col_size'].get()
            
            if rows <= 0 or cols <= 0:
                messagebox.showerror("Error", "Row and column sizes must be positive numbers")
                return
                
            ImageAudioConverter.SPLIT_NUMBER = (rows, cols)
            
        except Exception as e:
            messagebox.showerror("Error", f"Invalid resolution values: {str(e)}")
            return
        
        # Get selected musical key
        selected_key_name = self.selected_key.get()
        key = AVAILABLE_KEYS.get(selected_key_name, DEFAULT_KEY)
        
        # Create cypher with selected key
        cypher = musicgen.rules.TriadBaroqueCypher(key)
            
        # Configure processor based on performance settings
        if self.performance_mode.get() == "Speed":
            self.processor.chunk_size = CHUNK_SIZE * 2
            self.memory_optimizer.set_mode("speed")
        elif self.performance_mode.get() == "Quality":
            self.processor.chunk_size = CHUNK_SIZE // 2
            self.memory_optimizer.set_mode("quality")
        else:  # Balanced
            self.processor.chunk_size = CHUNK_SIZE
            self.memory_optimizer.set_mode("balanced")
            
        # Check if caching is enabled
        self.cache_manager.enabled = self.use_caching.get()
        
        # Start conversion in a separate thread to prevent UI freezing
        if self.file.name.endswith('.jpg') or self.file.name.endswith('.png'):
            threading.Thread(target=self._threaded_convert_img_to_music, 
                           args=(cypher,), daemon=True).start()
        else:
            threading.Thread(target=self._threaded_convert_music_to_img, 
                           args=(self.file.name, cypher), daemon=True).start()

    def _threaded_convert_img_to_music(self, cypher: musicgen.rules.Rules) -> None:
        """
        Threaded version of convert_img_to_music to keep the UI responsive
        
        :param cypher: The cypher used for conversion
        """
        try:
            self.update_progress(0, "Reading image...")
            
            # Check cache first if enabled
            cache_key = f"{self.file.name}_{ImageAudioConverter.SPLIT_NUMBER}"
            cached_data = None
            
            if self.cache_manager.enabled:
                cached_data = self.cache_manager.get_cached_data(cache_key)
                
            if cached_data:
                self.update_progress(30, "Using cached data...")
                notes, quarter_length, volume = cached_data
            else:
                # Load and optimize image
                self.update_progress(5, "Loading and optimizing image...")
                image = self.memory_optimizer.optimize_image_load(self.file.name)
                
                # Get the split data channels using parallel processing
                self.update_progress(10, "Extracting color channels...")
                notes, quarter_length, volume = self.split_image_transform(image)
                
                # Cache the results if enabled
                if self.cache_manager.enabled:
                    self.cache_manager.cache_data(cache_key, (notes, quarter_length, volume))
            
            self.update_progress(30, "Processing notes...")
            # Normalize the notes
            notes /= 255
            notes *= (len(self.available_notes) - 1)
            notes = np.rint(notes).astype(int)
            notes = np.vectorize(lambda x: self.available_notes[x])(notes)
            
            self.update_progress(40, "Processing timing...")
            # Normalize the quarter-length
            quarter_length /= 255
            quarter_length = quarter_length * (MAX_QUARTER_LENGTH - MIN_QUARTER_LENGTH) + MIN_QUARTER_LENGTH
            
            # Makes sure that the quarter-length are increments of 0.25
            quarter_length /= .25
            quarter_length = np.rint(quarter_length)
            quarter_length *= .25
            
            self.update_progress(50, "Processing volume...")
            # Normalize the volumes
            volume /= 255
            volume = volume * (MAX_VOLUME - MIN_VOLUME) + MIN_VOLUME
            
            self.update_progress(70, "Generating chord progression...")
            
            # Apply steganography if enabled
            note_data = list(zip(notes, quarter_length, volume))
            
            if self.steganography_enabled.get():
                self.update_progress(75, "Applying steganography...")
                
                # Get steganography method
                method_name = self.selected_steg_method.get()
                method_id = STEGANOGRAPHY_METHODS.get(method_name, DEFAULT_STEG_METHOD)
                
                # Configure additional parameters
                steg_params = {
                    "hide_throughout": self.hide_throughout.get(),
                    "key": cypher.key,
                    "method": method_id
                }
                
                # Apply steganography transformation
                note_data = steganography.apply_steganography(note_data, **steg_params)
                
                # Unzip the data back into separate lists
                notes, quarter_length, volume = zip(*note_data)
                
            # Convert the list of notes into a chord progression
            chords = musicgen.create_chords(
                [(note, vel, vol) for note, vel, vol in zip(notes, quarter_length, volume)],
                cypher
            )
            
            self.update_progress(90, "Saving MIDI file...")
            # Ask for save location
            file_types = [('Midi', '*.mid')]
            file = asksaveasfile(filetypes=file_types, defaultextension=file_types)
            
            if file is not None:
                chords.write("midi", file.name)
                self.last_midi_file = file.name
                
                # Enable preview button
                self.preview_button.config(state=tk.NORMAL)
                
                self.update_progress(100, f"Conversion complete! Saved to {os.path.basename(file.name)}")
                
                # Add metadata if steganography is enabled
                if self.steganography_enabled.get() and method_id == "metadata":
                    steganography.add_metadata_to_midi(file.name, {
                        "created_with": "SoundsAwful",
                        "original_image": os.path.basename(self.file.name),
                        "resolution": f"{ImageAudioConverter.SPLIT_NUMBER[0]}x{ImageAudioConverter.SPLIT_NUMBER[1]}",
                        "key": self.selected_key.get()
                    })
                
                self.open_file_in_system(file.name)
            else:
                self.update_progress(0, "Conversion cancelled")
                
        except Exception as e:
            self.update_progress(0, f"Error: {str(e)}")
            messagebox.showerror("Conversion Error", str(e))

    def _threaded_convert_music_to_img(self, music_in: str, cypher: musicgen.rules.Cypher) -> None:
        """
        Threaded version of convert_music_to_img to keep the UI responsive
        
        :param music_in: Path to input music file
        :param cypher: The cypher used for conversion
        """
        try:
            self.update_progress(0, "Reading MIDI file...")
            
            # Check for steganography data
            if self.steganography_enabled.get():
                self.update_progress(10, "Checking for hidden data...")
                
                # Get steganography method
                method_name = self.selected_steg_method.get()
                method_id = STEGANOGRAPHY_METHODS.get(method_name, DEFAULT_STEG_METHOD)
                
                # Try to extract metadata from the MIDI file
                if method_id == "metadata":
                    try:
                        metadata = steganography.extract_metadata_from_midi(music_in)
                        if metadata:
                            info = "\n".join([f"{k}: {v}" for k, v in metadata.items()])
                            messagebox.showinfo("Metadata Found", 
                                            f"Found hidden metadata in the MIDI file:\n\n{info}")
                    except Exception as e:
                        messagebox.showinfo("No Metadata", 
                                         f"No metadata found in the MIDI file: {str(e)}")
            
            # First, try direct music21 parsing
            self.update_progress(20, "Decoding MIDI data...")
            
            # Handle steganography extraction if enabled
            if self.steganography_enabled.get():
                method_name = self.selected_steg_method.get()
                method_id = STEGANOGRAPHY_METHODS.get(method_name, DEFAULT_STEG_METHOD)
                
                if method_id != "metadata":  # Metadata is handled separately
                    self.update_progress(30, f"Extracting hidden data using {method_name}...")
                    try:
                        note_identifiers = steganography.extract_from_midi(music_in, cypher, method_id)
                    except Exception as e:
                        self.update_progress(30, f"Steganography extraction failed: {str(e)}. Trying standard decode...")
                        # Fall back to standard decode if steganography extraction fails
                        note_identifiers = musicgen.decode(music_in, cypher)
                else:
                    note_identifiers = musicgen.decode(music_in, cypher)
            else:
                try:
                    note_identifiers = musicgen.decode(music_in, cypher)
                except Exception as e:
                    # If standard musicgen decode fails, try our robust direct MIDI parsing
                    self.update_progress(35, f"Standard decode failed: {str(e)}. Trying direct MIDI parsing...")
                    
                    try:
                        # Using built-in music21 to manually extract notes
                        from music21 import converter
                        
                        # Parse MIDI file
                        midi_data = converter.parse(music_in)
                        
                        # Extract notes with chord handling
                        notes = []
                        quarter_lengths = []
                        volumes = []
                        
                        for part in midi_data.parts:
                            for note_element in part.flatten().notes:
                                # Handle both individual notes and chords
                                if hasattr(note_element, 'isChord') and note_element.isChord:
                                    # Extract individual notes from chord
                                    for chord_note in note_element.notes:
                                        notes.append(chord_note.name[0])  # Just the letter (C, D, E, etc.)
                                        quarter_lengths.append(note_element.duration.quarterLength)
                                        volumes.append(chord_note.volume.velocity if chord_note.volume.velocity is not None else 80)
                                else:
                                    # Process individual note
                                    notes.append(note_element.name[0])  # Just the letter (C, D, E, etc.)
                                    quarter_lengths.append(note_element.duration.quarterLength)
                                    volumes.append(note_element.volume.velocity if note_element.volume.velocity is not None else 80)
                        
                        self.update_progress(60, "Reconstructing image...")
                        # Run the decode routine directly with extracted data
                        self.decode_music(np.asarray(notes), np.asarray(quarter_lengths), np.asarray(volumes))
                        return  # Exit after successful direct parsing
                    
                    except Exception as inner_e:
                        # If direct music21 parsing also fails, use binary parsing from simple_demo.py
                        self.update_progress(40, f"Music21 parsing failed: {str(inner_e)}. Using binary MIDI parsing...")
                        
                        # Try binary MIDI parsing (most robust method)
                        try:
                            # Use note names for the current key
                            key_name = self.selected_key.get()
                            current_key = AVAILABLE_KEYS.get(key_name, DEFAULT_KEY)
                            notes_list = list(map(lambda note: note.name[0], current_key.getPitches()))
                            
                            with open(music_in, 'rb') as f:
                                notes, quarter_lengths, volumes = self._parse_midi_binary(f, notes_list)
                                
                                if notes:
                                    self.update_progress(60, "Reconstructing image...")
                                    # Run the decode routine directly with extracted data
                                    self.decode_music(np.asarray(notes), np.asarray(quarter_lengths), np.asarray(volumes))
                                    return  # Exit after successful binary parsing
                        except Exception as binary_e:
                            self.update_progress(45, f"Binary parsing failed: {str(binary_e)}. Creating default image...")
                            
                            # Last resort: create a default image with the selected resolution
                            rows, cols = self.get_resolution()
                            total_blocks = rows * cols
                            
                            default_notes = ['C'] * total_blocks
                            default_quarter_lengths = [1.0] * total_blocks
                            default_volumes = [80] * total_blocks
                            
                            self.update_progress(60, "Creating default image (no notes could be extracted)...")
                            self.decode_music(np.asarray(default_notes), np.asarray(default_quarter_lengths), np.asarray(default_volumes))
                            return
                    
                    # If we reach here, something went wrong with direct parsing, raise the error
                    raise e
            
            notes = []
            quarter_lengths = []
            volumes = []
            
            self.update_progress(40, "Extracting note properties...")
            for note_identifier in note_identifiers:
                notes.append(note_identifier[0])
                quarter_lengths.append(note_identifier[1])
                volumes.append(note_identifier[2])
            
            self.update_progress(60, "Reconstructing image...")
            # Run the decode routine
            self.decode_music(np.asarray(notes), np.asarray(quarter_lengths), np.asarray(volumes))
            
        except Exception as e:
            self.update_progress(0, f"Error: {str(e)}")
            messagebox.showerror("Conversion Error", str(e))
    
    def _parse_midi_binary(self, file_handle, note_names_list):
        """
        Parse MIDI file directly using binary file operations
        This is more robust than music21 for some MIDI files
        
        :param file_handle: File handle to the MIDI file
        :param note_names_list: List of note names for the current key
        :return: Tuple of (notes, quarter_lengths, volumes)
        """
        self.update_status("Using binary MIDI parsing (robust method)...")
        
        # Note names corresponding to MIDI note numbers
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # For storing extracted data
        notes = []
        quarter_lengths = []
        volumes = []
        
        # Check if it's a valid MIDI file
        header = file_handle.read(4)
        if header != b'MThd':
            self.update_status("Warning: Not a standard MIDI file format")
            return [], [], []
        
        # Skip header length (always 6 bytes) and format type
        file_handle.seek(8)
        
        # Read number of tracks and time division
        tracks = int.from_bytes(file_handle.read(2), byteorder='big')
        time_division = int.from_bytes(file_handle.read(2), byteorder='big')
        
        self.update_status(f"MIDI file has {tracks} tracks, time division: {time_division}")
        
        # Process each track
        for track_idx in range(tracks):
            self.update_status(f"Processing track {track_idx+1}/{tracks}...")
            
            # Find the start of the track
            while True:
                chunk = file_handle.read(4)
                if not chunk:
                    break  # End of file
                if chunk == b'MTrk':
                    break
                # Skip other chunks
                length = int.from_bytes(file_handle.read(4), byteorder='big')
                file_handle.seek(length, 1)  # Relative seek
            
            if not chunk or chunk != b'MTrk':
                continue  # No track found or end of file
            
            # Read track length
            track_length = int.from_bytes(file_handle.read(4), byteorder='big')
            track_end = file_handle.tell() + track_length
            
            # Tempo default (500,000 microseconds per quarter note = 120 BPM)
            tempo = 500000
            
            # For note tracking
            active_notes = {}  # key: note, value: (start_time, velocity)
            current_time = 0
            
            # Read events
            while file_handle.tell() < track_end:
                # Read variable-length delta time
                delta = 0
                byte = file_handle.read(1)[0]
                delta = (delta << 7) | (byte & 0x7F)
                while byte & 0x80:
                    byte = file_handle.read(1)[0]
                    delta = (delta << 7) | (byte & 0x7F)
                
                current_time += delta
                
                # Read event type
                event_type = file_handle.read(1)[0]
                
                # Meta event
                if event_type == 0xFF:
                    meta_type = file_handle.read(1)[0]
                    length = int.from_bytes(file_handle.read(1), byteorder='big')
                    
                    # Tempo change
                    if meta_type == 0x51 and length == 3:
                        tempo_bytes = file_handle.read(3)
                        tempo = (tempo_bytes[0] << 16) | (tempo_bytes[1] << 8) | tempo_bytes[2]
                    else:
                        # Skip other meta events
                        file_handle.seek(length, 1)
                
                # System exclusive events
                elif event_type == 0xF0 or event_type == 0xF7:
                    length = int.from_bytes(file_handle.read(1), byteorder='big')
                    file_handle.seek(length, 1)
                
                # MIDI events
                else:
                    # Get status byte and channel
                    status = event_type & 0xF0
                    channel = event_type & 0x0F
                    
                    # Note-on event
                    if status == 0x90:
                        note = file_handle.read(1)[0]
                        velocity = file_handle.read(1)[0]
                        
                        # Some files use note-on with velocity 0 as note-off
                        if velocity > 0:
                            active_notes[note] = (current_time, velocity)
                    
                    # Note-off event
                    elif status == 0x80:
                        note = file_handle.read(1)[0]
                        velocity = file_handle.read(1)[0]  # Release velocity (usually ignored)
                        
                        # If we have a matching note-on, calculate duration
                        if note in active_notes:
                            start_time, start_velocity = active_notes.pop(note)
                            duration_ticks = current_time - start_time
                            
                            # Convert ticks to quarter notes
                            quarter_length = duration_ticks / time_division
                            
                            # Only consider notes longer than a 32nd note
                            if quarter_length >= 0.125:
                                # Get note name (simplified to just the letter)
                                note_letter = note_names[note % 12][0]
                                
                                # Map to current key's notes if possible
                                if note_letter in note_names_list:
                                    notes.append(note_letter)
                                    quarter_lengths.append(quarter_length)
                                    volumes.append(start_velocity)
                    
                    # Other MIDI events (Control Change, Program Change, etc)
                    else:
                        # Most events have 1 or 2 data bytes
                        if status in [0xC0, 0xD0]:  # Program Change, Channel Pressure
                            file_handle.read(1)
                        else:
                            file_handle.read(2)
        
        self.update_status(f"Binary MIDI parsing complete. Extracted {len(notes)} notes.")
        
        # If no notes were found, do a last resort binary scan
        if not notes:
            self.update_status("No notes found in formal parsing. Performing binary scan...")
            
            # Reset file pointer
            file_handle.seek(0)
            data = file_handle.read()
            
            # Scan for note-on events
            i = 0
            while i < len(data) - 3:
                # Look for note-on events (status byte 0x9x where x is the channel)
                if (data[i] & 0xF0) == 0x90 and data[i+2] > 0:
                    note = data[i+1]
                    velocity = data[i+2]
                    
                    # Get note name
                    note_letter = note_names[note % 12][0]
                    
                    notes.append(note_letter)
                    quarter_lengths.append(0.5)  # Assume quarter note
                    volumes.append(velocity)
                
                i += 1
            
            self.update_status(f"Binary scan found {len(notes)} potential notes.")
        
        # If we still found too many notes, limit to a reasonable number
        if len(notes) > 10000:
            self.update_status(f"Limiting {len(notes)} notes to 10000")
            notes = notes[:10000]
            quarter_lengths = quarter_lengths[:10000]
            volumes = volumes[:10000]
        
        return notes, quarter_lengths, volumes
    
    def update_status(self, message: str) -> None:
        """
        Update status display and log message
        
        :param message: Status message to display
        """
        if hasattr(self, 'status_label'):
            self.status_label.config(text=message)
            self.mainwindow.update_idletasks()
        print(message)


    def decode_music(self, notes: np.ndarray, quarter_lengths: np.ndarray, volumes: np.ndarray) -> None:
        """
        Main logic to convert the music to an image again
        
        :param notes: the string representation of the notes
        :param quarter_lengths: the duration of each note
        :param volumes: the volume of each note
        """
        try:
            # Get available notes for the current key
            key_name = self.selected_key.get()
            current_key = AVAILABLE_KEYS.get(key_name, DEFAULT_KEY)
            notes_list = list(map(lambda note: note.name[0], current_key.getPitches()))
            
            # Convert note names to indices first
            self.update_progress(70, "Converting note data...")
            
            # Handle notes correctly - map note names to indices
            note_indices = []
            for note in notes:
                try:
                    # Try to find the note in the list
                    idx = notes_list.index(note[0])
                    note_indices.append(idx)
                except (ValueError, IndexError):
                    # If the note is not in the list, use the first note
                    note_indices.append(0)
            
            note_indices = np.array(note_indices)
            
            # Properly denormalize the data back to 0-255 range
            self.update_progress(75, "Denormalizing to image values...")
            
            # Notes: Scale from note index (0-6 for example) to 0-255
            note_channel = (note_indices / (len(notes_list) - 1)) * 255
            
            # Quarter lengths: From actual durations back to 0-255
            qlen_channel = ((quarter_lengths - MIN_QUARTER_LENGTH) / 
                         (MAX_QUARTER_LENGTH - MIN_QUARTER_LENGTH)) * 255
            
            # Volumes: From MIDI velocity (20-127) back to 0-255
            vol_channel = ((volumes - MIN_VOLUME) / (MAX_VOLUME - MIN_VOLUME)) * 255
            
            # Clip values to valid range
            note_channel = np.clip(note_channel, 0, 255)
            qlen_channel = np.clip(qlen_channel, 0, 255)
            vol_channel = np.clip(vol_channel, 0, 255)
            
            # Check if we need to adjust the resolution
            total_elements = len(notes)
            
            if total_elements != ImageAudioConverter.SPLIT_NUMBER[0] * ImageAudioConverter.SPLIT_NUMBER[1]:
                # Import the enhanced resolution detection from midi_to_image_demo
                from midi_to_image_demo import detect_best_resolution
                
                # Try to detect the appropriate resolution using enhanced algorithm
                self.update_progress(80, "Detecting optimal resolution...")
                
                # Get the best resolution using our enhanced algorithm
                detected_resolution = detect_best_resolution(total_elements)
                
                # Find nearest standard resolution with good aspect ratio
                standard_resolutions = [(64, 64), (128, 128), (32, 32), (120, 90), (160, 90), (80, 60)]
                nearest_standard = min(standard_resolutions, 
                                     key=lambda res: abs(res[0] * res[1] - total_elements))
                
                # Create message with options for our custom dialog
                message = (f"The MIDI file contains {total_elements} notes which doesn't match the selected resolution "
                         f"of {ImageAudioConverter.SPLIT_NUMBER[0]}x{ImageAudioConverter.SPLIT_NUMBER[1]} "
                         f"({ImageAudioConverter.SPLIT_NUMBER[0] * ImageAudioConverter.SPLIT_NUMBER[1]} notes).\n\n"
                         f"Please choose a resolution option:\n"
                         f"1. Use detected resolution: {detected_resolution[0]}x{detected_resolution[1]} (exact fit)\n"
                         f"2. Use nearest standard resolution: {nearest_standard[0]}x{nearest_standard[1]} (may require resizing)\n"
                         f"3. Use selected resolution: {ImageAudioConverter.SPLIT_NUMBER[0]}x{ImageAudioConverter.SPLIT_NUMBER[1]} (will resize)")
                
                # To avoid the tk variable scope error, we'll create a thread-safe dialog approach
                # Create a synchronization primitive
                choice = ["1"]  # Default choice in a mutable container
                dialog_active = [True]
                
                # Function to create and manage dialog in the main thread
                def create_resolution_dialog():
                    nonlocal choice
                    # Import tkinter locally with different names to avoid scope conflicts
                    import tkinter as local_tk
                    from tkinter import ttk as local_ttk
                    
                    # Create dialog in main thread using local imports
                    resolution_dialog = local_tk.Toplevel(self.mainwindow)
                    resolution_dialog.title("Resolution Options")
                    resolution_dialog.geometry("500x300")
                    resolution_dialog.transient(self.mainwindow)
                    resolution_dialog.grab_set()
                    
                    # Add message text
                    message_label = local_ttk.Label(resolution_dialog, text=message, wraplength=480, justify="left")
                    message_label.pack(padx=20, pady=20)
                    
                    # Create radio buttons for choices
                    choice_var = local_tk.StringVar(value="1")
                    local_ttk.Radiobutton(resolution_dialog, text="Option 1: Use detected resolution", 
                                  variable=choice_var, value="1").pack(anchor="w", padx=20)
                    local_ttk.Radiobutton(resolution_dialog, text="Option 2: Use standard resolution", 
                                  variable=choice_var, value="2").pack(anchor="w", padx=20)
                    local_ttk.Radiobutton(resolution_dialog, text="Option 3: Use selected resolution", 
                                  variable=choice_var, value="3").pack(anchor="w", padx=20)
                    
                    # Add OK button
                    def on_ok():
                        choice[0] = choice_var.get()
                        dialog_active[0] = False
                        resolution_dialog.destroy()
                    
                    local_ttk.Button(resolution_dialog, text="OK", command=on_ok).pack(pady=20)
                    
                    # Handle window close via X button
                    def on_close():
                        dialog_active[0] = False
                        resolution_dialog.destroy()
                    
                    resolution_dialog.protocol("WM_DELETE_WINDOW", on_close)
                
                # Schedule dialog creation on main thread
                self.mainwindow.after(0, create_resolution_dialog)
                
                # Wait for dialog to complete
                while dialog_active[0]:
                    time.sleep(0.1)
                
                # Get the choice
                user_choice = choice[0]
                
                if user_choice == "1":
                    # Use detected resolution
                    best_dim = detected_resolution
                    self.update_progress(85, f"Using detected resolution: {best_dim[0]}x{best_dim[1]}...")
                    
                    # Update the resolution
                    ImageAudioConverter.SPLIT_NUMBER = best_dim
                    shape = (best_dim[0], best_dim[1], 1)
                    
                    # Update UI to reflect the new resolution
                    self.builder.tkvariables['row_size'].set(best_dim[0])
                    self.builder.tkvariables['col_size'].set(best_dim[1])
                    
                elif user_choice == "2":
                    # Use nearest standard resolution
                    best_dim = nearest_standard
                    self.update_progress(85, f"Using standard resolution: {best_dim[0]}x{best_dim[1]}...")
                    
                    # Update the resolution
                    ImageAudioConverter.SPLIT_NUMBER = best_dim
                    shape = (best_dim[0], best_dim[1], 1)
                    
                    # Update UI to reflect the new resolution
                    self.builder.tkvariables['row_size'].set(best_dim[0])
                    self.builder.tkvariables['col_size'].set(best_dim[1])
                    
                else:
                    # Use selected resolution (or default to this if they cancel)
                    self.update_progress(85, f"Using selected resolution: {ImageAudioConverter.SPLIT_NUMBER[0]}x{ImageAudioConverter.SPLIT_NUMBER[1]}...")
                    
                    # Use the closest viable size by truncating or padding the data
                    target_size = ImageAudioConverter.SPLIT_NUMBER[0] * ImageAudioConverter.SPLIT_NUMBER[1]
                    
                    if total_elements > target_size:
                        # Truncate excess data
                        note_channel = note_channel[:target_size]
                        qlen_channel = qlen_channel[:target_size]
                        vol_channel = vol_channel[:target_size]
                        self.update_progress(87, f"Truncating excess data to fit {ImageAudioConverter.SPLIT_NUMBER[0]}x{ImageAudioConverter.SPLIT_NUMBER[1]}...")
                    else:
                        # Pad with zeros
                        pad_size = target_size - total_elements
                        note_channel = np.pad(note_channel, (0, pad_size), 'constant', constant_values=0)
                        qlen_channel = np.pad(qlen_channel, (0, pad_size), 'constant', constant_values=0)
                        vol_channel = np.pad(vol_channel, (0, pad_size), 'constant', constant_values=0)
                        self.update_progress(87, f"Padding data to fit {ImageAudioConverter.SPLIT_NUMBER[0]}x{ImageAudioConverter.SPLIT_NUMBER[1]}...")
                    
                    shape = (ImageAudioConverter.SPLIT_NUMBER[0], ImageAudioConverter.SPLIT_NUMBER[1], 1)
            else:
                # The current resolution is correct
                shape = (ImageAudioConverter.SPLIT_NUMBER[0], ImageAudioConverter.SPLIT_NUMBER[1], 1)
            
            # Apply additional correction to ensure proper image reconstruction
            # Important: OpenCV uses BGR format (not RGB)
            
            # Convert to proper format for OpenCV BGR image
            # Ensure each channel is in proper 0-255 range
            blue_channel = np.round(np.clip(note_channel, 0, 255)).astype(np.uint8).reshape(shape)
            green_channel = np.round(np.clip(qlen_channel, 0, 255)).astype(np.uint8).reshape(shape)
            red_channel = np.round(np.clip(vol_channel, 0, 255)).astype(np.uint8).reshape(shape)
            
            # Create BGR (OpenCV) format image
            # Original encoding maps: Blue channel = notes, Green channel = durations, Red channel = volumes
            self.update_progress(88, "Creating BGR image (OpenCV format)...")
            
            # Sequence matters here - BGR order for OpenCV
            new_image = cv2.merge([blue_channel, green_channel, red_channel])
            
            # Apply additional image enhancement if needed
            # Try different color mapping if user requests
            answer = messagebox.askyesno(
                "Image Reconstruction",
                "Would you like to try alternate color mappings? This can sometimes improve reconstruction quality."
            )
            
            if answer:
                # Try alternate color mappings
                self.update_progress(89, "Trying alternate color mappings...")
                
                # Create alternate mappings
                images = [
                    # BGR format
                    new_image,
                    # BGR ↔ RGB swap (in case original was RGB format)
                    cv2.merge([red_channel, green_channel, blue_channel]),
                    # Try permutations
                    cv2.merge([blue_channel, red_channel, green_channel]),
                    cv2.merge([green_channel, blue_channel, red_channel]),
                    cv2.merge([green_channel, red_channel, blue_channel]),
                    cv2.merge([red_channel, blue_channel, green_channel])
                ]
                
                # Show image options for user to choose
                from PIL import Image, ImageTk
                import tkinter as tk
                
                selection_window = tk.Toplevel(self.mainwindow)
                selection_window.title("Select Best Image")
                
                # Selected image index
                selected_idx = [-1]  # Use list to make it mutable
                
                # Create frame for images
                frame = ttk.Frame(selection_window)
                frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                
                # Create buttons for each image
                buttons = []
                for i, img in enumerate(images):
                    # Convert to RGB for display
                    display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Scale for display
                    height, width = display_img.shape[:2]
                    max_size = 150
                    scale = min(max_size / width, max_size / height)
                    display_img = cv2.resize(display_img, (int(width * scale), int(height * scale)))
                    
                    # Add to frame
                    pil_img = Image.fromarray(display_img)
                    tk_img = ImageTk.PhotoImage(image=pil_img)
                    
                    # Create frame for this option
                    opt_frame = ttk.Frame(frame)
                    opt_frame.grid(row=i//3, column=i%3, padx=5, pady=5)
                    
                    # Create label and button
                    label = ttk.Label(opt_frame, image=tk_img)
                    label.image = tk_img  # Keep reference
                    label.pack()
                    
                    # Selection function
                    def select_image(idx=i):
                        selected_idx[0] = idx
                        selection_window.destroy()
                    
                    button = ttk.Button(opt_frame, text=f"Option {i+1}", command=lambda idx=i: select_image(idx))
                    button.pack(pady=5)
                    
                    buttons.append(button)
                
                # Wait for user selection
                self.mainwindow.wait_window(selection_window)
                
                # Use selected image if user made a choice
                if selected_idx[0] >= 0:
                    new_image = images[selected_idx[0]]
                    self.update_progress(92, f"Using selected color mapping (Option {selected_idx[0]+1})...")
            
            self.update_progress(90, "Saving image...")
            # Ask user where to save file
            file_types = [('PNG', "*.png"), ('JPEG', '*.jpeg')]
            file = asksaveasfile(filetypes=file_types, defaultextension=file_types)
            
            if file is not None:
                cv2.imwrite(file.name, new_image)
                self.update_progress(100, f"Conversion complete! Saved to {os.path.basename(file.name)}")
                
                # Show preview of the reconstructed image
                reconstructed_img = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
                height, width = reconstructed_img.shape[:2]
                max_size = 300
                scale = min(max_size / width, max_size / height)
                reconstructed_img = cv2.resize(reconstructed_img, 
                                            (int(width * scale), int(height * scale)))
                
                from PIL import Image, ImageTk
                pil_img = Image.fromarray(reconstructed_img)
                self.image_preview = ImageTk.PhotoImage(image=pil_img)
                self.preview_label.config(image=self.image_preview)
                
                # Open the file in the system viewer
                self.open_file_in_system(file.name)
            else:
                self.update_progress(0, "Conversion cancelled")
                
        except Exception as e:
            self.update_progress(0, f"Error: {str(e)}")
            messagebox.showerror("Conversion Error", str(e))

    def split_image_transform(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits the image into blocks and then averages each channel to get wanted data
        :param image: the input image to retrieve the features from
        :return: the three raw extracted features: the notes, the volume and the quarter_length
        """
        # Use the optimized image size
        d_height = image.shape[0] / ImageAudioConverter.SPLIT_NUMBER[0]
        d_width = image.shape[1] / ImageAudioConverter.SPLIT_NUMBER[1]

        ratio_h = np.floor(d_height) * ImageAudioConverter.SPLIT_NUMBER[0]
        ratio_w = np.floor(d_width) * ImageAudioConverter.SPLIT_NUMBER[1]

        # Resizes the image if it is too large and causes issues in the splitting
        image = cv2.resize(image, (int(ratio_w), int(ratio_h)))

        d_height = int(image.shape[0] / ImageAudioConverter.SPLIT_NUMBER[0])
        d_width = int(image.shape[1] / ImageAudioConverter.SPLIT_NUMBER[1])

        # Determine block indices
        r_indices = list(range(0, image.shape[0], d_height))
        c_indices = list(range(0, image.shape[1], d_width))
        
        # Prepare input for parallel processing
        blocks = []
        for r in r_indices:
            for c in c_indices:
                blocks.append((image, r, c, d_height, d_width))
        
        # Parallel processing function with enhanced data extraction
        def process_block(params):
            img, r, c, dh, dw = params
            crop = img[r:r + dh, c:c + dw]
            
            # Calculate multiple statistics for better representation
            # Using mean instead of median captures more information
            blue_mean = np.mean(crop[:, :, 0])
            green_mean = np.mean(crop[:, :, 1])
            red_mean = np.mean(crop[:, :, 2])
            
            # Also capture min/max values to encode additional information
            # These will be embedded in subtle variations of timing and volume
            blue_min = np.min(crop[:, :, 0])
            blue_max = np.max(crop[:, :, 0])
            green_min = np.min(crop[:, :, 1])
            green_max = np.max(crop[:, :, 1])
            red_min = np.min(crop[:, :, 2])
            red_max = np.max(crop[:, :, 2])
            
            # Pack additional data into the fractional parts of the values
            # This allows us to embed more information without significantly
            # affecting the musical quality
            blue_enhanced = blue_mean
            green_enhanced = green_mean + ((blue_max - blue_min) / 2550)  # Small increment (0.01-0.1)
            red_enhanced = red_mean + ((green_max - green_min) / 2550)    # Small increment (0.01-0.1)
            
            return (
                blue_enhanced,  # Primary note value
                green_enhanced, # Duration with embedded blue range
                red_enhanced    # Volume with embedded green range
            )
        
        # Process blocks in parallel
        total_blocks = len(blocks)
        
        # Process in chunks to update progress
        note = []
        quarter_length = []
        volume = []
        
        chunk_size = self.processor.chunk_size
        for i in range(0, total_blocks, chunk_size):
            chunk = blocks[i:i+chunk_size]
            
            # Process this chunk in parallel
            results = self.processor.process_parallel(process_block, chunk)
            
            # Extract results
            for n, q, v in results:
                note.append(n)
                quarter_length.append(q)
                volume.append(v)
            
            # Update progress
            progress = min(100, (i + chunk_size) / total_blocks * 100)
            self.update_progress(10 + (progress / 5), f"Processing image blocks: {i+len(results)}/{total_blocks}")

        return np.array(note), np.array(quarter_length), np.array(volume)

        
    def open_file_in_system(self, file: str) -> None:
        """
        Function to open a file using the system's default application
        :param file: the file to open
        """
        try:
            if self.platform == 'Darwin':  # macOS
                subprocess.call(('open', file))
            elif self.platform == 'Windows':  # Windows
                os.startfile(file)
            else:  # linux variants
                subprocess.call(('xdg-open', file))
        except Exception as e:
            messagebox.showerror("Error", f"Could not open file: {str(e)}")

        
    def show_help(self) -> None:
        """
        Display help information about using the application
        """
        help_text = """
        SoundsAwful - Image-Music Cryptography
        
        Instructions:
        
        1. Click "Select" to choose an image (.jpg, .png) or MIDI file (.mid)
        2. Set the desired resolution (higher = more detail but longer music)
        3. Choose a musical key from the dropdown
        4. Optional: Enable steganography options
        5. Click "Convert" to process the file
        6. Choose a location to save the output file
        
        For image to music conversion:
        - The blue channel becomes musical notes
        - The green channel becomes note duration
        - The red channel becomes note volume
        
        For music to image conversion:
        - The process is reversed to reconstruct the original image
        
        Performance options:
        - Speed mode: Faster processing but may use more memory
        - Balanced mode: Good balance between speed and quality
        - Quality mode: Best quality but slower processing
        - Thread count: More threads = faster processing on multi-core systems
        
        Steganography options:
        - Bass Note: Hides data in the bass notes of chords
        - Note Duration: Hides data in note duration patterns
        - Note Volume: Hides data in volume variations
        - Metadata: Embeds hidden information in MIDI metadata
        - Pattern-based: Uses musical patterns to encode additional data
        - Key selection: Changes the musical style of the output
        
        For more information, visit the GitHub repository.
        """
        
        help_window = tk.Toplevel(self.mainwindow)
        help_window.title("Help")
        help_window.geometry("500x400")
        
        text_widget = tk.Text(help_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert(tk.END, help_text)
        text_widget.config(state=tk.DISABLED)
        
        close_button = ttk.Button(help_window, text="Close", command=help_window.destroy)
        close_button.pack(pady=10)


# SPLIT_NUMBER is a class variable because it's referenced in instance methods
ImageAudioConverter.SPLIT_NUMBER = DEFAULT_RESOLUTION

if __name__ == '__main__':
    # Get available notes for the default key
    NOTES_LIST = list(map(lambda note: note.name[0], DEFAULT_KEY.getPitches()))
    
    # Create and run the app
    app = ImageAudioConverter(NOTES_LIST)
    app.run()
