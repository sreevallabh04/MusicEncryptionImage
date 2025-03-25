"""
Steganography Module for SoundsAwful

This module provides functionality to hide generated chords throughout a piece of music,
enabling more subtle data hiding for steganographic purposes.
"""

import json
import os
import random
from typing import List, Tuple, Dict, Optional, Any, Union

import numpy as np
from music21 import chord, key, note, stream, tempo, converter, metadata

# Main steganography functions required by ui_improved.py
def apply_steganography(
    note_data: List[Tuple[str, float, float]],
    method: str = "bass_note",
    hide_throughout: bool = False,
    key: Any = None,
    **kwargs
) -> List[Tuple[str, float, float]]:
    """
    Apply steganography transformation to note data
    
    :param note_data: List of (note, quarter_length, volume) tuples
    :param method: Steganography method to use
    :param hide_throughout: Whether to hide data throughout music
    :param key: Musical key for encoding
    :param kwargs: Additional method-specific parameters
    :return: Transformed note data
    """
    # Create a carrier stream for hiding if requested
    if hide_throughout and len(note_data) > 10:
        # Generate carrier music in the same key
        carrier_length = len(note_data) * 2  # Make carrier longer than data
        carrier_stream = _generate_carrier_music(carrier_length, key=key)
        
        # Convert note data to a stream
        data_stream = stream.Stream()
        for note_name, quarter_length, volume in note_data:
            n = note.Note(note_name)
            n.quarterLength = quarter_length
            n.volume.velocity = volume
            data_stream.append(n)
        
        # Hide data stream in carrier
        combined_stream = hide_chords_in_music(
            data_stream, None, hide_factor=0.5)
        
        # Extract notes back to tuples
        result = []
        for element in combined_stream.recurse().getElementsByClass(['Note', 'Chord']):
            if isinstance(element, chord.Chord):
                # Extract bass note
                bass = element.bass()
                result.append((bass.name, element.quarterLength, bass.volume.velocity))
            elif isinstance(element, note.Note):
                result.append((element.name, element.quarterLength, element.volume.velocity))
        
        return result
    
    # Apply method-specific transformations
    if method == "bass_note":
        # Subtle alterations to note names while preserving musical structure
        result = []
        pattern = kwargs.get('pattern', [0, 2, 5, 7])  # Default pattern for identification
        
        for i, (note_name, quarter_length, volume) in enumerate(note_data):
            if i % 4 == 0:  # Mark pattern start points
                # Add pattern identifier note
                pattern_idx = (i // 4) % len(pattern)
                note_val = _note_name_to_value(note_name)
                note_val = (note_val + pattern[pattern_idx]) % 12
                new_note = _value_to_note_name(note_val)
                result.append((new_note, quarter_length, volume))
            else:
                result.append((note_name, quarter_length, volume))
                
        return result
        
    elif method == "duration":
        # Hide data in subtle duration variations
        result = []
        for i, (note_name, quarter_length, volume) in enumerate(note_data):
            if i % 5 == 0:  # Every 5th note gets modified
                # Add small random variation to duration
                variation = 0.125  # 1/8th note
                new_duration = quarter_length + variation
                result.append((note_name, new_duration, volume))
            else:
                result.append((note_name, quarter_length, volume))
                
        return result
        
    elif method == "volume":
        # Hide data in volume patterns
        result = []
        for i, (note_name, quarter_length, volume) in enumerate(note_data):
            if i % 3 == 0:  # Every 3rd note gets modified
                # Adjust volume slightly
                modified_volume = min(127, volume + 10)
                result.append((note_name, quarter_length, modified_volume))
            else:
                result.append((note_name, quarter_length, volume))
                
        return result
        
    elif method == "metadata":
        # No transformation needed - metadata is added separately
        return note_data
        
    elif method == "pattern":
        # Use specific note patterns as markers
        sequence = [2, 5, 7, 12]  # Example pattern
        result = []
        
        for i, (note_name, quarter_length, volume) in enumerate(note_data):
            if i % (len(sequence) + 1) == 0 and i + len(sequence) < len(note_data):
                # Insert pattern sequence
                result.append((note_name, quarter_length, volume))
                for j, shift in enumerate(sequence):
                    orig_note, orig_qlen, orig_vol = note_data[i + j + 1]
                    note_val = _note_name_to_value(orig_note)
                    note_val = (note_val + shift) % 12
                    new_note = _value_to_note_name(note_val)
                    result.append((new_note, orig_qlen, orig_vol))
                
                # Skip the notes we used for the pattern
                i += len(sequence)
            else:
                result.append((note_name, quarter_length, volume))
                
        return result
    
    # Default - return unmodified
    return note_data

def extract_from_midi(
    midi_path: str,
    cypher: Any,
    method: str = "bass_note"
) -> List[Tuple[str, float, float]]:
    """
    Extract hidden data from a MIDI file
    
    :param midi_path: Path to the MIDI file
    :param cypher: Cypher used for decoding
    :param method: Steganography method used for hiding
    :return: Extracted note data
    """
    # First parse the MIDI file
    midi_stream = converter.parse(midi_path)
    
    # Use the appropriate extraction method
    if method == "bass_note":
        # Extract the stream using the bass note extraction
        extracted_stream = extract_chords_from_steganography(
            midi_stream, pattern=None, extraction_algorithm="bass_note")
            
    elif method == "duration":
        # Extract based on duration patterns
        extracted_stream = extract_chords_from_steganography(
            midi_stream, pattern=None, extraction_algorithm="duration")
            
    elif method == "volume":
        # Extract based on volume patterns
        extracted_stream = extract_chords_from_steganography(
            midi_stream, pattern=None, extraction_algorithm="volume")
            
    elif method == "pattern":
        # Extract based on specific patterns
        pattern = [2, 5, 7, 12]  # Must match the pattern used for hiding
        extracted_stream = extract_chords_from_steganography(
            midi_stream, pattern=pattern, extraction_algorithm="bass_note")
            
    else:
        # Default extraction or unknown method
        # Use the decode method from the cypher
        return cypher.decode(midi_path)
    
    # Convert extracted stream to note data tuples
    result = []
    for element in extracted_stream.recurse().getElementsByClass(['Note', 'Chord']):
        if isinstance(element, chord.Chord):
            # Get the bass note
            bass = element.bass()
            result.append((bass.name, element.quarterLength, bass.volume.velocity))
        elif isinstance(element, note.Note):
            result.append((element.name, element.quarterLength, element.volume.velocity))
    
    return result

def add_metadata_to_midi(midi_path: str, metadata_dict: Dict[str, str]) -> bool:
    """
    Add metadata to an existing MIDI file
    
    :param midi_path: Path to the MIDI file
    :param metadata_dict: Dictionary of metadata to add
    :return: True if successful, False otherwise
    """
    try:
        # Load the MIDI file
        midi_stream = converter.parse(midi_path)
        
        # Add metadata
        md = metadata.Metadata()
        for key, value in metadata_dict.items():
            # Handle different metadata fields
            if key.lower() == "title":
                md.title = value
            elif key.lower() == "composer":
                md.composer = value
            else:
                # For custom fields, use the contributor field with role
                md.addContributor(value, role=key)
        
        # Add metadata to the stream
        midi_stream.metadata = md
        
        # Also add hidden metadata using our embedding method
        midi_stream = embed_metadata(midi_stream, metadata_dict)
        
        # Write back to the same file
        midi_stream.write("midi", midi_path)
        return True
        
    except Exception as e:
        print(f"Error adding metadata: {str(e)}")
        return False

def extract_metadata_from_midi(midi_path: str) -> Dict[str, str]:
    """
    Extract metadata from a MIDI file
    
    :param midi_path: Path to the MIDI file
    :return: Dictionary of extracted metadata
    """
    try:
        # Load the MIDI file
        midi_stream = converter.parse(midi_path)
        
        # First try to extract using our hidden method
        hidden_metadata = extract_metadata(midi_stream)
        if hidden_metadata:
            return hidden_metadata
        
        # Fall back to standard metadata
        md = midi_stream.metadata
        if md is None:
            return {}
            
        result = {}
        
        # Extract standard fields
        if md.title:
            result["title"] = md.title
        if md.composer:
            result["composer"] = md.composer
            
        # Extract contributors
        for contributor in md.contributors:
            role = contributor.role if contributor.role else "contributor"
            result[role] = contributor.name
            
        return result
        
    except Exception as e:
        print(f"Error extracting metadata: {str(e)}")
        return {}

# Helper functions for note conversion
def _note_name_to_value(note_name: str) -> int:
    """Convert note name to numeric value (0-11)"""
    base_notes = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    
    # Handle sharps and flats
    accidental = 0
    if len(note_name) > 1:
        if note_name[1] == '#':
            accidental = 1
        elif note_name[1] == '-' or note_name[1] == 'b':
            accidental = -1
    
    return (base_notes[note_name[0]] + accidental) % 12

def _value_to_note_name(value: int) -> str:
    """Convert numeric value (0-11) to note name"""
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return note_names[value % 12]

def hide_chords_in_music(
    data_chords: stream.Stream, 
    carrier_music_path: Optional[str] = None,
    hide_factor: float = 0.25
) -> stream.Stream:
    """
    Hide data chords within a carrier piece of music.
    If no carrier music is provided, a new piece will be generated.
    
    :param data_chords: Stream of chords containing the hidden data
    :param carrier_music_path: Optional path to a MIDI file to use as carrier
    :param hide_factor: Ratio of data chords to carrier music (0.1-0.5 recommended)
    :return: A stream containing the hidden data
    """
    # If no carrier provided, generate one
    if carrier_music_path is None:
        carrier_stream = _generate_carrier_music(data_chords.quarterLength * 4)
    else:
        # Parse the provided MIDI file
        from music21 import converter
        carrier_stream = converter.parse(carrier_music_path)
    
    # Create the output stream
    output_stream = stream.Stream()
    
    # Add tempo marking
    output_stream.append(tempo.MetronomeMark(number=80))
    
    # Get data chords as a list
    data_chord_list = list(data_chords.getElementsByClass('Chord'))
    data_index = 0
    data_chord_count = len(data_chord_list)
    
    # Hide data chords throughout the carrier
    for element in carrier_stream.recurse():
        # Only consider notes or chords in the carrier
        if not isinstance(element, (note.Note, chord.Chord)):
            continue
            
        # Randomly decide to replace this carrier element with a data chord
        if data_index < data_chord_count and random.random() < hide_factor:
            # Insert the data chord
            data_chord = data_chord_list[data_index]
            output_stream.append(data_chord)
            data_index += 1
        else:
            # Keep the carrier element
            output_stream.append(element)
    
    # If we still have data chords left, append them to the end
    while data_index < data_chord_count:
        output_stream.append(data_chord_list[data_index])
        data_index += 1
    
    return output_stream

def extract_chords_from_steganography(
    stego_stream: stream.Stream,
    pattern: Optional[List[int]] = None,
    extraction_algorithm: str = 'bass_note'
) -> stream.Stream:
    """
    Extract hidden data chords from a steganographic music piece.
    
    :param stego_stream: Stream containing hidden data
    :param pattern: Optional pattern sequence to identify hidden chords
    :param extraction_algorithm: Algorithm to use for extraction ('bass_note', 'volume', 'duration')
    :return: Stream containing only the extracted data chords
    """
    output_stream = stream.Stream()
    
    # Select extraction algorithm
    if extraction_algorithm == 'bass_note':
        # Extract based on bass note patterns
        all_chords = list(stego_stream.recurse().getElementsByClass('Chord'))
        
        # Filter chords that match our pattern
        if pattern:
            # Apply pattern-based extraction
            pattern_length = len(pattern)
            for i, chd in enumerate(all_chords):
                if i % pattern_length == 0 and i + pattern_length <= len(all_chords):
                    # Check if this sequence matches our pattern
                    seq_match = True
                    for j in range(pattern_length):
                        # Check if bass note matches pattern
                        bass_pitch = all_chords[i+j].bass().pitch.midi % 12
                        if bass_pitch != pattern[j]:
                            seq_match = False
                            break
                    
                    if seq_match:
                        # Add the chord that follows the pattern
                        if i + pattern_length < len(all_chords):
                            output_stream.append(all_chords[i + pattern_length])
        else:
            # No pattern - attempt statistical analysis
            # Analyze volume and duration patterns
            volumes = [chord.volume.velocity for chord in all_chords if hasattr(chord, 'volume')]
            durations = [chord.quarterLength for chord in all_chords]
            
            # Find statistical outliers
            if volumes:
                vol_mean = np.mean(volumes)
                vol_std = np.std(volumes)
                
                for chd in all_chords:
                    if hasattr(chd, 'volume'):
                        # If the volume is significantly different than the average
                        if abs(chd.volume.velocity - vol_mean) > vol_std * 1.5:
                            output_stream.append(chd)
                
    elif extraction_algorithm == 'volume':
        # Extract based on volume characteristics
        all_elements = list(stego_stream.recurse().getElementsByClass(['Note', 'Chord']))
        
        # Calculate volume statistics
        volumes = [elem.volume.velocity for elem in all_elements if hasattr(elem, 'volume')]
        if volumes:
            vol_mean = np.mean(volumes)
            vol_std = np.std(volumes)
            
            # Extract elements with unusual volumes
            for elem in all_elements:
                if hasattr(elem, 'volume'):
                    if abs(elem.volume.velocity - vol_mean) > vol_std:
                        output_stream.append(elem)
    
    elif extraction_algorithm == 'duration':
        # Extract based on duration characteristics
        all_elements = list(stego_stream.recurse().getElementsByClass(['Note', 'Chord']))
        
        # Calculate duration statistics
        durations = [elem.quarterLength for elem in all_elements]
        dur_mean = np.mean(durations)
        dur_std = np.std(durations)
        
        # Extract elements with unusual durations
        for elem in all_elements:
            if abs(elem.quarterLength - dur_mean) > dur_std:
                output_stream.append(elem)
    
    return output_stream

def _generate_carrier_music(
    desired_length: float, 
    style: str = 'classical',
    carrier_key: key.Key = key.Key('C')
) -> stream.Stream:
    """
    Generate a simple carrier piece in the specified style and key.
    
    :param desired_length: The desired length in quarter notes
    :param style: Musical style to generate ('classical', 'jazz', 'pop')
    :param carrier_key: Key to generate in
    :return: A stream of the generated carrier music
    """
    carrier = stream.Stream()
    
    # Add tempo
    carrier.append(tempo.MetronomeMark(number=80))
    
    # Style parameters
    if style == 'classical':
        # Simple classical-like progression: I-IV-V-I
        patterns = [
            [0, 4, 7],      # I (major triad)
            [5, 9, 12],     # IV (major triad)
            [7, 11, 14],    # V (major triad)
            [0, 4, 7]       # I (major triad)
        ]
        durations = [1.0, 1.0, 1.0, 1.0]
    elif style == 'jazz':
        # Jazz-like progression with 7ths: IImaj7-V7-Imaj7
        patterns = [
            [2, 6, 9, 12],   # IImaj7
            [7, 11, 14, 17], # V7 (dominant)
            [0, 4, 7, 11]    # Imaj7
        ]
        durations = [1.5, 1.5, 2.0]
    else:  # pop
        # Pop-like progression: I-V-vi-IV
        patterns = [
            [0, 4, 7],       # I (major triad)
            [7, 11, 14],     # V (major triad)
            [9, 12, 16],     # vi (minor triad)
            [5, 9, 12]       # IV (major triad)
        ]
        durations = [1.0, 1.0, 1.0, 1.0]
    
    # Generate enough pattern repetitions to reach desired length
    current_length = 0
    root_note = carrier_key.tonic.midi
    
    while current_length < desired_length:
        for pattern, duration in zip(patterns, durations):
            # Create chord from pattern
            chord_notes = []
            for interval in pattern:
                n = note.Note()
                n.pitch.midi = root_note + interval
                chord_notes.append(n)
            
            c = chord.Chord(chord_notes)
            c.quarterLength = duration
            carrier.append(c)
            
            current_length += duration
            if current_length >= desired_length:
                break
    
    return carrier

def embed_metadata(music_stream: stream.Stream, metadata: dict) -> stream.Stream:
    """
    Embed metadata into a music stream using a simple encoding scheme.
    
    :param music_stream: The music stream to embed metadata into
    :param metadata: Dictionary of metadata to embed
    :return: Stream with embedded metadata
    """
    # Convert metadata to a simple string representation
    import json
    metadata_str = json.dumps(metadata)
    
    # Encode the string as a series of MIDI note values
    encoded_data = []
    for char in metadata_str:
        # Map each character to a MIDI note value (32-127)
        encoded_data.append(ord(char))
    
    # Create a hidden metadata track
    metadata_track = stream.Stream()
    
    # Create very short, quiet notes with the encoded data
    for value in encoded_data:
        n = note.Note()
        n.pitch.midi = value
        n.quarterLength = 0.0625  # 1/16th note
        n.volume.velocity = 10    # Very quiet
        metadata_track.append(n)
    
    # Append metadata track to a copy of the original stream
    result = stream.Stream()
    
    # Add the original content
    for elem in music_stream:
        result.append(elem)
    
    # Add metadata as a separate "hidden" part
    result.insert(0, metadata_track)
    
    return result

def extract_metadata(music_stream: stream.Stream) -> dict:
    """
    Extract metadata embedded in a music stream.
    
    :param music_stream: Stream possibly containing embedded metadata
    :return: Dictionary of extracted metadata, or empty dict if none found
    """
    # Look for very short, quiet notes that might contain metadata
    candidate_notes = []
    
    for elem in music_stream.recurse().getElementsByClass('Note'):
        if elem.quarterLength <= 0.0625 and hasattr(elem, 'volume') and elem.volume.velocity <= 15:
            candidate_notes.append(elem)
    
    if not candidate_notes:
        return {}
    
    # Try to decode as metadata
    try:
        # Convert MIDI note values back to characters
        metadata_str = ''.join(chr(n.pitch.midi) for n in candidate_notes)
        
        # Parse as JSON
        import json
        return json.loads(metadata_str)
    except:
        # If decoding fails, return empty dict
        return {}