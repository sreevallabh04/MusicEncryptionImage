"""
MIDI to Image Conversion Demo

This script demonstrates how to convert MIDI files to images with 
automatic resolution detection to fix the "cannot reshape array" error.
"""

import os
import sys
import numpy as np
import cv2
import music21
from music21 import converter

# Import from the main project
import musicgen
from keys import *

# Constants
DEFAULT_KEY = A_MINOR
MIN_QUARTER_LENGTH = 0.25
MAX_QUARTER_LENGTH = 2.0
MIN_VOLUME = 20
MAX_VOLUME = 127

def print_progress(percentage, message=""):
    """Print progress bar to console"""
    bar_length = 30
    filled_length = int(bar_length * percentage / 100)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write(f'\r[{bar}] {percentage:.1f}% {message}')
    sys.stdout.flush()

def detect_best_resolution(total_elements):
    """
    Find the best resolution that balances:
    1. Exact fit for the number of elements
    2. Closest to standard resolutions (powers of 2)
    3. Good aspect ratio (near square or 4:3/16:9)
    
    Args:
        total_elements: Total number of elements to fit
        
    Returns:
        Tuple of (rows, cols) representing the optimal resolution
    """
    # Standard target resolutions (in order of preference)
    standard_resolutions = [
        (64, 64),    # Square, power of 2 (4096 elements)
        (128, 128),  # Square, power of 2 (16384 elements)
        (32, 32),    # Square, power of 2 (1024 elements)
        (256, 256),  # Square, power of 2 (65536 elements)
        (120, 90),   # 4:3 aspect ratio (10800 elements)
        (160, 90),   # 16:9 aspect ratio (14400 elements)
        (80, 60),    # 4:3 aspect ratio (4800 elements)
    ]
    
    # Check if we're already close to a standard resolution (within 5%)
    for std_res in standard_resolutions:
        std_elements = std_res[0] * std_res[1]
        if abs(total_elements - std_elements) / std_elements < 0.05:
            print(f"Note count {total_elements} is close to standard resolution {std_res[0]}x{std_res[1]} ({std_elements} elements)")
            return std_res
    
    # Find exact divisors
    exact_dimensions = []
    
    # First, check if any powers of 2 are factors
    for i in [2, 4, 8, 16, 32, 64, 128, 256]:
        if total_elements % i == 0:
            other_dim = total_elements // i
            exact_dimensions.append((i, other_dim))
            
            # If both dimensions are powers of 2, prioritize this
            if (other_dim & (other_dim-1) == 0) and other_dim != 0:
                return (i, other_dim)  # Return immediately if we find a "perfect" resolution
    
    # If no power-of-2 dimensions found, try all divisors
    if not exact_dimensions:
        for i in range(1, int(np.sqrt(total_elements)) + 1):
            if total_elements % i == 0:
                exact_dimensions.append((i, total_elements // i))
    
    # If we found exact divisors, choose the one closest to a square or with best aspect ratio
    if exact_dimensions:
        # Calculate aspect ratio score (1.0 is perfect square, lower is better)
        dims_with_score = [(dims, abs(dims[0]/dims[1] - 1.0)) for dims in exact_dimensions]
        
        # Also calculate closeness to 4:3 and 16:9 aspect ratios
        for i, (dims, _) in enumerate(dims_with_score):
            ar_4_3_score = abs(dims[0]/dims[1] - 4/3)
            ar_16_9_score = abs(dims[0]/dims[1] - 16/9)
            # Use the better of square, 4:3, or 16:9 scores
            best_score = min(dims_with_score[i][1], ar_4_3_score, ar_16_9_score)
            dims_with_score[i] = (dims, best_score)
        
        # Return the dimensions with the best aspect ratio score
        return min(dims_with_score, key=lambda x: x[1])[0]
    
    # If we can't find exact divisors (which shouldn't happen), find closest standard resolution
    nearest_std_res = min(standard_resolutions, 
                        key=lambda res: abs(res[0] * res[1] - total_elements))
    
    print(f"Warning: Could not find exact divisors for {total_elements} elements.")
    print(f"Using nearest standard resolution: {nearest_std_res[0]}x{nearest_std_res[1]}")
    return nearest_std_res

def midi_to_image(midi_file, target_resolution=None, output_filename=None):
    """
    Convert a MIDI file to an image with enhanced reconstruction accuracy
    
    Args:
        midi_file: Path to the MIDI file
        target_resolution: Optional tuple (rows, cols) or None for auto-detection
        output_filename: Optional output filename, defaults to input filename with .png extension
    
    Returns:
        Path to the saved image file
    """
    print(f"Processing MIDI file: {midi_file}")
    
    if not os.path.exists(midi_file):
        raise FileNotFoundError(f"MIDI file not found: {midi_file}")
    
    # Create default output filename if not provided
    if not output_filename:
        output_filename = os.path.splitext(midi_file)[0] + ".png"
    
    # Decode MIDI
    print("Extracting notes from MIDI file...")
    cypher = musicgen.rules.TriadBaroqueCypher(DEFAULT_KEY)
    note_identifiers = musicgen.decode(midi_file, cypher)
    
    # Extract note properties
    notes = []
    quarter_lengths = []
    volumes = []
    
    print_progress(10, "Extracting note properties...")
    for note_identifier in note_identifiers:
        notes.append(note_identifier[0])
        quarter_lengths.append(note_identifier[1])
        volumes.append(note_identifier[2])
    
    print(f"\nExtracted {len(notes)} notes from MIDI file.")
    
    # Convert note names to indices
    notes_list = list(map(lambda note: note.name[0], DEFAULT_KEY.getPitches()))
    
    # Enhanced processing for better reconstruction
    print_progress(30, "Processing notes with enhanced accuracy...")
    
    # Extract any embedded data from subtle variations in quarter lengths and volumes
    quarter_lengths_array = np.array(quarter_lengths)
    volumes_array = np.array(volumes)
    
    # Arrays to store extracted enhancement data
    notes_min = np.zeros_like(quarter_lengths_array)
    notes_max = np.zeros_like(volumes_array)
    
    # Extract embedded min/max information
    for i in range(len(quarter_lengths_array)):
        # Extract fractional part that contains min information
        frac_part = quarter_lengths_array[i] % 0.125
        if frac_part > 0.01 and frac_part < 0.12:  # Threshold for valid embedded data
            notes_min[i] = frac_part / 0.01
        
        # Clean up the quarter length by removing the embedded data
        quarter_lengths_array[i] = quarter_lengths_array[i] - frac_part + (0.125 if frac_part > 0.06 else 0)
        
        # Extract the volume variation that contains max information
        vol_frac = volumes_array[i] % 1.0
        if vol_frac > 0.1 and vol_frac < 0.9:  # Threshold for valid embedded data
            notes_max[i] = vol_frac / 0.1
        
        # Clean up the volume by removing the embedded data
        volumes_array[i] = np.floor(volumes_array[i])
    
    # Process notes with enhanced color precision 
    notes_array = np.array([notes_list.index(note[0]) if note[0] in notes_list else 0 for note in notes])
    
    # Increase the range to allow for more color values
    # Use 12-note scale instead of 7-note scale for better color resolution
    notes_array = notes_array * (255.0 / max(6, len(notes_list) - 1))
    
    # Apply the extracted min/max information to enhance color accuracy
    # Scale to 0-255 range
    notes_min = np.clip(notes_min, 0, 3) * (255.0 / 3.0)  
    notes_max = np.clip(notes_max, 0, 3) * (255.0 / 3.0)
    
    # Blend values for better reconstruction
    enhanced_notes = np.zeros_like(notes_array)
    for i in range(len(notes_array)):
        if notes_min[i] > 0 or notes_max[i] > 0:
            # We have enhanced data - use weighted blend
            blend_factor = 0.6  # Weight for main note value
            min_factor = 0.2    # Weight for min value
            max_factor = 0.2    # Weight for max value
            
            enhanced_notes[i] = (notes_array[i] * blend_factor + 
                               notes_min[i] * min_factor + 
                               notes_max[i] * max_factor)
        else:
            # No enhanced data - use original
            enhanced_notes[i] = notes_array[i]
    
    # Use the enhanced notes for better color reproduction
    notes_array = enhanced_notes
    
    print_progress(40, "Processing timing...")
    quarter_lengths_array = (quarter_lengths_array - MIN_QUARTER_LENGTH) * 255 / (MAX_QUARTER_LENGTH - MIN_QUARTER_LENGTH)
    
    print_progress(50, "Processing volume...")
    volumes_array = (volumes_array - MIN_VOLUME) * 255 / (MAX_VOLUME - MIN_VOLUME)
    
    # Handle resolution
    total_elements = len(notes)
    
    # If target resolution not provided or doesn't match note count
    if target_resolution is None or (target_resolution[0] * target_resolution[1] != total_elements):
        # Auto-detect resolution
        detected_resolution = detect_best_resolution(total_elements)
        
        if target_resolution is None:
            print(f"\nAuto-detected resolution: {detected_resolution[0]}x{detected_resolution[1]}")
            resolution = detected_resolution
        else:
            print(f"\nNote count ({total_elements}) doesn't match requested resolution "
                  f"({target_resolution[0]}x{target_resolution[1]} = {target_resolution[0]*target_resolution[1]})")
            print(f"Detected resolution: {detected_resolution[0]}x{detected_resolution[1]}")
            
            # Present better options to the user
            print("\nResolution Options:")
            print(f"1. Use detected resolution: {detected_resolution[0]}x{detected_resolution[1]} (exact fit)")
            
            # Find nearest standard resolution
            nearest_standard = None
            standard_resolutions = [(64, 64), (128, 128), (32, 32), (120, 90), (160, 90)]
            nearest_standard = min(standard_resolutions, 
                                  key=lambda res: abs(res[0] * res[1] - total_elements))
            
            print(f"2. Use standard resolution: {nearest_standard[0]}x{nearest_standard[1]} (may require resizing)")
            print(f"3. Use requested resolution: {target_resolution[0]}x{target_resolution[1]} (will resize)")
            
            while True:
                choice = input("Choose option (1/2/3): ").strip()
                
                if choice == '1':
                    resolution = detected_resolution
                    break
                elif choice == '2':
                    resolution = nearest_standard
                    target_size = resolution[0] * resolution[1]
                    
                    if total_elements > target_size:
                        # Truncate excess data
                        print(f"Truncating excess data ({total_elements-target_size} notes) to fit {resolution[0]}x{resolution[1]}")
                        notes_array = notes_array[:target_size]
                        quarter_lengths_array = quarter_lengths_array[:target_size]
                        volumes_array = volumes_array[:target_size]
                    else:
                        # Pad with zeros
                        pad_size = target_size - total_elements
                        print(f"Padding with {pad_size} zeros to fit {resolution[0]}x{resolution[1]}")
                        notes_array = np.pad(notes_array, (0, pad_size), 'constant', constant_values=0)
                        quarter_lengths_array = np.pad(quarter_lengths_array, (0, pad_size), 'constant', constant_values=0)
                        volumes_array = np.pad(volumes_array, (0, pad_size), 'constant', constant_values=0)
                    break
                elif choice == '3':
                    # Use the user's resolution by truncating or padding
                    resolution = target_resolution
                    target_size = resolution[0] * resolution[1]
                    
                    if total_elements > target_size:
                        # Truncate excess data
                        print(f"Truncating excess data ({total_elements-target_size} notes) to fit {resolution[0]}x{resolution[1]}")
                        notes_array = notes_array[:target_size]
                        quarter_lengths_array = quarter_lengths_array[:target_size]
                        volumes_array = volumes_array[:target_size]
                    else:
                        # Pad with zeros
                        pad_size = target_size - total_elements
                        print(f"Padding with {pad_size} zeros to fit {resolution[0]}x{resolution[1]}")
                        notes_array = np.pad(notes_array, (0, pad_size), 'constant', constant_values=0)
                        quarter_lengths_array = np.pad(quarter_lengths_array, (0, pad_size), 'constant', constant_values=0)
                        volumes_array = np.pad(volumes_array, (0, pad_size), 'constant', constant_values=0)
                    break
                else:
                    print("Please choose option 1, 2, or 3")
    else:
        # Use the provided resolution since it matches the note count
        resolution = target_resolution
        print(f"\nUsing provided resolution: {resolution[0]}x{resolution[1]}")
    
    # Reshape arrays to image dimensions
    shape = (resolution[0], resolution[1], 1)
    print_progress(60, f"Reshaping to {resolution[0]}x{resolution[1]}...")
    
    notes_array = notes_array.reshape(shape)
    quarter_lengths_array = quarter_lengths_array.reshape(shape)
    volumes_array = volumes_array.reshape(shape)
    
    # Combine channels
    print_progress(80, "Creating image...")
    image = np.concatenate((notes_array, quarter_lengths_array, volumes_array), axis=2)
    image = image.astype(np.uint8)
    
    # Save image
    print_progress(90, f"Saving to {output_filename}...")
    cv2.imwrite(output_filename, image)
    
    print(f"\nConversion complete! Image saved to {output_filename}")
    return output_filename

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python midi_to_image_demo.py <midi_file> [rows cols]")
        print("  midi_file: Path to the MIDI file to convert")
        print("  rows cols: Optional resolution (e.g., 64 64)")
        sys.exit(1)
    
    midi_file = sys.argv[1]
    
    # Check if resolution is provided
    if len(sys.argv) >= 4:
        try:
            rows = int(sys.argv[2])
            cols = int(sys.argv[3])
            resolution = (rows, cols)
        except ValueError:
            print("Resolution must be positive integers")
            sys.exit(1)
    else:
        resolution = None
    
    try:
        output_file = midi_to_image(midi_file, resolution)
        print(f"Successfully converted {midi_file} to {output_file}")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)