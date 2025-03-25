"""
Test script to verify the fixing of Music21 deprecation warnings and UI errors.
This script creates a simple test image, converts it to MIDI, then converts that MIDI back to an image.
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw

def create_test_image(filename, size=(64, 64)):
    """Create a simple test image to use for conversion testing"""
    print(f"Creating test image {filename} with size {size[0]}x{size[1]}...")
    
    # Create a simple gradient test image
    image = Image.new("RGB", size)
    draw = ImageDraw.Draw(image)
    
    # Draw gradient background
    for y in range(size[1]):
        for x in range(size[0]):
            r = int(255 * x / size[0])
            g = int(255 * y / size[1])
            b = int(255 * (x + y) / (size[0] + size[1]))
            draw.point((x, y), fill=(r, g, b))
    
    # Add some shapes for visual interest
    # Red circle
    draw.ellipse(
        [(size[0] // 4, size[1] // 4), 
         (size[0] // 4 * 3, size[1] // 4 * 3)], 
        outline=(255, 0, 0), width=2
    )
    
    # Blue rectangle
    draw.rectangle(
        [(size[0] // 3, size[1] // 3), 
         (size[0] // 3 * 2, size[1] // 3 * 2)], 
        outline=(0, 0, 255), width=2
    )
    
    # Save the image
    image.save(filename)
    print(f"Test image saved to {filename}")
    return filename

def main():
    """Main test function"""
    print("\nTesting MIDI-Image Conversion Cycle")
    print("===================================\n")
    
    # Create test directories if they don't exist
    os.makedirs("test_output", exist_ok=True)
    
    # Step 1: Create a test image
    test_image = create_test_image("test_output/test_image.png", (64, 64))
    
    # Step 2: Convert image to MIDI using simple_demo.py
    print("\nConverting test image to MIDI...")
    try:
        # Import our conversion module
        sys.path.insert(0, os.getcwd())
        from simple_demo import process_image, create_simple_midi
        
        # Read the test image
        image = cv2.imread(test_image)
        if image is None:
            print("Error: Could not read test image")
            return
        
        # Process the image
        resolution = (64, 64)
        notes, quarter_lengths, volumes = process_image(image, resolution)
        
        # Get available notes (using a simple C Major scale for demo)
        notes_list = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        
        # Create note data
        note_data = []
        for i in range(len(notes)):
            note_idx = min(int(round(notes[i])), len(notes_list) - 1)
            note = notes_list[note_idx]
            quarter_length = float(round(quarter_lengths[i] * 4) / 4)  # Round to nearest 0.25
            volume = float(volumes[i])
            note_data.append((note, quarter_length, volume))
        
        # Create MIDI file
        midi_output = "test_output/test_output.mid"
        create_simple_midi(note_data, midi_output, original_image=image)
        print(f"Successfully created MIDI file: {midi_output}")
        
        # Step 3: Convert MIDI back to image
        print("\nConverting MIDI back to image...")
        from simple_demo import midi_to_image
        
        # Convert MIDI to image
        image_output = "test_output/reconstructed_image.png"
        midi_to_image(midi_output, image_output, target_resolution=resolution)
        print(f"Successfully created reconstructed image: {image_output}")
        
        print("\nTest completed! If no Music21 deprecation warnings appear, the fix was successful!")
        print("Please check the reconstructed image for accuracy.")
        
    except ImportError as e:
        print(f"Error: Required module not found: {str(e)}")
        print("Make sure all required modules are installed.")
    except Exception as e:
        print(f"Error during testing: {str(e)}")

if __name__ == "__main__":
    main()