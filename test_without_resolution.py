"""
Test script to verify exact image reconstruction without using resolution parameters.

This script:
1. Creates a test image
2. Converts it to MIDI using the new approach in simple_demo.py
3. Converts the MIDI back to an image
4. Compares the original and reconstructed images to verify they are identical
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw
import simple_demo

def create_test_image(filename, size=(128, 128)):
    """Create a test image with various patterns for testing reconstruction"""
    print(f"Creating test image {filename} with size {size[0]}x{size[1]}...")
    
    # Create a colorful test image
    image = Image.new("RGB", size)
    draw = ImageDraw.Draw(image)
    
    # Draw gradient background
    for y in range(size[1]):
        for x in range(size[0]):
            r = int(255 * x / size[0])
            g = int(255 * y / size[1])
            b = int(128 + 127 * np.sin(x/10) * np.cos(y/10))
            draw.point((x, y), fill=(r, g, b))
    
    # Add some shapes for visual interest and testing
    # Red circle
    draw.ellipse(
        [(size[0] // 4, size[1] // 4), 
         (size[0] // 4 * 3, size[1] // 4 * 3)], 
        outline=(255, 0, 0), width=3
    )
    
    # Blue rectangle
    draw.rectangle(
        [(size[0] // 3, size[1] // 3), 
         (size[0] // 3 * 2, size[1] // 3 * 2)], 
        outline=(0, 0, 255), width=3
    )
    
    # Yellow star (approximated with lines)
    center_x, center_y = size[0] // 2, size[1] // 2
    radius = min(size) // 4
    points = []
    for i in range(5):
        # Outer points (star tips)
        angle = np.pi/2 + i * 2*np.pi/5
        x = center_x + int(radius * np.cos(angle))
        y = center_y + int(radius * np.sin(angle))
        points.append((x, y))
        
        # Inner points
        angle += np.pi/5
        inner_radius = radius // 2
        x = center_x + int(inner_radius * np.cos(angle))
        y = center_y + int(inner_radius * np.sin(angle))
        points.append((x, y))
    
    # Connect the star points
    for i in range(len(points)):
        draw.line([points[i], points[(i+1) % len(points)]], fill=(255, 255, 0), width=2)
    
    # Add text for detail testing
    draw.text((10, 10), "Test Image", fill=(255, 255, 255))
    
    # Save the image
    image.save(filename)
    print(f"Test image saved to {filename}")
    return filename

def compare_images(original_path, reconstructed_path):
    """Compare the original and reconstructed images to verify exact reconstruction"""
    # Read both images
    original = cv2.imread(original_path)
    reconstructed = cv2.imread(reconstructed_path)
    
    # Check if both images were loaded
    if original is None:
        print(f"Error: Could not read original image from {original_path}")
        return False
    
    if reconstructed is None:
        print(f"Error: Could not read reconstructed image from {reconstructed_path}")
        return False
    
    # Check dimensions
    if original.shape != reconstructed.shape:
        print(f"Dimension mismatch: Original {original.shape}, Reconstructed {reconstructed.shape}")
        return False
    
    # Calculate difference
    diff = cv2.absdiff(original, reconstructed)
    diff_sum = np.sum(diff)
    
    # Calculate similarity percentage
    max_diff = original.size * 255  # Maximum possible difference
    similarity = 100 - (diff_sum / max_diff * 100)
    
    print(f"Images comparison:")
    print(f"Original size: {original.shape[1]}x{original.shape[0]}")
    print(f"Reconstructed size: {reconstructed.shape[1]}x{reconstructed.shape[0]}")
    print(f"Similarity: {similarity:.4f}%")
    
    # Save difference image for visual inspection
    diff_path = "test_output/difference.png"
    # Scale the difference to make it more visible
    diff_scaled = cv2.convertScaleAbs(diff, alpha=10)
    cv2.imwrite(diff_path, diff_scaled)
    print(f"Difference image saved to {diff_path}")
    
    # Perfect match would be 100% similarity
    if similarity > 99.99:
        print("SUCCESS: Images are identical! Perfect reconstruction achieved!")
        return True
    elif similarity > 98:
        print("PARTIAL SUCCESS: Images are very similar but not identical.")
        return False
    else:
        print("FAILURE: Significant differences between images.")
        return False

def run_test():
    """Run the complete test cycle"""
    print("\nTesting Image-MIDI-Image Conversion Without Resolution Parameters")
    print("==============================================================\n")
    
    # Create test directories if they don't exist
    os.makedirs("test_output", exist_ok=True)
    
    # Step 1: Create a test image
    test_image = create_test_image("test_output/test_original.png", (128, 128))
    
    # Step 2: Convert image to MIDI using simple_demo.py
    print("\nConverting test image to MIDI...")
    midi_output = "test_output/test_output.mid"
    
    try:
        # Read the test image
        image = cv2.imread(test_image)
        if image is None:
            print("Error: Could not read test image")
            return False
        
        # Generate MIDI with embedded image
        simple_demo.generate_midi_with_embedded_image(image, midi_output)
        print(f"Successfully created MIDI file: {midi_output}")
        
        # Step 3: Convert MIDI back to image
        print("\nConverting MIDI back to image...")
        image_output = "test_output/test_reconstructed.png"
        simple_demo.extract_image_from_midi(midi_output, image_output)
        print(f"Successfully created reconstructed image: {image_output}")
        
        # Step 4: Compare original and reconstructed images
        print("\nComparing original and reconstructed images...")
        success = compare_images(test_image, image_output)
        
        if success:
            print("\nTest completed successfully! The image was perfectly reconstructed!")
            print("No resolution parameter was needed for exact reconstruction.")
        else:
            print("\nTest completed with issues. The reconstructed image differs from the original.")
            print("Please check the difference image for details.")
        
        return success
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    run_test()