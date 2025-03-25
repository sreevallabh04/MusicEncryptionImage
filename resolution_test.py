"""
Resolution Test for MIDI to Image Conversion

This script tests the improved resolution handling in midi_to_image_demo.py
by simulating a MIDI file with 4551 notes (like in the screenshot).
"""

import sys
import numpy as np
from midi_to_image_demo import detect_best_resolution

def test_resolution_detection():
    """Test the improved resolution detection algorithm"""
    print("Testing resolution detection with various note counts...")
    
    # Test with 4551 notes (from the screenshot)
    test_case_1 = 4551
    print(f"\nTest case 1: {test_case_1} notes (like in the screenshot)")
    resolution = detect_best_resolution(test_case_1)
    print(f"Best resolution: {resolution[0]} x {resolution[1]} = {resolution[0] * resolution[1]} elements")
    print(f"Aspect ratio: {resolution[0] / resolution[1]:.2f}")
    
    # Test with close to a standard resolution
    test_case_2 = 4100  # Close to 64x64 = 4096
    print(f"\nTest case 2: {test_case_2} notes (close to 64x64 standard)")
    resolution = detect_best_resolution(test_case_2)
    print(f"Best resolution: {resolution[0]} x {resolution[1]} = {resolution[0] * resolution[1]} elements")
    print(f"Aspect ratio: {resolution[0] / resolution[1]:.2f}")
    
    # Test with a perfect power-of-2 square
    test_case_3 = 4096  # Exactly 64x64
    print(f"\nTest case 3: {test_case_3} notes (exactly 64x64)")
    resolution = detect_best_resolution(test_case_3)
    print(f"Best resolution: {resolution[0]} x {resolution[1]} = {resolution[0] * resolution[1]} elements")
    print(f"Aspect ratio: {resolution[0] / resolution[1]:.2f}")
    
    # Test with a number that has good 4:3 aspect ratio factors
    test_case_4 = 4800  # Can be 80x60 (4:3 ratio)
    print(f"\nTest case 4: {test_case_4} notes (can be 80x60 with 4:3 ratio)")
    resolution = detect_best_resolution(test_case_4)
    print(f"Best resolution: {resolution[0]} x {resolution[1]} = {resolution[0] * resolution[1]} elements")
    print(f"Aspect ratio: {resolution[0] / resolution[1]:.2f}")

def simulate_mismatch_scenario():
    """Simulate the mismatch scenario from the screenshot"""
    print("\nSimulating mismatch scenario from screenshot...")
    
    total_elements = 4551
    target_resolution = (64, 64)  # 4096 elements
    
    print(f"MIDI file contains {total_elements} notes which doesn't match the")
    print(f"selected resolution of {target_resolution[0]}x{target_resolution[1]} ({target_resolution[0] * target_resolution[1]} notes).")
    
    # Detect best resolution
    detected_resolution = detect_best_resolution(total_elements)
    print(f"\nDetected resolution: {detected_resolution[0]}x{detected_resolution[1]}")
    
    # Find nearest standard resolution
    standard_resolutions = [(64, 64), (128, 128), (32, 32), (120, 90), (160, 90)]
    nearest_standard = min(standard_resolutions, 
                          key=lambda res: abs(res[0] * res[1] - total_elements))
    
    print(f"Nearest standard resolution: {nearest_standard[0]}x{nearest_standard[1]}")
    
    print("\nResolution Options:")
    print(f"1. Use detected resolution: {detected_resolution[0]}x{detected_resolution[1]} (exact fit)")
    print(f"2. Use standard resolution: {nearest_standard[0]}x{nearest_standard[1]} (may require resizing)")
    print(f"3. Use requested resolution: {target_resolution[0]}x{target_resolution[1]} (will resize)")

if __name__ == "__main__":
    print("Resolution Handling Test")
    print("=======================")
    
    test_resolution_detection()
    simulate_mismatch_scenario()
    
    print("\nTest completed successfully!")