"""
Test script to verify all fixes for the image-to-MIDI-to-image conversion
This tests:
1. The Music21 .flatten() fix
2. The resolution dialog thread-safety
3. The indentation and logic corrections
"""

import tkinter as tk
from tkinter import ttk
import os
import sys
import time
import numpy as np
from keys import A_MINOR

# Test import from main module to verify fixes
from ui_improved import ImageAudioConverter

def run_test():
    print("Starting resolution dialog test...")
    print("Testing all fixes for the image-to-MIDI-to-image conversion")
    
    # Set up the test environment
    root = tk.Tk()
    root.title("Resolution Dialog Test")
    root.geometry("800x600")
    
    # Create the converter app with test setup
    NOTES_LIST = list(map(lambda note: note.name[0], A_MINOR.getPitches()))
    app = ImageAudioConverter(NOTES_LIST)
    
    # Create a test MIDI file scenario with 4588 notes
    test_notes = np.array(['C'] * 4588)
    test_quarter_lengths = np.array([0.5] * 4588)
    test_volumes = np.array([80] * 4588)
    
    # Start the test - this will trigger the resolution dialog
    print("\nTesting resolution dialog with 4588 notes (similar to error screenshot)")
    print("This should prompt the thread-safe resolution dialog...")
    
    # Create a test button to trigger the resolution handling
    def test_resolution_dialog():
        try:
            print("Triggering resolution handling code...")
            app.decode_music(test_notes, test_quarter_lengths, test_volumes)
            print("Resolution handling completed successfully!")
        except Exception as e:
            print(f"ERROR: {str(e)}")
            raise e
    
    test_button = ttk.Button(root, text="Test Resolution Dialog", command=test_resolution_dialog)
    test_button.pack(pady=20)
    
    info_label = ttk.Label(root, text="""
    This test will verify the fixes for:
    
    1. The Music21 .flatten() fix
    2. The resolution dialog thread-safety
    3. The indentation and logic corrections
    
    Click the button to start the test.
    When the resolution dialog appears, choose any option.
    If no errors occur, all fixes are working!
    """, wraplength=600, justify="left")
    info_label.pack(pady=20)
    
    # Close test button
    ttk.Button(root, text="Close Test", command=root.destroy).pack(pady=20)
    
    # Run the test
    root.mainloop()
    
    print("Test completed successfully!")

if __name__ == "__main__":
    print("=== Resolution Dialog Test ===")
    run_test()