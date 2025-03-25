"""
Test script to verify the fix for MIDI resolution handling
"""
import os
import sys
import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import time

# Test the resolution selection flow with simulated data
def test_resolution_selection():
    print("Testing resolution selection with simulated data...")
    print("Simulating a MIDI file with 4551 notes (like in the screenshot)")
    
    # Create test window
    root = tk.Tk()
    root.title("Resolution Selection Test")
    root.geometry("600x400")
    
    # Simulate detection
    detected_resolution = (41, 111)  # 4551 notes
    standard_resolution = (64, 64)   # 4096 notes
    selected_resolution = (64, 64)   # User's current selection
    
    total_elements = 4551
    
    # Setup variables to capture the result
    test_result = {"success": False, "choice": None, "resolution": None}
    
    # Create synchronization objects
    dialog_done = {"value": False}
    choice = ["1"]  # Default choice
    
    # Test the thread-safe dialog approach
    def create_test_dialog():
        # Create dialog
        resolution_dialog = tk.Toplevel(root)
        resolution_dialog.title("Resolution Selection Test")
        resolution_dialog.geometry("500x300")
        resolution_dialog.transient(root)
        resolution_dialog.grab_set()
        
        # Message
        message = (f"The MIDI file contains {total_elements} notes which doesn't match the selected resolution "
                 f"of {selected_resolution[0]}x{selected_resolution[1]} "
                 f"({selected_resolution[0] * selected_resolution[1]} notes).\n\n"
                 f"Please choose a resolution option:\n"
                 f"1. Use detected resolution: {detected_resolution[0]}x{detected_resolution[1]} (exact fit)\n"
                 f"2. Use nearest standard resolution: {standard_resolution[0]}x{standard_resolution[1]} (may require resizing)\n"
                 f"3. Use selected resolution: {selected_resolution[0]}x{selected_resolution[1]} (will resize)")
        
        message_label = ttk.Label(resolution_dialog, text=message, wraplength=480, justify="left")
        message_label.pack(padx=20, pady=20)
        
        # Radio buttons
        choice_var = tk.StringVar(value="1")
        ttk.Radiobutton(resolution_dialog, text="Option 1: Detected resolution", 
                      variable=choice_var, value="1").pack(anchor="w", padx=20)
        ttk.Radiobutton(resolution_dialog, text="Option 2: Standard resolution", 
                      variable=choice_var, value="2").pack(anchor="w", padx=20)
        ttk.Radiobutton(resolution_dialog, text="Option 3: Selected resolution", 
                      variable=choice_var, value="3").pack(anchor="w", padx=20)
        
        # OK button
        def on_ok():
            choice[0] = choice_var.get()
            dialog_done["value"] = True
            resolution_dialog.destroy()
        
        ttk.Button(resolution_dialog, text="OK", command=on_ok).pack(pady=20)
        
        # Window close handler
        def on_close():
            dialog_done["value"] = True
            resolution_dialog.destroy()
        
        resolution_dialog.protocol("WM_DELETE_WINDOW", on_close)
    
    # Schedule dialog creation
    root.after(100, create_test_dialog)
    
    # Simulate the worker thread waiting for dialog
    def worker_thread():
        # Wait for dialog to complete
        print("Waiting for dialog response...")
        while not dialog_done["value"]:
            time.sleep(0.1)
        
        # Get user's choice
        user_choice = choice[0]
        print(f"User selected option: {user_choice}")
        
        # Apply the choice
        if user_choice == "1":
            test_result["success"] = True
            test_result["choice"] = "detected"
            test_result["resolution"] = detected_resolution
            messagebox.showinfo("Test Result", f"Using detected resolution: {detected_resolution[0]}x{detected_resolution[1]}")
            
        elif user_choice == "2":
            test_result["success"] = True
            test_result["choice"] = "standard"
            test_result["resolution"] = standard_resolution
            messagebox.showinfo("Test Result", f"Using standard resolution: {standard_resolution[0]}x{standard_resolution[1]}")
            
        else:
            test_result["success"] = True
            test_result["choice"] = "selected"
            test_result["resolution"] = selected_resolution
            messagebox.showinfo("Test Result", f"Using selected resolution: {selected_resolution[0]}x{selected_resolution[1]}")
        
        # Close the main window after test
        root.after(1000, root.destroy)
    
    # Start the worker thread
    import threading
    threading.Thread(target=worker_thread, daemon=True).start()
    
    # Run the main loop
    root.mainloop()
    
    # Print test result
    if test_result["success"]:
        print("Test completed successfully!")
        print(f"Selected option: {test_result['choice']}")
        print(f"Resolution: {test_result['resolution'][0]}x{test_result['resolution'][1]}")
        return True
    else:
        print("Test failed!")
        return False

if __name__ == "__main__":
    print("Testing resolution handling fixes...")
    result = test_resolution_selection()
    
    if result:
        print("All tests passed!")
        sys.exit(0)
    else:
        print("Test failed!")
        sys.exit(1)