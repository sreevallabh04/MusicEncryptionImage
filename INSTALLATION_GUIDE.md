# SoundsAwful Installation Guide

This document provides detailed instructions for setting up and running the SoundsAwful application, including troubleshooting common installation issues.

## Prerequisites

- Python 3.6+ (3.8 or 3.9 recommended)
- pip package manager
- Virtual environment (recommended)

## Step-by-Step Installation

### 1. Set Up a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies in the Correct Order

Dependencies should be installed in a specific order to avoid conflicts:

```bash
# First install numpy (required by other packages)
pip install numpy==1.19.2

# Then install basic dependencies
pip install music21==6.1.0
pip install Pillow>=8.0.0

# Install OpenCV (with headless version if you have issues)
pip install opencv-python==4.0.1
# OR, if having issues:
# pip install opencv-python-headless==4.0.1

# Install pygubu for the GUI
pip install pygubu==0.10.3

# Finally install scikit-image
pip install scikit-image==0.16.2
```

### 3. Verify Installation

Check if all dependencies are installed correctly:

```bash
pip list | grep -E "numpy|pygubu|scikit_image|music21|opencv-python|pillow"
```

## Troubleshooting Common Issues

### ModuleNotFoundError: No module named 'pygubu'

**Solution**: Install pygubu manually:
```bash
pip install pygubu==0.10.3
```

### Issues with scikit-image Installation

**Solution 1**: Install from a wheel file:
```bash
# Download wheel file for your platform
pip install path/to/scikit_image-0.16.2-wheel-file.whl
```

**Solution 2**: Update setuptools and try again:
```bash
pip install --upgrade setuptools
pip install scikit-image==0.16.2
```

### OpenCV Installation Problems

**Solution**: Try the headless version:
```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

## Running the Application

### Running the Original Version
```bash
python ui.py
```

### Running the Enhanced Version
```bash
python ui_improved.py
```

## Alternative Ways to Run

### Without the GUI (Command Line Mode)

If the GUI dependencies are problematic, you can use this simplified script to test the core functionality:

```python
# Create a new file named 'simple_test.py' with this content:

import cv2
import numpy as np
from musicgen import create_chords
from keys import A_MINOR
import musicgen.rules

# Load image
image_path = 'path/to/image.jpg'  # Replace with your image path
image = cv2.imread(image_path)

# Resize image to manageable size
image = cv2.resize(image, (64, 64))

# Split into channels
blue = image[:, :, 0].flatten()
green = image[:, :, 1].flatten()
red = image[:, :, 2].flatten()

# Normalize
blue = blue / 255.0 * 7  # 7 notes in a scale
green = green / 255.0 * 1.75 + 0.25  # Between 0.25 and 2.0
red = red / 255.0 * 107 + 20  # Between 20 and 127

# Create note data
notes_list = list(map(lambda note: note.name[0], A_MINOR.getPitches()))
note_data = []
for b, g, r in zip(blue, green, red):
    note_idx = int(round(b))
    if note_idx >= len(notes_list):
        note_idx = len(notes_list) - 1
    note_data.append((notes_list[note_idx], float(round(g * 4) / 4), float(r)))

# Create cypher
cypher = musicgen.rules.TriadBaroqueCypher(A_MINOR)

# Generate chords
print("Generating music...")
output = create_chords(note_data[:100], cypher)  # Use only first 100 notes for test

# Save to file
output_path = 'output_test.mid'
output.write('midi', output_path)
print(f"Saved to {output_path}")
```

Run with:
```bash
python simple_test.py
```

## Docker Alternative (Advanced)

For a consistent environment, you can use Docker:

```dockerfile
# Dockerfile
FROM python:3.8-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies in the correct order
COPY requirements.txt .
RUN pip install numpy==1.19.2 && \
    pip install music21==6.1.0 Pillow && \
    pip install opencv-python-headless==4.0.1 && \
    pip install pygubu==0.10.3 && \
    pip install scikit-image==0.16.2

# Copy application files
COPY . .

# Command to run the application
CMD ["python", "ui.py"]
```

Build and run:
```bash
docker build -t sounds-awful .
docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix sounds-awful
```

## Contact for Help

If you continue to experience issues, please open an issue on the GitHub repository with:
1. Your operating system and Python version
2. Full error message
3. Steps you've taken to troubleshoot

## Additional Resources

- [pygubu documentation](https://github.com/alejandroautalan/pygubu)
- [music21 documentation](https://web.mit.edu/music21/doc/)
- [OpenCV Python tutorials](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)