# SoundsAwful Enhanced - Image to Music Cryptography

## Overview
SoundsAwful provides a novel method for cryptographically encoding pictures as music and reversing the encryption with a simple Python GUI. This enhanced version addresses the limitations of the original implementation with significant improvements in functionality, performance, and user experience.

![SoundsAwful Screenshot](https://via.placeholder.com/800x500?text=SoundsAwful+Enhanced+Screenshot)

## Key Improvements

### Enhanced User Interface
- **Image Preview**: Preview selected images before conversion
- **Progress Indicators**: Real-time feedback on conversion progress
- **Musical Key Selection**: Choose from multiple musical keys for different styles
- **Advanced Options**: Configure conversion parameters for better results

### Steganography Implementation
- **Hidden Data**: Conceal image data within natural-sounding music
- **Multiple Algorithms**: Choose from bass note, volume, or duration-based hiding methods
- **Metadata Embedding**: Include additional information in the music file

### Performance Optimizations
- **Parallel Processing**: Multi-threaded image processing for faster conversion
- **Memory Efficiency**: Optimized handling of large images
- **Batch Processing**: Efficient chord generation in memory-friendly batches
- **Caching**: Reduction of redundant operations

### Technical Enhancements
- **Error Handling**: Robust error detection and user feedback
- **Modular Design**: Separated components for easier maintenance and extension
- **Documentation**: Comprehensive code documentation and examples

## Installation

### Requirements
```
numpy==1.19.2
pygubu==0.10.3
scikit_image==0.16.2
music21==6.1.0
opencv-python==4.0.1
pillow>=8.0.0
```

### Setup
1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage
Run the improved UI:
```
python ui_improved.py
```

### Image to Music Conversion
1. Launch the application
2. Click "Select" and choose an image file (.jpg, .png)
3. Adjust resolution settings (higher values = more detail but longer music)
4. Select a musical key from the dropdown
5. Optional: Enable steganography options
6. Click "Convert"
7. Choose a location to save the MIDI file

### Music to Image Conversion
1. Launch the application
2. Click "Select" and choose a MIDI file (.mid)
3. Select the same resolution and key used during encoding
4. If steganography was used, enable the appropriate options
5. Click "Convert"
6. Choose a location to save the reconstructed image

## Advanced Features

### Steganography Options
The steganography module provides several methods to hide data within music:

#### Hide Throughout Music
This option disperses the image data throughout the generated music, making it less obvious that the file contains hidden data.

```python
from steganography import hide_chords_in_music

# Hide data chords within carrier music
hidden_stream = hide_chords_in_music(data_chords, carrier_music_path, hide_factor=0.3)
```

#### Extraction Algorithms
Three different algorithms can be used to extract hidden data:

1. **Bass Note**: Identifies patterns in bass notes to extract data
2. **Volume**: Detects unusual volume patterns
3. **Duration**: Identifies note durations that don't match carrier patterns

```python
from steganography import extract_chords_from_steganography

# Extract data using bass note patterns
extracted_data = extract_chords_from_steganography(
    music_stream, 
    extraction_algorithm='bass_note'
)
```

### Performance Tuning
For large images or performance-critical applications, use the performance module:

```python
from performance import parallel_image_split, memory_optimized_image_processing

# Process large images efficiently
notes, quarter_lengths, volumes = memory_optimized_image_processing(
    image_path, 
    split_number=(128, 128),
    progress_callback=update_progress_bar
)
```

## Technical Details

### The Image-to-Music Algorithm

1. **Image Preparation**:
   - Read the image from file
   - Downscale to the specified resolution
   - Split into RGB channels

2. **Channel Mapping**:
   - Blue channel → musical notes (normalized to fit chosen musical key)
   - Green channel → note duration (normalized between 0.25 and 2.00 quarter-notes)
   - Red channel → note volume (normalized between 20 and 127 MIDI velocity)

3. **Chord Generation**:
   - Generate chords using Baroque progression rules
   - Apply key-specific voicing
   - Create MIDI stream of chords
   
4. **Optional Steganography**:
   - If enabled, distribute data throughout a musical carrier
   - Apply embedding algorithm to hide data structure
   - Add metadata as needed

### The Music-to-Image Algorithm

The decoding process reverses these steps:
1. Extract bass notes, durations, and volumes (or use steganography extraction)
2. Denormalize values back to RGB color ranges
3. Reconstruct image using the extracted data
4. Save as image file

## Module Structure

### ui_improved.py
Enhanced user interface with added features and error handling.

### steganography.py
Implements methods for hiding and extracting data in music.

### performance.py
Contains optimizations for processing large images and generating music more efficiently.

## Development

### Adding New Features
The modular design makes it easy to add new functionality:

1. **New Steganography Algorithms**: Extend `steganography.py` with additional hiding methods
2. **Additional Musical Keys**: Add to the `AVAILABLE_KEYS` dictionary in `ui_improved.py`
3. **Performance Optimizations**: Implement new techniques in `performance.py`

### Future Development
Potential areas for future improvement:

- Machine learning-based steganography detection resistance
- Additional file formats support (SVG, TIFF, WAV, MP3)
- Web-based interface
- Mobile app implementation

## Addressing Original Limitations

The enhanced version directly addresses all limitations mentioned in the original README:

1. **"Hiding the generated chords throughout a piece of music"** - Implemented via the steganography module
2. **"More precise compression when converting from image -> midi"** - Improved with advanced channel mapping and normalization
3. **"A more visually pleasing GUI"** - Completely redesigned with image preview, progress indicators, and better organization
4. **"Quicker conversion (especially with larger resolutions)"** - Implemented parallel processing and memory optimizations
5. **"More comprehensive encryption algorithms"** - Added steganography, metadata embedding, and multiple extraction algorithms

## Contributors
Original project by [Marius Juston](https://github.com/Marius-Juston), [Russell Newton](https://github.com/Russell-Newton), and [Akshin Vemana](https://github.com/AkshinVemana).

Enhanced version improvements by [Your Name].

## License
[MIT License](LICENSE)