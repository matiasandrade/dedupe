# Image Deduplication Tool

An interactive command-line tool that helps you find and manage duplicate or similar images using CNN-based image recognition.

## Features

- Finds similar images using CNN-based comparison
- Shows side-by-side image comparisons with resolution and file size information
- Interactive CLI interface for deciding which images to keep
- Safely archives removed duplicates instead of deleting them
- Supports multiple image formats: JPG, JPEG, PNG, GIF, BMP, TIFF, WebP
- Maintains aspect ratios when displaying comparisons
- Scales images appropriately for fair comparison

## Requirements

- Python 3.10
- uv package manager
- viu terminal image viewer
- Required Python packages (automatically installed by uv):
  - imagededup
  - numpy
  - pillow
  - typing

## Installation

1. Make sure you have Python 3.10 and uv installed
2. Install the viu terminal image viewer:
   ```bash
   # On Ubuntu/Debian
   apt-get install viu
   
   # On macOS
   brew install viu
   ```
3. Clone this repository and navigate to the script directory
4. The script will automatically install required Python dependencies using uv

## Usage

Run the script by providing the target directory containing images:

```bash
./dedup_images.py /path/to/image/directory
```

### Interactive Commands

When reviewing potential duplicates, you'll be presented with these options:

- `1` - Keep the first image, move the second to archive
- `2` - Keep the second image, move the first to archive
- `b` - Keep both images
- `s` - Skip this pair
- `q` - Quit the program

### Output

- Displays similar image pairs with their similarity scores
- Shows side-by-side comparison with resolution and file size information
- Creates a `duplicates_archive` directory within the target directory
- Moved duplicates are placed in the archive directory

## How It Works

1. Uses CNN-based image comparison from the `imagededup` library
2. Compares all images in the target directory
3. For each pair of similar images:
   - Displays them side by side with metadata
   - Scales images to equal areas for fair comparison
   - Shows resolution and file size information
4. Allows user to decide which image to keep
5. Safely moves unwanted duplicates to an archive directory

## Note

- The script uses a neural network for comparison, so results may vary based on image content
- Comparison is based on visual similarity, not byte-for-byte comparison
- Archived files are moved, not deleted, so you can recover them if needed
- The script requires the viu terminal image viewer for displaying comparisons

## Error Handling

- Gracefully handles missing viu installation
- Skips already processed or missing images
- Validates user input for image selection
- Creates archive directory if it doesn't exist

## License

This script is provided as-is. Feel free to modify and distribute as needed.
