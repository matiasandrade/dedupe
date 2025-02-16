#!/usr/bin/env -S uv run --script
# /// script
# requires-python = '>=3.10,<3.11'
# dependencies = [
# "imagededup",
# "numpy", 
# "pillow",
# "typing"
# ]
# ///

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Set

import numpy as np
from imagededup.methods import CNN
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}


def get_similarity_dict(target_dir: Path) -> Dict[str, List[Tuple[str, float]]]:
    cnn_method = CNN()

    # Get list of valid image files and copy to temp dir
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        image_files = []
        
        # Copy valid images to temp dir
        for f in target_dir.iterdir():
            if f.suffix.lower() in IMAGE_EXTENSIONS:
                dest = temp_path / f.name
                shutil.copy2(f, dest)
                image_files.append(str(dest))

        # Find duplicates across all images
        duplicates = cnn_method.find_duplicates(
            image_dir=temp_dir,
            scores=True,
            num_enc_workers=24
        )

    # Convert scores to list format if needed
    filtered_duplicates = {}
    for filename, matches in duplicates.items():
        if not isinstance(matches, list):
            matches = list(matches.items())
        # Map temp paths back to original paths
        orig_filename = str(target_dir / Path(filename).name)
        if matches:
            filtered_duplicates[orig_filename] = [
                (str(target_dir / Path(m[0]).name), m[1]) for m in matches
            ]

    return filtered_duplicates


def create_side_by_side_image(img1_path: str, img2_path: str) -> str:
    with Image.open(img1_path) as img1, Image.open(img2_path) as img2:
        # Calculate target dimensions to make images roughly equal area
        img1_area = img1.width * img1.height
        img2_area = img2.width * img2.height
        
        # Scale the larger image down to match area of smaller image
        if img1_area > img2_area:
            scale = (img2_area / img1_area) ** 0.5
            new_width = int(img1.width * scale)
            new_height = int(img1.height * scale)
            img1 = img1.resize((new_width, new_height), Image.Resampling.LANCZOS)
        elif img2_area > img1_area:
            scale = (img1_area / img2_area) ** 0.5
            new_width = int(img2.width * scale)
            new_height = int(img2.height * scale)
            img2 = img2.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Calculate total width and height with extra space for text
        total_width = img1.width + img2.width
        max_height = max(img1.height, img2.height)
        text_height = 80  # Increased space for larger text
        combined = Image.new(
            "RGB", (total_width, max_height + text_height), color="white"
        )

        # Paste images
        combined.paste(img1, (0, 0))
        combined.paste(img2, (img1.width, 0))

        # Add text labels
        from PIL import ImageDraw, ImageFont

        draw = ImageDraw.Draw(combined)

        # Try to use a default font, fallback to default
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/TTF/DejaVuSansMono.ttf", 36  # Increased font size
            )
        except IOError:
            font = ImageFont.load_default()

        # Draw resolution and file size under first image
        img1_info = f"{img1.width}x{img1.height} - {os.path.getsize(img1_path)/1024:.1f}KB"
        draw.text(
            (img1.width // 2 - 36, max_height + 15), img1_info, fill="black", font=font
        )

        # Draw resolution and file size under second image
        img2_info = f"{img2.width}x{img2.height} - {os.path.getsize(img2_path)/1024:.1f}KB"
        draw.text(
            (img1.width + img2.width // 2 - 36, max_height + 15),
            img2_info,
            fill="black",
            font=font,
        )

        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        combined.save(temp_file.name)
        return temp_file.name


def move_to_archive(path: Path, archive_dir: Path):
    """Move file to archive directory"""
    archive_dir.mkdir(exist_ok=True)
    archive_path = archive_dir / path.name
    shutil.move(path, archive_path)
    print(f"Moved to archive directory: {archive_path}")

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python script.py <target_dir> [similarity]")
        sys.exit(1)

    target_dir = Path(sys.argv[1])
    similarity_threshold = float(sys.argv[2]) if len(sys.argv) == 3 else 0.92
    archive_dir = target_dir / "duplicates_archive"

    print("Finding matches...")
    similarity_dict = get_similarity_dict(target_dir)

    if not similarity_dict:
        print("No matching images found")
        return

    # Keep track of which pairs we've already compared
    compared_pairs = set()

    # Process each group of similar images
    for img1, matches in similarity_dict.items():
        img1_path = Path(img1)
        if not img1_path.exists():  # Skip if already processed
            continue

        for img2, similarity in matches:
            if similarity < similarity_threshold:
                continue
            img2_path = Path(img2)
            if not img2_path.exists():  # Skip if already processed
                continue

            # Create a sorted tuple of the two paths to ensure consistent ordering
            pair = tuple(sorted([str(img1_path), str(img2_path)]))
            if pair in compared_pairs:  # Skip if we've already compared these images
                continue
            compared_pairs.add(pair)

            print(f"\nPotential duplicates (similarity score: {similarity:.3f}):")
            print(f"Image 1: {img1_path}")
            print(f"Image 2: {img2_path}")

            try:
                # Verify files still exist before creating comparison
                if not img1_path.exists() or not img2_path.exists():
                    print("One or both images no longer exist, skipping...")
                    continue
                    
                combined_image = create_side_by_side_image(
                    str(img1_path), str(img2_path)
                )
                subprocess.run(["viu", combined_image], check=True)
                os.unlink(combined_image)
            except subprocess.CalledProcessError:
                print("Error: viu not installed or failed to display images")
            except Exception as e:
                print(f"Error processing images: {e}")
                continue

            while True:
                choice = input(
                    "\nKeep which image? [1]first, [2]second, [s]kip, [q]uit: "
                ).lower()
                
                # Verify files still exist before taking action
                if not img1_path.exists() or not img2_path.exists():
                    print("One or both images no longer exist, skipping...")
                    break
                    
                if choice == "1":
                    if img2_path.exists():
                        move_to_archive(img2_path, archive_dir)
                    break
                elif choice == "2":
                    if img1_path.exists():
                        move_to_archive(img1_path, archive_dir)
                    break
                elif choice == "s":
                    break
                elif choice == "q":
                    return
                else:
                    print("Invalid choice")

if __name__ == "__main__":
    main()
