# dedupe

Interactive CLI for finding and managing duplicate or visually similar images using CNN-based comparison.

## Installation

Requires Python 3.10 and [`viu`](https://github.com/atanunq/viu) for terminal image display.

```bash
brew install viu
uv tool install -e .
```

## Usage

```bash
dedupe <directory> [similarity_threshold]
```

- `similarity_threshold` — float between 0 and 1, default `0.92`. Lower values catch more pairs.

## Interactive Commands

For each duplicate pair shown:

| Key | Action |
|-----|--------|
| `1` | Keep first image, archive second |
| `2` | Keep second image, archive first |
| `s` or Enter | Skip this pair |
| `q` | Quit |

Archived images are moved to `<directory>/duplicates_archive/` — nothing is permanently deleted.

## How It Works

1. CNN embeddings (via `imagededup`) are generated for all images in the target directory
2. Pairs above the similarity threshold are surfaced for review
3. For each pair, detailed metadata is printed (resolution, file size, format) and a side-by-side visual comparison is rendered in the terminal via `viu`
4. Your choice moves one image to the archive or skips the pair

## Supported Formats

JPG, JPEG, PNG, GIF, BMP, TIFF, WebP
