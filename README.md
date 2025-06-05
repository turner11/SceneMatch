# SceneMatch

A tool for matching and analyzing drone video frames to identify overlapping scenes and differences.

## Features

- Frame matching between drone videos using SIFT features and FAISS for efficient similarity search
- Support for frame metadata including timestamps and locations
- Configurable frame sampling interval
- Efficient vector-based similarity search
- Real-time visualization of frame processing and feature detection
- Side-by-side comparison of matched frames
- Rich CLI interface with progress bars and colored output

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SceneMatch.git
cd SceneMatch
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
uv pip install -r requirements.txt
```

## Usage

### Command Line Interface

The easiest way to use SceneMatch is through the command line interface:

```bash
# Basic usage with visualization
python -m scenematch analyze reference_video.mp4 comparison_video.mp4

# Customize sampling interval
python -m scenematch analyze reference_video.mp4 comparison_video.mp4 --sample-interval 5

# Disable visualization
python -m scenematch analyze reference_video.mp4 comparison_video.mp4 --no-visualize

# Save results to a JSON file
python -m scenematch analyze reference_video.mp4 comparison_video.mp4 --output results.json
```

### Python API

You can also use SceneMatch in your Python code:

```python
from scenematch.frame_matcher import FrameMatcher

# Initialize the frame matcher with visualization enabled
matcher = FrameMatcher(sample_interval=10, visualize=True)

# Build index from reference video (will show frames and detected features)
matcher.build_index("path/to/reference_video.mp4", "video_a")

# Find matches in another video (will show side-by-side comparison of matches)
matches = matcher.find_matches("path/to/comparison_video.mp4", "video_b")

# Process matches
for match in matches:
  print(f"Match found between frames:")
  print(f"Video A: Frame {match.frame.frame_index} at {match.frame.timestamp}s")
  print(f"Video B: Frame {match.frame_reference.frame_index} at {match.frame_reference.timestamp}s")
  print(f"Similarity score: {match.distance_score}")
```

### Visualization Controls

- Press `ESC` to stop the visualization and processing
- The visualization window can be resized by dragging its corners
- During indexing, you'll see:
  - The current frame being processed
  - SIFT keypoints detected in the frame (green circles with orientation lines)
- During matching, you'll see:
  - Side-by-side comparison of matched frames
  - Keypoints in both frames
  - Lines connecting matching keypoints

## Running Tests

To run the test suite:

```bash
pytest tests/
```

## Project Structure

```
scenematch/
├── __init__.py
├── __main__.py
├── frame_matcher.py
├── visualizer.py
├── requirements.txt
└── tests/
    └── test_frame_matcher.py
```

## Dependencies

- OpenCV (opencv-python)
- NumPy
- FAISS (faiss-cpu)
- pytest (for testing)
- coloredlog (for colored logging)
- rich-click (for CLI interface)

## License

MIT License 