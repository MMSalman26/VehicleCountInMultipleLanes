# Multi-Lane Vehicle Counting System

## Project Overview

This project implements a computer vision-based system for counting vehicles across multiple lanes in real-time video footage. It utilizes advanced object detection and tracking algorithms to accurately identify and count vehicles in different lanes of a road.

## Features

- Real-time vehicle detection and counting
- Multi-lane support
- Custom lane definition through user interface
- Performance analysis and visualization tools

## Repository Structure

```
.
├── CountingVehiclesInLanes.py
├── arrange.py
├── SplittingLanes.py
├── requirements.txt
├── README.md
└── data/
    └── sample_video.mp4
```

## Installation

1. Clone this repository:

   ```
   git clone https://github.com/MMSalman26/VehicleCountInMultipleLanes.git
   cd VehicleCountInMultipleLanes
   ```

2. Create a virtual environment (optional but recommended):

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Defining Lanes

Run the `SplittingLanes.py` script to define the lanes in your video:

```
python SplittingLanes.py
```

Click on the video frame to define polygon points for each lane. Press 'Esc' when finished.

### Counting Vehicles

After defining the lanes, run the main counting script:

```
python CountingVehiclesInLanes.py
```

This will process the video and output the vehicle counts for each lane.

## Configuration

- In `CountingVehiclesInLanes.py`, you can adjust the `video_path` variable to point to your own video file.
- The `yolov8n.pt` model is used by default. You can change this to other YOLO models for potentially better performance.

## Results

The system outputs:

- Real-time vehicle counts for each lane
- Performance metrics including detection accuracy and processing time
- Visualizations of traffic patterns and system performance

Example visualizations:

- Average vehicle count per lane
- Vehicle count over time
- Detection accuracy distribution
- Processing time vs. accuracy
- Vehicle count distribution by lane
- Correlation of traffic between lanes

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your changes.

## Acknowledgments

- YOLO for object detection
- OpenCV for image processing
- NumPy and Matplotlib for data handling and visualization

## Contact

For any queries or suggestions, please open an issue on this GitHub repository.
