
# Exercise Analysis and Repetition Counter

This project provides an AI-powered solution for exercise analysis, focusing on counting repetitions for specific exercises such as bicep curls and side lateral raises. It leverages computer vision and machine learning techniques to ensure accurate tracking and analysis while maintaining a user-friendly interface.

## Features

- **Repetition Counting**: Automatically count repetitions for bicep curls and side lateral raises.
- **Real-Time Analysis**: Analyze exercises in real-time while maintaining accuracy.
- **Extendable Design**: Easily add support for new exercises.
- **Utility Functions**: Helper functions to facilitate exercise detection and analysis.

## Directory Structure

- `main.py`: Entry point for the application, responsible for setting up and running the analysis.
- `bicep_curl.py`: Contains logic and functions specific to analyzing bicep curls.
- `side_lateral_raise.py`: Contains logic and functions specific to analyzing side lateral raises.
- `utils.py`: Includes utility functions that support the core functionality, such as common calculations and pre-processing.

## Installation

### Prerequisites
- Python 3.8 or above
- Required libraries: `numpy`, `opencv-python`, `mediapipe`

### Steps
1. Clone the repository:
   ```bash
   git clone 
   cd pose-detection-backend
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   python main.py
   ```
2. Follow the on-screen instructions to start the exercise analysis.

## Adding New Exercises

1. Create a new Python file for the exercise (e.g., `new_exercise.py`).
2. Define functions for detecting exercise-specific movements and repetitions.
3. Integrate the new file into `main.py`.
