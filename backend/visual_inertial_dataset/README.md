# Visual-Inertial Dataset for Drone

## Dataset Overview
This dataset contains synchronized stereo camera images, IMU measurements, and ground truth data for a simulated indoor drone flight.

## Dataset Specifications

### Duration
- Total duration: 30 seconds
- Indoor flight trajectory with turns, altitude changes, and rotations

### Stereo Cameras (cam0 and cam1)
- Frame rate: 20 Hz
- Resolution: 640×480 pixels
- Baseline: 0.12 meters
- Total frames per camera: 600
- Format: PNG images
- Naming: {timestamp_ns}.png

### IMU (imu0)
- Sample rate: 200 Hz
- Total samples: 6000
- Measurements:
  - 3-axis gyroscope (rad/s)
  - 3-axis accelerometer (m/s²)

### IMU Sensor Characteristics
- Accelerometer noise density: 0.01 m/s²/√Hz
- Accelerometer bias random walk: 0.0002 m/s³/√Hz
- Gyroscope noise density: 0.0002 rad/s/√Hz
- Gyroscope bias random walk: 4e-06 rad/s²/√Hz

### Ground Truth
- Sample rate: 200 Hz (synchronized with IMU)
- Total samples: 6000
- Data includes:
  - Position (x, y, z) in meters
  - Orientation as quaternion (w, x, y, z)
  - Orientation as Euler angles (roll, pitch, yaw) in radians
  - Velocity (vx, vy, vz) in m/s

## Directory Structure
```
visual_inertial_dataset/
├── cam0/
│   ├── data/              # Left camera images
│   └── timestamps.csv     # Camera frame timestamps
├── cam1/
│   ├── data/              # Right camera images
│   └── timestamps.csv     # Camera frame timestamps
├── imu0/
│   └── imu_data.csv       # IMU measurements
├── ground_truth/
│   └── ground_truth.csv   # Ground truth pose and velocity
└── README.md              # This file
```

## File Formats

### IMU Data (imu_data.csv)
Columns:
- timestamp_ns: Timestamp in nanoseconds
- timestamp_s: Timestamp in seconds
- gyro_x, gyro_y, gyro_z: Angular velocity in rad/s
- accel_x, accel_y, accel_z: Specific force in m/s²

### Ground Truth (ground_truth.csv)
Columns:
- timestamp_ns: Timestamp in nanoseconds
- timestamp_s: Timestamp in seconds
- pos_x, pos_y, pos_z: Position in meters
- quat_w, quat_x, quat_y, quat_z: Orientation quaternion
- vel_x, vel_y, vel_z: Velocity in m/s
- roll, pitch, yaw: Euler angles in radians

### Camera Timestamps (timestamps.csv)
Columns:
- timestamp_ns: Timestamp in nanoseconds
- timestamp_s: Timestamp in seconds

## Coordinate Frames

### World Frame
- X: Forward
- Y: Left
- Z: Up

### Body Frame (IMU/Camera)
- X: Forward
- Y: Left
- Z: Up

## Usage Notes
1. All timestamps are synchronized and start from 0
2. IMU measurements include realistic sensor noise and bias drift
3. Ground truth is provided at IMU rate (200 Hz)
4. Camera images are provided at 20 Hz
5. Stereo baseline is 0.12m between cam0 (left) and cam1 (right)

## Citation
Generated using MOSTLY AI synthetic data platform.
