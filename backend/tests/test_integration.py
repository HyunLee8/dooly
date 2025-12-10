import sys
from pathlib import Path

backend_src = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(backend_src))

import pandas as pd
from drone_control.state_estimation import IMUSLAMFusion, IMUData, SLAMPose
import cv2
import time
import time

df = pd.read_csv('IMU_data.csv')
df.drop(columns=['MagX(uT)', 'MagY(uT)', 'MagZ(uT)', 'Roll(deg)', 'Pitch(deg)', 'Yaw(deg)'], inplace=True)

"""
    Need to normalize data first and put them into the
    corect format for the fusion algorithm to work.

    IMU DATA -->
    TIMESTAMP,
    AccelX(m/s^2) 
    AccelY(m/s^2) 
    AccelZ(m/s^2)
    GyroX(deg/s)
    GyroY(deg/s)
    GyroZ(deg/s)
"""

def test_imu_slam_fusion():
    fusion = IMUSLAMFusion()
    i = 0
    while True:
        i = i + 1
        imu = IMUData(
            ax = df['AccX(m/s^2)'].iloc[i], ay = df['AccY(m/s^2)'].iloc[i], az = df['AccZ(m/s^2)'].iloc[i],
            gx = df['GyroX(rad/s)'].iloc[i], gy = df['GyroY(rad/s)'].iloc[i], gz = df['GyroZ(rad/s)'].iloc[i],
            timestamp = df['Time(ms)'][i]
        )
        slam = SLAMPose(
            x=0.5, y=0.3, z=1.2,
            qw=1.0, qx=0.0, qy=0.0, qz=0.0,
            tracking_status=True,
            timestamp=time.time()
        )

        fusion.predict(imu)
        updated_pose = fusion.update(slam, slam_status=False)
        print("updated Orientation: ", updated_pose.qw, updated_pose.qx, updated_pose.qy, updated_pose.qz)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    test_imu_slam_fusion()
