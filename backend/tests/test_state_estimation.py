import pandas as pd
from .state_estimation import IMUSLAMFusion
import cv2
import time

df = pd.read_csv('IMU_data.csv')
df.drop(columns=['MagX(uT)', 'MagY(uT)', 'MagZ(uT)', 'Roll(deg)', 'Pitch(deg)', 'Yaw(deg)'], inplace=True)
print(df.head())

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
    while(true):
        i = i + 1
        imu = IMNUData(
            ax = df['AccelX(m/s^2)'][i], ay = df['AccelY(m/s^2)'][i], az = df['AccelZ(m/s^2)'][i],
            gx = df['GyroX(deg/s)'][i], gy = df['GyroY(deg/s)'][i], gz = df['GyroZ(deg/s)'][i],
            timestamp = df['TIMESTAMP'][i]
        )
        fusion.predict(imu)
        updated_pose = fusion.update()
        print("updated Orientation: ", updated_pose.qw, updated_pose.qx, updated_pose.qy, updated_pose.qz)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

