import pandas as pd
from state_estimation import IMUSLAMFusion
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
        time = df['TIMESTAMP'].dloc[i]
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

