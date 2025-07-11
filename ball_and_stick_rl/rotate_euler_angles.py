import numpy as np
from scipy.spatial.transform import Rotation as R

# Step 1: Define the rotation sequence: yaw (Z), then pitch (Y)
# Rotations in degrees
yaw_deg = -240     # rotate around Z axis
pitch_deg = -45   # then rotate around Y axis

# Step 2: Create the rotation objects
yaw_rotation = R.from_euler('z', yaw_deg, degrees=True)
pitch_rotation = R.from_euler('y', pitch_deg, degrees=True)

# Step 3: Combine them in the correct order: pitch * yaw
# This applies yaw first, then pitch
combined_rotation = pitch_rotation * yaw_rotation

# Step 4: Get the resulting orientation in RPY (XYZ) format
result_rpy = combined_rotation.as_euler('xyz', degrees=True)

print("Resulting RPY after yaw then pitch:", result_rpy)