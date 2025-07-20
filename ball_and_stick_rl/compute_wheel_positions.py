import numpy as np
from scipy.spatial.transform import Rotation as R


def transform_pose(x, y, z, roll, pitch, yaw, y_d, z_d):
    """
    Transform a 3D pose by rotating y_d degrees about y-axis and z_d degrees about z-axis.

    Args:
        x, y, z: Position coordinates (meters)
        roll, pitch, yaw: Orientation in degrees (RPY)
        y_d, z_d: Rotation angles in degrees

    Returns:
        tuple: Transformed (x, y, z, roll, pitch, yaw) with RPY in degrees
    """
    # Convert degrees to radians for input RPY
    roll_rad = np.deg2rad(roll)
    pitch_rad = np.deg2rad(pitch)
    yaw_rad = np.deg2rad(yaw)

    # Convert degrees to radians for rotation angles
    y_d_rad = np.deg2rad(y_d)
    z_d_rad = np.deg2rad(z_d)

    # Original position vector
    pos = np.array([x, y, z])

    # Rotation matrix for y-axis rotation
    Ry = np.array(
        [
            [np.cos(y_d_rad), 0, np.sin(y_d_rad)],
            [0, 1, 0],
            [-np.sin(y_d_rad), 0, np.cos(y_d_rad)],
        ]
    )

    # Rotation matrix for z-axis rotation
    Rz = np.array(
        [
            [np.cos(z_d_rad), -np.sin(z_d_rad), 0],
            [np.sin(z_d_rad), np.cos(z_d_rad), 0],
            [0, 0, 1],
        ]
    )

    # Combined rotation matrix (Rz * Ry)
    R_combined = np.dot(Rz, Ry)

    # Transform position
    pos_transformed = np.dot(R_combined, pos)

    # Convert input RPY to rotation matrix
    R_orig = R.from_euler("xyz", [roll, pitch, yaw], degrees=True).as_matrix()

    # Apply combined rotation to the original orientation
    R_new = np.dot(R_combined, R_orig)

    # Convert resulting rotation matrix back to Euler angles (RPY)
    r = R.from_matrix(R_new)
    roll_new, pitch_new, yaw_new = r.as_euler("xyz", degrees=True)

    return (
        pos_transformed[0],
        pos_transformed[1],
        pos_transformed[2],
        roll_new,
        pitch_new,
        yaw_new,
    )


# Example usage
if __name__ == "__main__":
    ball_radius = 0.15
    wheel_radius = 0.03

    wheel_1_pose = transform_pose(
        x=0, y=0, z=-0.18, roll=0.0, pitch=-90.0, yaw=0.0, y_d=45.0, z_d=0
    )
    print(f"wheel_1_pose: {wheel_1_pose}")

    wheel_2_pose = transform_pose(
        x=0, y=0, z=0.18, roll=0.0, pitch=-90.0, yaw=0.0, y_d=-45.0, z_d=-120
    )
    print(f"wheel_2_pose: {wheel_2_pose}")

    wheel_3_pose = transform_pose(
        x=0, y=0, z=0.18, roll=0.0, pitch=-90.0, yaw=0.0, y_d=45.0, z_d=-240
    )
    print(f"wheel_3_pose: {wheel_3_pose}")
