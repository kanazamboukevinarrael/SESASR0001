"""Task 3, option B: Task 2 EKF (state [x, y, theta, v, omega], fusion) on the real robot."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(get_package_share_directory('ekf_localization'),
                          'config', 'ekf_real_task2.yaml')
    return LaunchDescription([
        Node(
            package='ekf_localization',
            executable='ekf_node_ext',
            name='ekf_node_ext',
            output='screen',
            parameters=[params],   # landmarks_topic and imu_topic are set in the yaml
        ),
    ])
