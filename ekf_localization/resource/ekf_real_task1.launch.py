"""Task 3, option A: Task 1 EKF (state [x, y, theta]) on the real robot."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(get_package_share_directory('ekf_localization'),
                          'config', 'ekf_real_task1.yaml')
    return LaunchDescription([
        Node(
            package='ekf_localization',
            executable='ekf_node',
            name='ekf_node',
            output='screen',
            parameters=[params],
            # the real camera publishes on /camera/landmarks
            remappings=[('/landmarks', '/camera/landmarks')],
        ),
    ])
