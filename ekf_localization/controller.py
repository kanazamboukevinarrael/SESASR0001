# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest import case

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from geometry_msgs.msg import Twist

###################################
#### Multiple steps of sampling ###
###################################
from ekf_localization.my_velocity_task_0 import heart_velocity_model
###################################
#### Multiple steps of sampling ###
###################################


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_command = self.create_publisher(Twist, 'cmd_vel', 10)
        timer_period = 1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        self.x = 0
        self.y = 0

    def timer_callback(self):
        twist = Twist()
        match  self.i % 4:
            case  0:
                self.get_logger().info('Publishing: "%s"' % twist)
                twist.linear.x = 1.0
                twist.angular.z = 0.0
            case  1:
                twist.linear.x = 0.0
                twist.angular.z = 1.0
            case  2:
                twist.linear.x = -1.0
                twist.angular.z = 0.0
            case  3:
                twist.linear.x = 0.0
                twist.angular.z = -1.0
        self.publisher_command.publish(twist)
        
        ###################################
        #### Multiple steps of sampling ###
        ###################################
        self.hearth_pose = heart_velocity_model(
        x=[2, 4, 0],
        u=[0.8, 0],
        a=[0.05, 0.1, 0.05, 0.1, 0.025, 0.025])
        self.get_logger().info(f'Publishing: "hearth_pose: {self.hearth_pose}"')
        
        ###################################
        #### Multiple steps of sampling ###
        ###################################


def main(args=None):
    rclpy.init()

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
