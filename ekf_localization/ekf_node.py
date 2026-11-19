import math
import os

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from landmark_msgs.msg import LandmarkArray
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from ekf_localization.ekf import RobotEKF
from ekf_localization.ekf_models import (
    velocity_g, velocity_Gt, velocity_Vt,
    landmark_h, landmark_H, residual, normalize_angle,
)


class EKFNode(Node):

    def __init__(self):
        super().__init__('ekf_node')

        # Parameters
        self.declare_parameter('ekf_frequency', 20.0)
        self.declare_parameter('initial_pose', [-2.0, -0.5, 0.0])   # x, y, yaw
        self.declare_parameter('initial_std', [0.1, 0.1, 0.1])
        self.declare_parameter('std_lin_vel', 0.1)                   # [m/s]
        self.declare_parameter('std_ang_vel', math.radians(1.0))     # [rad/s]
        self.declare_parameter('std_range', 0.01)                    # [m]
        self.declare_parameter('std_bearing', math.radians(1.0))     # [rad]
        self.declare_parameter('landmarks_file', '')
        self.declare_parameter('frame_id', 'map')

        freq = float(self.get_parameter('ekf_frequency').value)
        self.dt = 1.0 / freq

        # Landmark map from the course YAML file
        path = self.get_parameter('landmarks_file').value or os.path.join(
            get_package_share_directory('turtlebot3_perception'),
            'config', 'landmarks.yaml')
        with open(path) as f:
            lm = yaml.safe_load(f)['landmarks']
        self.landmarks = {int(i): np.array([x, y], float)
                          for i, x, y in zip(lm['id'], lm['x'], lm['y'])}

        # EKF
        self.ekf = RobotEKF(dim_x=3, dim_u=2, eval_gux=velocity_g,
                            eval_Gt=velocity_Gt, eval_Vt=velocity_Vt)
        self.ekf.mu = np.array(self.get_parameter('initial_pose').value, float)
        self.ekf.Sigma = np.diag(
            np.square(self.get_parameter('initial_std').value))
        std_v = float(self.get_parameter('std_lin_vel').value)
        std_w = float(self.get_parameter('std_ang_vel').value)
        self.ekf.Mt = np.diag([std_v ** 2, std_w ** 2])
        std_r = float(self.get_parameter('std_range').value)
        std_phi = float(self.get_parameter('std_bearing').value)
        self.Qt = np.diag([std_r ** 2, std_phi ** 2])

        self.u = np.zeros(2)            # latest [v, omega] from /odom
        self.zero_noise = np.zeros(2)   # deterministic mean prediction

        # Interfaces
        self.create_subscription(Odometry, '/odom', self.odom_callback,
                                 qos_profile_sensor_data)
        self.create_subscription(LandmarkArray, '/landmarks',
                                 self.landmarks_callback,
                                 qos_profile_sensor_data)
        self.ekf_pub = self.create_publisher(Odometry, '/ekf', 10)
        self.create_timer(self.dt, self.prediction_step)

        self.get_logger().info('EKF node started')

    def odom_callback(self, msg):
        self.u = np.array([msg.twist.twist.linear.x,
                           msg.twist.twist.angular.z])

    def prediction_step(self):
        self.ekf.predict(u=self.u, sigma_u=self.zero_noise,
                         g_extra_args=(self.dt,))
        self.ekf.mu[2] = normalize_angle(self.ekf.mu[2])

    def landmarks_callback(self, msg):
        for det in msg.landmarks:
            m = self.landmarks.get(det.id)
            if m is None:
                self.get_logger().warn(f'Unknown landmark id {det.id}',
                                       throttle_duration_sec=5.0)
                continue
            z = np.array([det.range, det.bearing])
            self.ekf.update(
                z, eval_hx=landmark_h, eval_Ht=landmark_H, Qt=self.Qt,
                Ht_args=(self.ekf.mu, m), hx_args=(self.ekf.mu, m),
                residual=residual, angle_idx=-1)
            self.ekf.mu[2] = normalize_angle(self.ekf.mu[2])
        self.publish_estimate()

    def publish_estimate(self):
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.get_parameter('frame_id').value
        x, y, th = self.ekf.mu
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.orientation.z = math.sin(th / 2.0)
        msg.pose.pose.orientation.w = math.cos(th / 2.0)
        cov = np.zeros((6, 6))
        idx = [0, 1, 5]                      # x, y, yaw
        cov[np.ix_(idx, idx)] = self.ekf.Sigma
        msg.pose.covariance = cov.flatten().tolist()
        self.ekf_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = EKFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()