"""Task 2: EKF with state mu = [x, y, theta, v, omega] and sensor fusion.

- timer (20 Hz)  -> prediction with g5 (velocities are in the state, no command input)
- /odom          -> update v, omega        (ht_odom)
- /imu           -> update omega only      (ht_imu)
- /landmarks     -> update pose, per landmark, then publish /ekf
"""
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
from sensor_msgs.msg import Imu

from ekf_localization.ekf import RobotEKF
from ekf_localization.ekf_models import residual, normalize_angle
from ekf_localization.ekf_models_ext import (
    g5, G5, V5, landmark_h5, landmark_H5, odom_h, odom_H, imu_h, imu_H,
)


class EKFNodeExt(Node):

    def __init__(self):
        super().__init__('ekf_node_ext')

        # ---------------- parameters ----------------
        self.declare_parameter('ekf_frequency', 20.0)
        self.declare_parameter('initial_pose', [-2.0, -0.5, 0.0])      # x, y, yaw
        self.declare_parameter('initial_std', [0.1, 0.1, 0.1])         # x, y, yaw
        self.declare_parameter('initial_std_vel', [0.1, 0.1])          # v, omega
        self.declare_parameter('std_lin_acc', 1.0)                     # [m/s^2]   process noise on v
        self.declare_parameter('std_ang_acc', 3.0)                     # [rad/s^2] process noise on omega
        self.declare_parameter('std_range', 0.05)                      # [m]
        self.declare_parameter('std_bearing', math.radians(2.0))       # [rad]
        self.declare_parameter('std_odom', [0.05, 0.10])               # v [m/s], omega [rad/s]
        self.declare_parameter('std_imu', 0.02)                        # omega [rad/s] (gyro is more precise)
        self.declare_parameter('imu_topic', '/imu')
        self.declare_parameter('landmarks_topic', '/landmarks')        # '/camera/landmarks' on the real robot
        self.declare_parameter('landmarks_file', '')
        self.declare_parameter('frame_id', 'map')

        freq = float(self.get_parameter('ekf_frequency').value)
        self.dt = 1.0 / freq

        # ---------------- landmark map ----------------
        path = self.get_parameter('landmarks_file').value or os.path.join(
            get_package_share_directory('turtlebot3_perception'),
            'config', 'landmarks.yaml')
        with open(path) as f:
            lm = yaml.safe_load(f)['landmarks']
        self.landmarks = {int(i): np.array([x, y], float)
                          for i, x, y in zip(lm['id'], lm['x'], lm['y'])}

        # ---------------- EKF ----------------
        self.ekf = RobotEKF(dim_x=5, dim_u=2, eval_gux=g5, eval_Gt=G5, eval_Vt=V5)
        p0 = np.array(self.get_parameter('initial_pose').value, float)
        self.ekf.mu = np.concatenate([p0, [0.0, 0.0]])                 # v = omega = 0 at start
        std0 = list(self.get_parameter('initial_std').value) + \
            list(self.get_parameter('initial_std_vel').value)
        self.ekf.Sigma = np.diag(np.square(std0))

        # process noise: random walk on v and omega (acceleration noise integrated over dt)
        sa_v = float(self.get_parameter('std_lin_acc').value)
        sa_w = float(self.get_parameter('std_ang_acc').value)
        self.ekf.Mt = np.diag([(sa_v * self.dt) ** 2, (sa_w * self.dt) ** 2])

        # measurement noise
        std_r = float(self.get_parameter('std_range').value)
        std_b = float(self.get_parameter('std_bearing').value)
        self.Q_landmark = np.diag([std_r ** 2, std_b ** 2])
        std_odom = np.array(self.get_parameter('std_odom').value, float)
        self.Q_odom = np.diag(std_odom ** 2)
        std_imu = float(self.get_parameter('std_imu').value)
        self.Q_imu = np.array([[std_imu ** 2]])

        # the prediction has no command: velocities come from the state
        self.no_cmd = np.zeros(0)
        self.no_noise = np.zeros(2)

        # ---------------- interfaces ----------------
        self.create_subscription(Odometry, '/odom', self.odom_callback, qos_profile_sensor_data)
        self.create_subscription(Imu, self.get_parameter('imu_topic').value,
                                 self.imu_callback, qos_profile_sensor_data)
        self.create_subscription(LandmarkArray, self.get_parameter('landmarks_topic').value,
                                 self.landmarks_callback, qos_profile_sensor_data)
        self.ekf_pub = self.create_publisher(Odometry, '/ekf', 10)
        self.create_timer(self.dt, self.prediction_step)

        self.get_logger().info('EKF (extended state, sensor fusion) started')

    # ---------------- prediction ----------------
    def prediction_step(self):
        self.ekf.predict(u=self.no_cmd, sigma_u=self.no_noise, g_extra_args=(self.dt,))
        self.ekf.mu[2] = normalize_angle(self.ekf.mu[2])

    # ---------------- updates ----------------
    def odom_callback(self, msg):
        z = np.array([msg.twist.twist.linear.x, msg.twist.twist.angular.z])
        self.ekf.update(z, eval_hx=odom_h, eval_Ht=odom_H, Qt=self.Q_odom,
                        Ht_args=(self.ekf.mu,), hx_args=(self.ekf.mu,))   # plain subtraction, no angle

    def imu_callback(self, msg):
        z = np.array([msg.angular_velocity.z])
        self.ekf.update(z, eval_hx=imu_h, eval_Ht=imu_H, Qt=self.Q_imu,
                        Ht_args=(self.ekf.mu,), hx_args=(self.ekf.mu,))

    def landmarks_callback(self, msg):
        for det in msg.landmarks:
            m = self.landmarks.get(det.id)
            if m is None:
                self.get_logger().warn(f'Unknown landmark id {det.id}', throttle_duration_sec=5.0)
                continue
            z = np.array([det.range, det.bearing])
            self.ekf.update(z, eval_hx=landmark_h5, eval_Ht=landmark_H5, Qt=self.Q_landmark,
                            Ht_args=(self.ekf.mu, m), hx_args=(self.ekf.mu, m),
                            residual=residual, angle_idx=-1)
            self.ekf.mu[2] = normalize_angle(self.ekf.mu[2])
        self.publish_estimate()

    # ---------------- output ----------------
    def publish_estimate(self):
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.get_parameter('frame_id').value
        x, y, th, v, w = self.ekf.mu
        S = self.ekf.Sigma

        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.orientation.z = math.sin(th / 2.0)
        msg.pose.pose.orientation.w = math.cos(th / 2.0)
        cov = np.zeros((6, 6))
        idx = [0, 1, 5]                                   # x, y, yaw
        cov[np.ix_(idx, idx)] = S[:3, :3]
        msg.pose.covariance = cov.flatten().tolist()

        msg.twist.twist.linear.x = float(v)
        msg.twist.twist.angular.z = float(w)
        tcov = np.zeros((6, 6))
        tidx = [0, 5]                                     # v (linear.x), omega (angular.z)
        tcov[np.ix_(tidx, tidx)] = S[3:5, 3:5]
        msg.twist.covariance = tcov.flatten().tolist()

        self.ekf_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = EKFNodeExt()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
