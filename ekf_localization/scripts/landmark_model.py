import numpy as np


def normalize_angle(angle):
    """Wrap an angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def expected_measurement(x, m):
    """Noise-free range and bearing of landmark m seen from pose x: h(x)."""
    dx, dy = m[0] - x[0], m[1] - x[1]
    return np.array([np.hypot(dx, dy),
                     normalize_angle(np.arctan2(dy, dx) - x[2])])


def sample_landmark_measurement(x, m, std):
    """Sample a noisy [range, bearing] of landmark m seen from pose x.

    std = (std_range [m], std_bearing [rad]).
    """
    std_r, std_phi = std
    dx, dy = m[0] - x[0], m[1] - x[1]
    r = np.hypot(dx, dy) + np.random.normal(0.0, std_r)
    phi = normalize_angle(
        np.arctan2(dy, dx) - x[2] + np.random.normal(0.0, std_phi)
    )
    return np.array([r, phi])


def sample_pose_from_measurement(z, m, std):
    """Sample a pose [x, y, theta] compatible with z = [r, phi] of landmark m."""
    r, phi = z
    std_r, std_phi = std
    r_hat = r + np.random.normal(0.0, std_r)
    phi_hat = phi + np.random.normal(0.0, std_phi)
    gamma = np.random.uniform(0.0, 2 * np.pi)   # direction robot -> landmark
    x = m[0] - r_hat * np.cos(gamma)
    y = m[1] - r_hat * np.sin(gamma)
    theta = normalize_angle(gamma - phi_hat)
    return np.array([x, y, theta])


def landmark_jacobian(x, m):
    """Jacobian H (2x3) of h(x) with respect to the state [x, y, theta]."""
    dx, dy = m[0] - x[0], m[1] - x[1]
    q = dx ** 2 + dy ** 2
    r = np.sqrt(q)
    return np.array([[-dx / r, -dy / r, 0.0],
                     [dy / q, -dx / q, -1.0]])