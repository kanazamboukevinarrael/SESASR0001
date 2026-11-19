"""Deterministic models and Jacobians for the EKF (Task 1)."""
import numpy as np

W_MIN = 1e-3   # below this |omega| the straight-line limit is used


def normalize_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def residual(a, b, angle_idx=-1):
    """Difference of two measurements with the angle wrapped to [-pi, pi]."""
    y = np.asarray(a, float) - np.asarray(b, float)
    y[angle_idx] = normalize_angle(y[angle_idx])
    return y


# ---- velocity motion model (mean, no noise) ----
def velocity_g(x, u, a, dt):
    """`a` is ignored: it only matches the signature used by RobotEKF.predict."""
    px, py, th = x
    v, w = u
    if abs(w) < W_MIN:
        return np.array([px + v * dt * np.cos(th),
                         py + v * dt * np.sin(th),
                         th + w * dt])
    r = v / w
    return np.array([px - r * np.sin(th) + r * np.sin(th + w * dt),
                     py + r * np.cos(th) - r * np.cos(th + w * dt),
                     th + w * dt])


def velocity_Gt(x, y, th, v, w, dt):
    if abs(w) < W_MIN:
        return np.array([[1.0, 0.0, -v * dt * np.sin(th)],
                         [0.0, 1.0,  v * dt * np.cos(th)],
                         [0.0, 0.0, 1.0]])
    r = v / w
    return np.array([[1.0, 0.0, -r * np.cos(th) + r * np.cos(th + w * dt)],
                     [0.0, 1.0, -r * np.sin(th) + r * np.sin(th + w * dt)],
                     [0.0, 0.0, 1.0]])


def velocity_Vt(x, y, th, v, w, dt):
    if abs(w) < W_MIN:
        return np.array([[np.cos(th) * dt, -0.5 * v * np.sin(th) * dt ** 2],
                         [np.sin(th) * dt,  0.5 * v * np.cos(th) * dt ** 2],
                         [0.0, dt]])
    s0, c0 = np.sin(th), np.cos(th)
    s1, c1 = np.sin(th + w * dt), np.cos(th + w * dt)
    return np.array([
        [(-s0 + s1) / w, dt * v * c1 / w + v * (s0 - s1) / w ** 2],
        [(c0 - c1) / w,  dt * v * s1 / w - v * (c0 - c1) / w ** 2],
        [0.0, dt],
    ])


# ---- landmark measurement model: [range, bearing] ----
def landmark_h(x, m):
    dx, dy = m[0] - x[0], m[1] - x[1]
    return np.array([np.hypot(dx, dy),
                     normalize_angle(np.arctan2(dy, dx) - x[2])])


def landmark_H(x, m):
    dx, dy = m[0] - x[0], m[1] - x[1]
    q = max(dx ** 2 + dy ** 2, 1e-12)
    r = np.sqrt(q)
    return np.array([[-dx / r, -dy / r, 0.0],
                     [dy / q, -dx / q, -1.0]])