
import numpy as np


def velocity_motion_model(x, u, alpha, dt):
    """
    Probabilistic velocity-based motion model.

    Parameters
    ----------
    x : array [x, y, theta]
        Current robot pose.
    u : array [v, omega]
        Linear and angular velocity command.
    alpha : array [alpha1, ..., alpha6]
        Motion noise parameters.
    dt : float
        Duration of the motion.

    Returns
    -------
    x_new : numpy array [x, y, theta]
        Sampled next robot pose.
    """

    px, py, theta = x
    v, omega = u

    a1, a2, a3, a4, a5, a6 = alpha

    # Standard deviations of the three noise terms
    std_v = np.sqrt(a1 * v**2 + a2 * omega**2)
    std_omega = np.sqrt(a3 * v**2 + a4 * omega**2)
    std_gamma = np.sqrt(a5 * v**2 + a6 * omega**2)

    # Sample noisy velocities
    v_hat = v + np.random.normal(0, std_v)
    omega_hat = omega + np.random.normal(0, std_omega)
    gamma_hat = np.random.normal(0, std_gamma)

    # Straight-line motion: avoid division by zero
    if abs(omega_hat) < 1e-6:
        x_new = px + v_hat * dt * np.cos(theta)
        y_new = py + v_hat * dt * np.sin(theta)
    else:
        x_new = (
            px
            - (v_hat / omega_hat) * np.sin(theta)
            + (v_hat / omega_hat) * np.sin(theta + omega_hat * dt)
        )

        y_new = (
            py
            + (v_hat / omega_hat) * np.cos(theta)
            - (v_hat / omega_hat) * np.cos(theta + omega_hat * dt)
        )

    theta_new = theta + (omega_hat + gamma_hat) * dt

    # Normalize angle in [-pi, pi]
    theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi

    return np.array([x_new, y_new, theta_new])
