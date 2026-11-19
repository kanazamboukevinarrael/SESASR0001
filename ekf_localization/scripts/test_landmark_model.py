import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from landmark_model import (
    expected_measurement,
    sample_pose_from_measurement,
    landmark_jacobian,
)

np.random.seed(0)

N = 1000
m = np.array([0.0, 0.0])              # landmark 22 of landmarks.yaml
x0 = np.array([-1.5, -0.5, 0.3])      # example initial pose [x, y, theta]

# (std_range [m], std_bearing [rad])
noise_sets = [
    ("High range uncertainty", (0.20, np.deg2rad(1.0))),
    ("High bearing uncertainty", (0.01, np.deg2rad(15.0))),
]

# Measurement obtained from the initial pose (noise-free)
z = expected_measurement(x0, m)
print("z = [range, bearing] =", z)


def numeric_jacobian(f, s, eps=1e-6):
    cols = []
    for i in range(len(s)):
        d = np.zeros(len(s))
        d[i] = eps
        cols.append((f(s + d) - f(s - d)) / (2 * eps))
    return np.array(cols).T


H = landmark_jacobian(x0, m)
H_num = numeric_jacobian(lambda s: expected_measurement(s, m), x0)
print("max |H - H_numeric| =", np.abs(H - H_num).max())

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (title, std) in zip(axes, noise_sets):
    poses = np.array(
        [sample_pose_from_measurement(z, m, std) for _ in range(N)]
    )

    ax.scatter(poses[:, 0], poses[:, 1], s=8, alpha=0.3, label="Sampled poses")
    # Heading of a subset of samples
    sub = poses[:100]
    ax.quiver(sub[:, 0], sub[:, 1], np.cos(sub[:, 2]), np.sin(sub[:, 2]),
              angles="xy", scale_units="xy", scale=8, width=0.003,
              color="tab:blue", alpha=0.6)
    ax.scatter(m[0], m[1], marker="^", color="black", s=100, label="Landmark")
    ax.scatter(x0[0], x0[1], marker="s", color="red", s=80, label="Initial pose")
    ax.quiver(x0[0], x0[1], np.cos(x0[2]), np.sin(x0[2]),
              angles="xy", scale_units="xy", scale=4, color="red")

    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()

plt.tight_layout()
output_dir = Path(__file__).resolve().parents[1] / "plots"
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / "landmark_samples.png", dpi=300)
print("Saved:", output_dir / "landmark_samples.png")
plt.show()