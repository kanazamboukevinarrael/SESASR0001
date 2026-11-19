
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from motion_model import velocity_motion_model


# Initial robot pose
x0 = np.array([0.0, 0.0, 0.0])

# Fixed velocity command
u = np.array([0.2, 0.5])

# Motion duration
dt = 1.0

# Number of samples
N = 500

# Two noise configurations
# Case 1: high angular uncertainty
alpha_angular = [0.001, 0.001, 0.05, 0.05, 0.001, 0.001]

# Case 2: high linear uncertainty
alpha_linear  = [0.02, 0.02, 0.001, 0.001, 0.001, 0.001]


def generate_samples(alpha):
    samples = []

    for _ in range(N):
        pose = velocity_motion_model(x0, u, alpha, dt)
        samples.append(pose)

    return np.array(samples)


# Generate samples
samples_angular = generate_samples(alpha_angular)
samples_linear = generate_samples(alpha_linear)


# Plot results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cases = [
    (samples_angular, "High angular uncertainty"),
    (samples_linear, "High linear uncertainty")
]

for ax, (samples, title) in zip(axes, cases):

    ax.scatter(
        samples[:, 0],
        samples[:, 1],
        s=12,
        alpha=0.4,
        label="Sampled poses"
    )

    ax.scatter(
        x0[0],
        x0[1],
        marker="s",
        color="red",
        s=80,
        label="Initial pose"
    )

    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()

plt.tight_layout()
# Cartella plots nella directory principale del package
output_dir = Path(__file__).resolve().parents[1] / "plots"
output_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(output_dir / "motion_samples.png", dpi=300)
print("Grafico salvato con successo in:", output_dir / "motion_samples.png")
plt.show()

