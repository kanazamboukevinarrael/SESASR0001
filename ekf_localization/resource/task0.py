"""
Lab 04-05-06, Task 0: probabilistic models (plain Python, no ROS).

  1. Velocity motion model (sampling)  + Jacobians G, V      (sympy)
  2. Landmark measurement model h(x, m, sigma) + Jacobian H  (sympy)
  3. Inverse landmark model p(x | z, m): candidate robot poses from one measurement

Run:   python3 task0.py            (saves plots/ and plots/jacobians.txt)
       python3 task0.py --show     (also opens the figures)
"""
import sys
from pathlib import Path

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


def wrap_angle(a):
    """Wrap an angle (or array of angles) to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


# =====================================================================
# 1. VELOCITY MOTION MODEL
# =====================================================================
def motion_model(x, u, dt):
    """Noise-free velocity motion model g(u, x). x = [x, y, theta], u = [v, w]."""
    v, w = u
    th = x[2]
    if abs(w) > 1e-9:
        r = v / w
        xp = x[0] - r * np.sin(th) + r * np.sin(th + w * dt)
        yp = x[1] + r * np.cos(th) - r * np.cos(th + w * dt)
    else:  # straight-line limit (avoids division by zero)
        xp = x[0] + v * dt * np.cos(th)
        yp = x[1] + v * dt * np.sin(th)
    return np.array([xp, yp, th + w * dt])


def sample_velocity_motion_model(x, u, a, dt):
    """
    Sample the next pose x' ~ p(x' | x, u).
      x  -- current pose [x, y, theta]
      u  -- command [v, w]
      a  -- noise coefficients [a1, a2, a3, a4]  (VARIANCE coefficients):
              var(v_hat) = a1*v^2 + a2*w^2
              var(w_hat) = a3*v^2 + a4*w^2
      dt -- time step
    Returns a numpy array [x', y', theta'].
    """
    v, w = u
    v_hat = v + np.random.normal(0, np.sqrt(a[0] * v**2 + a[1] * w**2))
    w_hat = w + np.random.normal(0, np.sqrt(a[2] * v**2 + a[3] * w**2))
    return motion_model(x, [v_hat, w_hat], dt)


def velocity_motion_jacobians():
    """Symbolic g, G = dg/dx (3x3), V = dg/du (3x2). Returns sympy objects."""
    x, y, th, v, w, dt = sp.symbols("x y theta v omega dt", real=True)
    g = sp.Matrix([
        x - v / w * sp.sin(th) + v / w * sp.sin(th + w * dt),
        y + v / w * sp.cos(th) - v / w * sp.cos(th + w * dt),
        th + w * dt,
    ])
    state, cmd = sp.Matrix([x, y, th]), sp.Matrix([v, w])
    G = g.jacobian(state).applyfunc(sp.simplify)
    V = g.jacobian(cmd).applyfunc(sp.simplify)
    return g, G, V, (x, y, th, v, w, dt)


# =====================================================================
# 2. LANDMARK MEASUREMENT MODEL
# =====================================================================
def expected_measurement(x, m):
    """Noise-free h(x, m) = [range, bearing]."""
    dx, dy = m[0] - x[0], m[1] - x[1]
    return np.array([np.hypot(dx, dy), wrap_angle(np.arctan2(dy, dx) - x[2])])


def landmark_model(x, m, sigma):
    """
    Sampling form of the sensor model h(x, m, sigma).
      x     -- robot pose [x, y, theta]
      m     -- landmark position [mx, my]
      sigma -- [sigma_r, sigma_phi] (standard deviations)
    Returns a noisy [range, bearing] as a numpy array.
    """
    z = expected_measurement(x, m)
    return np.array([z[0] + np.random.normal(0, sigma[0]),
                     wrap_angle(z[1] + np.random.normal(0, sigma[1]))])


def landmark_jacobian_sympy():
    """Symbolic h and H = dh/dx (2x3)."""
    x, y, th, mx, my = sp.symbols("x y theta m_x m_y", real=True)
    h = sp.Matrix([
        sp.sqrt((mx - x) ** 2 + (my - y) ** 2),
        sp.atan2(my - y, mx - x) - th,
    ])
    H = h.jacobian(sp.Matrix([x, y, th])).applyfunc(sp.simplify)
    return h, H, (x, y, th, mx, my)


# =====================================================================
# 3. INVERSE LANDMARK MODEL
# =====================================================================
def sample_pose_from_measurement(z, m):
    """
    Candidate robot pose consistent with ONE measurement z = [range, bearing]
    of a landmark at m. The direction from the robot to the landmark (psi) is
    unknown, so it is drawn uniformly: the candidates lie on a ring around m.
        position = m - r * [cos(psi), sin(psi)]
        theta    = psi - bearing
    """
    r, phi = z
    psi = np.random.uniform(-np.pi, np.pi)
    return np.array([m[0] - r * np.cos(psi), m[1] - r * np.sin(psi), wrap_angle(psi - phi)])


# =====================================================================
# Helpers: numeric checks, plots
# =====================================================================
def numeric_jacobian(f, s, eps=1e-6):
    cols = []
    for i in range(len(s)):
        d = np.zeros(len(s))
        d[i] = eps
        cols.append((f(s + d) - f(s - d)) / (2 * eps))
    return np.array(cols).T


def plot_motion_samples(x0, u, dt, noise_sets, n, outdir):
    # stacked panels, shared axes: same scale for both noise sets, wide enough for equal aspect
    fig, axes = plt.subplots(len(noise_sets), 1, figsize=(9, 3.6 * len(noise_sets)),
                             sharex=True, sharey=True)
    for ax, (title, a) in zip(np.atleast_1d(axes), noise_sets.items()):
        S = np.array([sample_velocity_motion_model(x0, u, a, dt) for _ in range(n)])
        ax.scatter(S[:, 0], S[:, 1], s=6, c="tab:red", alpha=0.5, label=f"{n} sampled poses")
        sub = S[:100]
        ax.quiver(sub[:, 0], sub[:, 1], np.cos(sub[:, 2]), np.sin(sub[:, 2]),
                  angles="xy", scale_units="xy", scale=25, width=0.002, color="gray")
        ax.plot(*S.mean(0)[:2], "ko", markersize=5, label="mean of samples")
        ax.plot(*x0[:2], "b*", markersize=16, label="current pose")
        ax.quiver(x0[0], x0[1], np.cos(x0[2]), np.sin(x0[2]),
                  angles="xy", scale_units="xy", scale=8, color="b")
        ax.set_title(f"{title}\na = {a}", fontsize=9)
        ax.set_ylabel("y [m]")
        ax.grid(True)
        ax.set_aspect("equal")
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
        print(f"[motion] {title}: std x,y,theta = {S.std(0).round(3)}")
    np.atleast_1d(axes)[-1].set_xlabel("x [m]")
    fig.suptitle(f"Velocity motion model: x0={x0.tolist()}, u={list(u)}, dt={dt}")
    fig.savefig(outdir / "motion_samples.png", dpi=200, bbox_inches="tight")
    return fig


def plot_landmark_samples(x0, m, noise_sets, n, outdir):
    z_true = expected_measurement(x0, m)
    print("[landmark] true z = [range, bearing] =", z_true.round(4))
    fig, axes = plt.subplots(1, len(noise_sets), figsize=(6.5 * len(noise_sets), 6))
    for ax, (title, sigma) in zip(np.atleast_1d(axes), noise_sets):
        # simulate n noisy sensor outputs z = z_true + noise, then invert each one
        poses = []
        for _ in range(n):
            z = np.array([z_true[0] + np.random.normal(0, sigma[0]),
                          z_true[1] + np.random.normal(0, sigma[1])])
            poses.append(sample_pose_from_measurement(z, m))
        poses = np.array(poses)
        ax.scatter(poses[:, 0], poses[:, 1], s=8, alpha=0.3, label="Sampled poses")
        sub = poses[:100]
        ax.quiver(sub[:, 0], sub[:, 1], np.cos(sub[:, 2]), np.sin(sub[:, 2]),
                  angles="xy", scale_units="xy", scale=8, width=0.003, color="tab:blue", alpha=0.6)
        ax.scatter(*m, marker="^", color="black", s=100, label="Landmark")
        ax.scatter(*x0[:2], marker="s", color="red", s=80, label="Initial pose")
        ax.quiver(x0[0], x0[1], np.cos(x0[2]), np.sin(x0[2]),
                  angles="xy", scale_units="xy", scale=4, color="red")
        ax.set_title(f"{title}\nsigma_r={sigma[0]} m, sigma_phi={np.rad2deg(sigma[1]):.1f} deg", fontsize=10)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")
        ax.grid(True)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8)
    fig.savefig(outdir / "landmark_samples.png", dpi=200, bbox_inches="tight")
    return fig


def jacobian_report(outdir):
    """Print and save G, V, H (symbolic + linearized form) and check them numerically."""
    lines = []
    g, G, V, (x, y, th, v, w, dt) = velocity_motion_jacobians()
    h, H, (hx, hy, hth, mx, my) = landmark_jacobian_sympy()

    lines += ["MOTION MODEL  g(u, x), state [x, y, theta], command [v, omega]", sp.latex(g), "",
              "G = dg/dx", sp.latex(G), "", "V = dg/du", sp.latex(V), "",
              "Linearization:  g(u, x) ~ g(u, mu) + G (x - mu)   (also w.r.t. u with V)", "",
              "LANDMARK MODEL  h(x, m)", sp.latex(h), "",
              "H = dh/dx", sp.latex(H), "",
              "Linearization:  h(x, m) ~ h(mu, m) + H (x - mu)"]
    (outdir / "jacobians.txt").write_text("\n".join(lines))

    print("\n--- G ---");  sp.pprint(G)
    print("\n--- V ---");  sp.pprint(V)
    print("\n--- H ---");  sp.pprint(H)

    # numeric checks against finite differences
    G_f = sp.lambdify((x, y, th, v, w, dt), G, "numpy")
    V_f = sp.lambdify((x, y, th, v, w, dt), V, "numpy")
    H_f = sp.lambdify((hx, hy, hth, mx, my), H, "numpy")
    xs, us, dts, m0 = np.array([2.0, 4.0, 0.3]), np.array([0.8, 0.6]), 0.5, np.array([0.0, 0.0])
    G_n = numeric_jacobian(lambda s: motion_model(s, us, dts), xs)
    V_n = numeric_jacobian(lambda s: motion_model(xs, s, dts), us)
    H_n = numeric_jacobian(lambda s: expected_measurement(s, m0), xs)
    print("\nmax|G - G_num| =", np.abs(np.array(G_f(*xs, *us, dts), float) - G_n).max())
    print("max|V - V_num| =", np.abs(np.array(V_f(*xs, *us, dts), float) - V_n).max())
    print("max|H - H_num| =", np.abs(np.array(H_f(*xs, *m0), float) - H_n).max())


# =====================================================================
def main():
    np.random.seed(0)
    outdir = Path(__file__).resolve().parent / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- 1. motion model: 500 samples, ALL from the same x0, two noise sets ----
    x0 = np.array([2.0, 4.0, 0.0])
    u = [0.8, 0.6]
    dt = 0.5
    motion_noise = {
        "Linear noise dominates":  [0.5, 0.5, 0.001, 0.001],
        "Angular noise dominates": [0.001, 0.001, 0.2, 0.2],
    }
    plot_motion_samples(x0, u, dt, motion_noise, n=500, outdir=outdir)

    # ---- 2/3. landmark model: forward sample + inverse model, 1000 samples ----
    m = np.array([0.0, 0.0])                 # landmark 22
    xr = np.array([-1.5, -0.5, 0.3])         # true robot pose
    print("[landmark] one noisy z from h(x, m, sigma):", landmark_model(xr, m, [0.1, 0.02]).round(4))
    landmark_noise = [
        ("High range uncertainty", (0.20, np.deg2rad(1.0))),
        ("High bearing uncertainty", (0.01, np.deg2rad(15.0))),
    ]
    plot_landmark_samples(xr, m, landmark_noise, n=1000, outdir=outdir)

    # ---- Jacobians ----
    jacobian_report(outdir)

    print("\nSaved plots and jacobians.txt in", outdir)
    if "--show" in sys.argv:
        plt.show()


if __name__ == "__main__":
    main()
