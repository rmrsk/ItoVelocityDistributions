#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def folded_normal_pdf(r, mu, sigma):
    """
    PDF of R = |X| for X ~ N(mu, sigma^2), i.e., the folded normal.
    For r >= 0:
        f_R(r) = (1/sigma) * [phi((r - mu)/sigma) + phi((r + mu)/sigma)]
    where phi is the standard normal PDF.
    """
    r = np.asarray(r)
    pdf = np.zeros_like(r, dtype=float)
    mask = r >= 0
    z1 = (r[mask] - mu) / sigma
    z2 = (r[mask] + mu) / sigma
    pdf[mask] = (norm.pdf(z1) + norm.pdf(z2)) / sigma
    return pdf


def main():
    parser = argparse.ArgumentParser(
        description="1D Ito diffusion: sample velocities and plot velocity and speed (magnitude) PDFs."
    )
    parser.add_argument("--v",  type=float, default=0.0, help="Drift velocity v (default: 0.0)")
    parser.add_argument("--D",  type=float, default=1.0, help="Diffusion coefficient D (default: 1.0)")
    parser.add_argument("--dt", type=float, default=0.1, help="Time step dt (default: 0.1)")
    parser.add_argument("--n",  type=int,   default=10000, help="Number of particles (default: 10000)")
    parser.add_argument("--bins", type=int, default=60, help="Histogram bins (default: 60)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting (print stats only)")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    v, D, dt, n = args.v, args.D, args.dt, args.n
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if D < 0:
        raise ValueError("D must be nonnegative.")
    if n < 2:
        raise ValueError("n must be >= 2.")

    # Component std of velocities
    sigma = np.sqrt(2.0 * D / dt)

    # --- Sample 1D velocities ---
    W = np.random.normal(size=n)
    V = v + sigma * W           # velocities
    R = np.abs(V)               # speeds (magnitudes)

    # --- Basic stats ---
    emp_mean = V.mean()
    emp_std = V.std(ddof=1)
    emp_speed_mean = R.mean()
    emp_speed_std = R.std(ddof=1)

    print("=== 1D Velocity sampling (V = dX/dt) ===")
    print(f"n = {n}, D = {D}, dt = {dt}")
    print(f"Drift v = {v}")
    print(f"Component theoretical std sigma = sqrt(2D/dt) = {sigma:.6f}")
    print(f"Empirical velocity mean: {emp_mean:.6f}")
    print(f"Empirical velocity std:  {emp_std:.6f}")
    print(f"Empirical speed mean:    {emp_speed_mean:.6f}")
    print(f"Empirical speed std:     {emp_speed_std:.6f}")

    if args.no_plot:
        return

    # --- Plots ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: velocity PDF f(v)
    ax_v = axes[0]
    ax_v.hist(V, bins=args.bins, density=True, alpha=0.6, edgecolor="black",
              label="Empirical velocity PDF (hist)")
    x = np.linspace(V.min(), V.max(), 800)
    ax_v.plot(x, norm.pdf(x, loc=v, scale=sigma), lw=2, label=f"Theoretical N({v:.3g}, {sigma:.3g}²)")
    ax_v.set_title("Velocity PDF  f(v)")
    ax_v.set_xlabel("v")
    ax_v.set_ylabel("Density")
    ax_v.legend()

    # Right: speed (magnitude) PDF g(r) for r = |V|
    ax_r = axes[1]
    ax_r.hist(R, bins=args.bins, density=True, alpha=0.6, edgecolor="black",
              label="Empirical |v| PDF (hist)")
    r_grid = np.linspace(0.0, R.max(), 800)
    ax_r.plot(r_grid, folded_normal_pdf(r_grid, mu=v, sigma=sigma), lw=2,
              label="Theoretical folded normal PDF")
    ax_r.set_title("Speed (|v|) PDF — normalized over magnitude (1D)")
    ax_r.set_xlabel("|v|")
    ax_r.set_ylabel("Density")
    ax_r.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
