#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def main():
    parser = argparse.ArgumentParser(
        description="Sample Ito diffusion dX = v*dt + sqrt(2*D*dt)*W and plot f(v) and f(v)*v^2 (v>=0)."
    )
    parser.add_argument("--v", type=float, default=0.0, help="Drift velocity v (default: 0.0)")
    parser.add_argument("--D", type=float, default=1.0, help="Diffusion coefficient D (default: 1.0)")
    parser.add_argument("--dt", type=float, default=0.1, help="Time step dt (default: 0.1)")
    parser.add_argument("--n", type=int, default=10000, help="Number of particles (default: 10000)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    parser.add_argument("--bins", type=int, default=60, help="Histogram bins (default: 60)")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting (print stats only)")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    v, D, dt, n = args.v, args.D, args.dt, args.n

    if dt <= 0:
        raise ValueError("dt must be positive.")
    if D < 0:
        raise ValueError("D must be nonnegative.")
    if n <= 1:
        raise ValueError("n must be > 1.")

    # --- Sample Ito increment and velocities ---
    W = np.random.normal(loc=0.0, scale=1.0, size=n)
    dX = v * dt + np.sqrt(2.0 * D * dt) * W
    velocities = dX / dt  # V = v + sqrt(2D/dt) * N(0,1)

    # Stats
    emp_mean = velocities.mean()
    emp_std = velocities.std(ddof=1)
    theo_mean = v
    theo_std = np.sqrt(2.0 * D / dt)

    print("=== Velocity distribution (V = dX/dt) ===")
    print(f"n = {n}, v = {v}, D = {D}, dt = {dt}")
    print(f"Empirical mean: {emp_mean:.6f}")
    print(f"Empirical std:  {emp_std:.6f}")
    print(f"Theoretical mean: {theo_mean:.6f}")
    print(f"Theoretical std:  {theo_std:.6f}")

    if args.no_plot:
        return

    # --- Figure with two panels ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ---------------------------
    # Left: f(v) as histogram PDF
    # ---------------------------
    # Use density=True so the bar heights approximate the unconditional PDF f(v)
    n_hist, bins_hist, _ = axes[0].hist(
        velocities, bins=args.bins, density=True, alpha=0.5, edgecolor="black", label="Empirical PDF (hist)"
    )
    bin_centers = 0.5 * (bins_hist[1:] + bins_hist[:-1])

    # Theoretical Normal PDF overlay
    x = np.linspace(velocities.min(), velocities.max(), 1000)
    axes[0].plot(x, norm.pdf(x, loc=theo_mean, scale=theo_std), lw=2, label="Theoretical PDF")

    axes[0].set_title("Velocity PDF  f(v)")
    axes[0].set_xlabel("Velocity v")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # --------------------------------------------------------
    # Right: velocity-weighted distribution f(v) * v^2 (v ≥ 0)
    # --------------------------------------------------------
    # IMPORTANT: f(v) is the original (unconditional) PDF.
    # We therefore compute the PDF histogram over ALL velocities (already done above),
    # then weight the positive-velocity bins by v^2 and plot bars for v >= 0.

    # Reuse density histogram from ALL velocities:
    fv = n_hist  # density estimate per bin
    widths = np.diff(bins_hist)

    # Compute f(v)*v^2 at bin centers; zero-out for v < 0
    fv_times_v2 = fv * (bin_centers ** 2)
    pos_mask = bin_centers >= 0

    # Draw as a histogram-like bar plot (numerical sample displayed as bars)
    axes[1].bar(
        bin_centers[pos_mask],
        fv_times_v2[pos_mask],
        width=widths[pos_mask],
        alpha=0.6,
        edgecolor="black",
        label="Empirical f(v)·v² (bars)"
    )

    # Theoretical curve for comparison (not sampled; just a line)
    x_pos = np.linspace(0.0, max(velocities.max(), 1e-6), 600)
    f_theo = norm.pdf(x_pos, loc=theo_mean, scale=theo_std)
    axes[1].plot(x_pos, f_theo * (x_pos ** 2), lw=2, label="Theoretical f(v)·v²")

    axes[1].set_title("Velocity-weighted distribution  f(v)·v²  (v ≥ 0)")
    axes[1].set_xlabel("Velocity v")
    axes[1].set_ylabel("f(v)·v²")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
