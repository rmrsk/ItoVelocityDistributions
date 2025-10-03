#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def main():
    parser = argparse.ArgumentParser(
        description="Sample Ito diffusion dX = v*dt + sqrt(2*D*dt)*W and compute velocity distribution."
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

    # Sample the increment dX for each particle
    W = np.random.normal(loc=0.0, scale=1.0, size=n)
    dX = v * dt + np.sqrt(2.0 * D * dt) * W

    # Instantaneous velocities over the step
    velocities = dX / dt  # = v + sqrt(2D/dt) * N(0,1)

    # Empirical statistics
    emp_mean = velocities.mean()
    emp_std = velocities.std(ddof=1)

    # Theoretical distribution parameters
    theo_mean = v
    theo_std = np.sqrt(2.0 * D / dt)

    print("=== Velocity distribution (V = dX/dt) ===")
    print(f"n = {n}, v = {v}, D = {D}, dt = {dt}")
    print(f"Empirical mean: {emp_mean:.6f}")
    print(f"Empirical std:  {emp_std:.6f}")
    print(f"Theoretical mean: {theo_mean:.6f}")
    print(f"Theoretical std:  {theo_std:.6f}")

    if not args.no_plot:
        # Histogram
        fig, ax = plt.subplots()
        ax.hist(velocities, bins=args.bins, density=True, alpha=0.6, edgecolor='black')
        # Overlay theoretical Normal PDF
        x_min = min(velocities.min(), theo_mean - 5 * theo_std)
        x_max = max(velocities.max(), theo_mean + 5 * theo_std)
        x = np.linspace(x_min, x_max, 1000)
        ax.plot(x, norm.pdf(x, loc=theo_mean, scale=theo_std), linewidth=2)

        ax.set_title("Velocity distribution from Ito step")
        ax.set_xlabel("Velocity")
        ax.set_ylabel("Probability density")
        ax.legend(["Theoretical Normal PDF", "Empirical histogram"])
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
