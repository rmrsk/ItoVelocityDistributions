#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, ncx2


def speed_pdf_theoretical_2d(r, sigma, vmag):
    """
    Stable theoretical PDF of the speed R = ||V|| in 2D for V ~ N_2(mu, sigma^2 I).
    If T = (R/sigma)^2, then T ~ ncx2(df=2, nc=(|mu|/sigma)^2).
    Change of variables:
        f_R(r) = (2r / sigma^2) * f_ncx2( (r/sigma)^2 ; df=2, nc=(|mu|/sigma)^2 ),  r >= 0
    """
    r = np.asarray(r)
    if sigma == 0.0:
        # Degenerate: no diffusion; all mass at r = |mu|
        pdf = np.zeros_like(r)
        pdf[np.isclose(r, vmag, atol=1e-12)] = np.inf  # Dirac spike (for plotting only)
        return pdf

    t = (r / sigma) ** 2
    lam = (vmag / sigma) ** 2
    return (2.0 * r / (sigma ** 2)) * ncx2.pdf(t, df=2, nc=lam)


def main():
    parser = argparse.ArgumentParser(
        description="2D Ito diffusion: sample velocities and plot component PDFs and speed PDF (normalized over magnitude)."
    )
    # Drift options
    parser.add_argument("--vx", type=float, default=0.0, help="Drift component v_x (default: 0.0)")
    parser.add_argument("--vy", type=float, default=0.0, help="Drift component v_y (default: 0.0)")
    parser.add_argument("--v",  type=float, default=None, help="(Optional) Back-compat: sets v_x only.")
    # Process params
    parser.add_argument("--D",  type=float, default=1.0, help="Diffusion coefficient D (default: 1.0)")
    parser.add_argument("--dt", type=float, default=0.1, help="Time step dt (default: 0.1)")
    parser.add_argument("--n",  type=int,   default=10000, help="Number of particles (default: 10000)")
    # Misc
    parser.add_argument("--bins", type=int, default=60, help="Histogram bins (default: 60)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting (print stats only)")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    # Drift vector (2D)
    vx = args.vx if args.v is None else args.v
    vy = args.vy
    vvec = np.array([vx, vy], dtype=float)
    vmag = np.linalg.norm(vvec)

    D, dt, n = args.D, args.dt, args.n
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if D < 0:
        raise ValueError("D must be nonnegative.")
    if n < 2:
        raise ValueError("n must be >= 2.")

    # Component std of velocities: V = v + sqrt(2D/dt) * N(0, I_2)
    sigma = np.sqrt(2.0 * D / dt)

    # --- Sample 2D velocities ---
    W = np.random.normal(size=(n, 2))  # i.i.d. N(0,1)
    V = vvec + sigma * W               # shape (n, 2)
    speeds = np.linalg.norm(V, axis=1) # |V|

    # --- Basic stats ---
    emp_mean_vec = V.mean(axis=0)
    emp_cov = np.cov(V.T, ddof=1)
    emp_speed_mean = speeds.mean()
    emp_speed_std = speeds.std(ddof=1)

    print("=== 2D Velocity sampling (V = dX/dt) ===")
    print(f"n = {n}, D = {D}, dt = {dt}")
    print(f"Drift vector v = ({vx}, {vy}), |v| = {vmag}")
    print(f"Component theoretical std sigma = sqrt(2D/dt) = {sigma:.6f}")
    print("\nEmpirical component means (Vx, Vy):", emp_mean_vec)
    print("Empirical component covariance matrix:\n", emp_cov)
    print(f"Empirical speed mean: {emp_speed_mean:.6f}")
    print(f"Empirical speed std:  {emp_speed_std:.6f}")

    if args.no_plot:
        return

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Component histograms (density=True => component PDFs)
    comp_labels = ["Vx", "Vy"]
    for i, ax in enumerate([axes[0, 0], axes[0, 1]]):
        data = V[:, i]
        ax.hist(data, bins=args.bins, density=True, alpha=0.6, edgecolor="black",
                label="Empirical (hist)")
        x = np.linspace(data.min(), data.max(), 600)
        ax.plot(x, norm.pdf(x, loc=vvec[i], scale=sigma), lw=2,
                label=f"Theoretical N({vvec[i]:.3g}, {sigma:.3g}²)")
        ax.set_title(f"{comp_labels[i]} PDF")
        ax.set_xlabel(comp_labels[i])
        ax.set_ylabel("Density")
        ax.legend()

    # Speed histogram (density=True => PDF in speed, normalized over magnitude)
    ax_speed = axes[1, 0]
    ax_speed.hist(speeds, bins=args.bins, density=True, alpha=0.6, edgecolor="black",
                  label="Empirical |V| PDF (hist)")

    # Stable theoretical speed PDF using ncx2 change-of-variables (df=2)
    r_grid = np.linspace(0.0, speeds.max(), 800)
    f_r = speed_pdf_theoretical_2d(r_grid, sigma=sigma, vmag=vmag)
    ax_speed.plot(r_grid, f_r, lw=2, label="Theoretical |V| PDF")

    ax_speed.set_title("Speed (|V|) PDF — normalized over magnitude (2D)")
    ax_speed.set_xlabel("Speed |V|")
    ax_speed.set_ylabel("Density")
    ax_speed.legend()

    # Hide the unused bottom-right axis
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
