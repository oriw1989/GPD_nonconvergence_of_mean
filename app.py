import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Generalized Pareto Distribution mean nonconvergence Demo", layout="centered")

st.title("Generalized Pareto Distribution mean nonconvergence demo")

# --- Sidebar controls (only XI and N, as requested) ---
st.sidebar.header("Parameters")
XI = st.sidebar.slider("Tail shape ξ", min_value=0.0, max_value=1.5, value=1.1, step=0.01)
N = st.sidebar.number_input("Sample size N", min_value=10_000, max_value=1_000_000, value=200_000, step=10_000)

run = st.sidebar.button("Run simulation")

# --- Fixed parameters (same spirit as your notebook) ---
P_TAIL = 0.05            # probability routed to tail (kept fixed for simplicity)
K, THETA = 5.0, 2.0      # Gamma core (mean = 10)
SEED = 20250823          # fixed seed for reproducibility

rng = np.random.default_rng(SEED)

# --- Helpers from the notebook (slightly guarded for xi≈0) ---
def rvs_gpd(n, xi, beta, rng):
    u0 = rng.uniform(size=n)
    if abs(xi) < 1e-12:
        return -beta * np.log(1.0 - u0)  # exponential limit
    return (beta/xi) * ((1.0 - u0) ** (-xi) - 1.0)

def sample_gamma_gpd(n, p_tail, xi, beta, u, k, theta, rng):
    flags = rng.uniform(size=n) < p_tail
    x = rng.gamma(shape=k, scale=theta, size=n)
    x = np.minimum(x, u)  # keep core at/below threshold (simple splice for demo)
    m = int(flags.sum())
    if m > 0:
        x[flags] = u + rvs_gpd(m, xi=xi, beta=beta, rng=rng)
    return x

# Main panel text
st.write(
    "This app stitches a fixed **Gamma(5,2)** core with a **GPD** tail. "
    "Pick ξ and N, click **Run simulation**, and see whether the converges."
)

if run:
    # 1) Estimate threshold u and GPD scale beta from a large draw of the core
    core_sample = rng.gamma(shape=K, scale=THETA, size=400_000)
    # Pick u so that the Gamma core's exceedance probability matches p_tail
    u = float(np.quantile(core_sample, 1.0 - P_TAIL))
    exceed = core_sample[core_sample > u] - u
    beta = float(exceed.mean()) if exceed.size > 0 else 1.0  # smooth scale match

    # 2) Theoretical verdict on mean convergence
    mean_infinite = (XI >= 1.0) and (P_TAIL > 0.0)
    verdict = "WILL NOT converge (infinite mean in theory)" if mean_infinite else "WILL converge (finite mean in theory)"
    st.success(f"You picked ξ = {XI:.2f} with p_tail = {P_TAIL:.3f} ⇒ the mean **{verdict}**.")

    # 3) Simulate and plot running mean
    x = sample_gamma_gpd(N, P_TAIL, XI, beta, u, K, THETA, rng)
    rm = np.cumsum(x) / np.arange(1, x.size + 1)

    fig, ax = plt.subplots()
    ax.plot(np.arange(1, rm.size + 1), rm)
    ax.set_xlabel("n")
    ax.set_ylabel("Running mean")
    ax.set_title(f"Running mean — Gamma core + GPD tail (ξ={XI:.2f}, N={N:,})")
    st.pyplot(fig)
else:
    st.info("Set ξ and N in the sidebar, then click **Run simulation**.")
