# app.py
# ------------------------------------------------------------
# #### This app will use a generalized Pareto distribution to
# demonstrate the property of convergence / non-convergence
# of the mean, depending on the shape parameter ξ.
#
# #### The distribution has a Gamma core and GPD tail. If ξ ≥ 1
# (and some tail probability > 0), the theoretical mean is infinite.
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ---- Streamlit page setup ----
st.set_page_config(page_title="Gamma core + GPD tail — mean convergence demo", layout="centered")
st.title("Gamma core + GPD tail — mean convergence demo")

# ---- Fixed parameters ----
P_TAIL = 0.05          # probability of routing to the tail
K, THETA = 5.0, 2.0    # Gamma core
SEED = 20250823        # random seed for reproducibility

rng = np.random.default_rng(SEED)

# ---- GPD sampler via inverse CDF ----
def rvs_gpd(n, xi, beta, rng):
    u0 = rng.uniform(size=n)
    if abs(xi) < 1e-12:  # exponential limit case
        return -beta * np.log(1.0 - u0)
    return (beta / xi) * ((1.0 - u0) ** (-xi) - 1.0)

# ---- Gamma + GPD sampler ----
def sample_gamma_gpd(n, p_tail, xi, beta, u, k, theta, rng):
    flags = rng.uniform(size=n) < p_tail
    x = rng.gamma(shape=k, scale=theta, size=n)
    x = np.minimum(x, u)
    m = int(flags.sum())
    if m > 0:
        x[flags] = u + rvs_gpd(m, xi=xi, beta=beta, rng=rng)
    return x

# ---- Run one full demo ----
def run_simulation(xi, N):
    # Estimate threshold u from Gamma core, matching tail probability
    core_sample = rng.gamma(shape=K, scale=THETA, size=400_000)
    u = float(np.quantile(core_sample, 1.0 - P_TAIL))

    # Exceedance mean sets GPD scale β
    exceed = core_sample[core_sample > u] - u
    beta = float(exceed.mean()) if exceed.size > 0 else 1.0

    # Decide theoretical convergence
    mean_infinite = (xi >= 1.0) and (P_TAIL > 0.0)
    if mean_infinite:
        st.write(f"You picked ξ={xi:.2f}, which is greater or equal to 1 → the mean will **NOT** converge.")
    else:
        st.write(f"You picked ξ={xi:.2f}, which is less than 1 → the mean **WILL** converge.")

    # Generate N samples
    x = sample_gamma_gpd(N, P_TAIL, xi, beta, u, K, THETA, rng)

    # Running mean
    rm = np.cumsum(x) / np.arange(1, x.size + 1)

    fig, ax = plt.subplots()
    ax.plot(np.arange(1, rm.size + 1), rm)
    ax.set_xlabel("n")
    ax.set_ylabel("Running mean")
    ax.set_title(f"Running mean — Gamma core + GPD tail (ξ={xi:.2f}, N={N:,})")
    st.pyplot(fig)

# ---- Sidebar controls ----
st.sidebar.header("Parameters")
XI = st.sidebar.slider("Tail shape ξ", min_value=0.0, max_value=1.5, value=1.1, step=0.01)
N = st.sidebar.number_input("Sample size N", min_value=10_000, max_value=1_000_000, value=200_000, step=10_000)

# Button
run_clicked = st.sidebar.button("Run simulation")

# ---- Default run on app load ----
if run_clicked:
    run_simulation(XI, N)
else:
    run_simulation(XI, N)
