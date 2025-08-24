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
st.set_page_config(page_title="Generalized Pareto Distribution mean convergence demo", layout="centered")
st.markdown("## Pareto Distribution mean convergence")
st.write("This app shows the property of the Pareto distribution, where the mean sometimes does not converge. This distribution can, for example, describe the behavior of viral content at times.")
st.write("A. pick a shape parameter ξ and number of samples to run for mean estimation.") 
st.write("B. Depending on your choice of ξ, you will find either the mean converges, or it will continue to experience jumps due to outliers, and will not converge.")

# ---- Fixed parameters ----
P_TAIL = 0.05          # probability of routing to the tail (kept fixed)
K, THETA = 5.0, 2.0    # Gamma core (mean = K*THETA = 10)
BASE_SEED = 20250823   # base seed; we will offset this per click via session_state

# ---- GPD sampler via inverse CDF (with xi≈0 guard for exponential limit) ----
def rvs_gpd(n, xi, beta, rng):
    u0 = rng.uniform(size=n)
    if abs(xi) < 1e-12:  # xi == 0 => exponential limit
        return -beta * np.log(1.0 - u0)
    return (beta / xi) * ((1.0 - u0) ** (-xi) - 1.0)

# ---- Generate a mixed sample with a gamma core and GPD tail ----
def sample_gamma_gpd(n, p_tail, xi, beta, u, k, theta, rng):
    # flags indicate which samples are routed to the tail (with probability p_tail)
    flags = rng.uniform(size=n) < p_tail

    # core draw
    x = rng.gamma(shape=k, scale=theta, size=n)

    # keep the core at/below threshold u (simple splice, like the notebook)
    x = np.minimum(x, u)

    # replace routed samples with u + GPD exceedances
    m = int(flags.sum())
    if m > 0:
        x[flags] = u + rvs_gpd(m, xi=xi, beta=beta, rng=rng)

    return x

# ---- One function that runs the full demo (message + running mean plot) ----
def run_simulation(xi, N, rng):
    # ## Sample the core distribution to estimate threshold and set a smooth tail scale
    # threshold u is set at the (1 - P_TAIL) quantile, so tail mass matches p_tail
    core_sample = rng.gamma(shape=K, scale=THETA, size=400_000)
    u = float(np.quantile(core_sample, 1.0 - P_TAIL))

    # exceedance distribution (for estimating GPD scale β to blend smoothly)
    exceed = core_sample[core_sample > u] - u
    beta = float(exceed.mean()) if exceed.size > 0 else 1.0  # matching the exceedance scale to GPD

    # ## Theoretical verdict on convergence (for any positive p_tail)
    mean_infinite = (xi >= 1.0) and (P_TAIL > 0.0)
    if mean_infinite:
        st.write(f"You picked ξ={xi:.2f}, which is greater or equal to 1 → the mean will **NOT** converge.")
    else:
        st.write(f"You picked ξ={xi:.2f}, which is less than 1 → the mean **WILL** converge.")

    # ## Generate N samples
    x = sample_gamma_gpd(N, P_TAIL, xi, beta, u, K, THETA, rng)

    # ## Plot running mean to show converge / not-converge visually
    rm = np.cumsum(x) / np.arange(1, x.size + 1)

    fig, ax = plt.subplots()
    ax.plot(np.arange(1, rm.size + 1), rm)
    ax.set_xlabel("n")
    ax.set_ylabel("Running mean")
    ax.set_title(f"Running mean — Gamma core + GPD tail (ξ={xi:.2f}, N={N:,})")
    st.pyplot(fig)

# =========================
# UI + RUN LOGIC (no auto-rerun on slider change; fresh RNG per click)
# =========================

# ---- Sidebar controls (only XI and N, as requested) ----
st.sidebar.header("Parameters")
XI = st.sidebar.slider("Tail shape ξ", min_value=0.0, max_value=1.5, value=1.1, step=0.01, key="xi")
N = st.sidebar.number_input("Sample size N", min_value=10_000, max_value=1_000_000, value=200_000, step=10_000, key="n")

# ---- Session state to control when we actually run, and to get new randomness each time ----
if "has_run" not in st.session_state:
    st.session_state.has_run = False   # we haven't run yet (first load)

if "run_count" not in st.session_state:
    st.session_state.run_count = 0     # how many times the user clicked "Run"

if "seed" not in st.session_state:
    st.session_state.seed = BASE_SEED  # seed that we update on each click

# button: on click, bump the counter and seed
run_clicked = st.sidebar.button("Run simulation")
if run_clicked:
    st.session_state.run_count += 1
    st.session_state.seed = BASE_SEED + st.session_state.run_count

# helper to build a fresh RNG using the current stateful seed
def make_rng():
    return np.random.default_rng(st.session_state.seed)

# ---- Run once on first load, then only when the button is clicked ----
if not st.session_state.has_run:
    # first render: run once with defaults (matches the notebook demo feel)
    rng = make_rng()
    run_simulation(XI, N, rng)
    st.session_state.has_run = True
elif run_clicked:
    # explicit button click → produce a fresh sample
    rng = make_rng()
    run_simulation(XI, N, rng)
else:
    # no run: user changed sliders but hasn't clicked; avoid surprise reruns
    st.info("Adjust ξ and N, then click **Run simulation** to generate a new sample.")
