import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.sparse import diags, vstack, csr_matrix

#################################
# Load Data
#################################
dataset = scipy.io.loadmat("dataset1.mat", squeeze_me=True)

r = np.asarray(dataset["r"]).squeeze() # laser range to center of cylindear
x_true = np.asarray(dataset["x_true"]).squeeze() # ground-truth position
t = np.asarray(dataset["t"]).squeeze() # timestamps
v = np.asarray(dataset["v"]).squeeze() # odometry linear velocity
x_c = np.asarray(dataset.get("l", dataset.get("x_c"))).squeeze() # cylinder center position
r_var = np.asarray(dataset["r_var"]).item() # variance of range measurements
v_var = np.asarray(dataset["v_var"]).squeeze() # variance of velocity measurements

# handout
y = x_c - r

# timestep
T = 0.1

# K=12709
K = len(x_true)

# there are K-1 state transitions, so need K-1 process noise variances
Q_steps = (T**2) * np.full(K - 1, float(v_var)) # process variance for each step

#################################
# Kalman filter + RTS smoother
#################################
def kf_rts_1d(u, y_full, use_meas_mask, Q_steps, r_var, T, big_P0=1e12):

    # create storage for K timesteps (indices 0 to K-1)
    x_f = np.zeros(K) # filtered mean
    P_f = np.zeros(K) # filtered variance
    x_pred = np.zeros(K) # one-step-ahead mean
    P_pred = np.zeros(K) # one-step-ahead variance

    # Prior for the first state (k=0, corresponding to x_1 in 1-based math)
    # Start with an arbitrary mean and a huge variance as a non-informative prior.
    x_pred[0] = 0.0
    P_pred[0] = big_P0

    # The first step is update-only (weve defined its prior)
    if use_meas_mask[0]:
        S0 = P_pred[0] + r_var
        K0 = P_pred[0] / S0 # kalman gain (scalar)
        innov0 = y_full[0] - x_pred[0]
        x_f[0] = x_pred[0] + K0 * innov0 # posterior mean at k=0
        P_f[0] = (1.0 - K0) * P_pred[0] # posterior variance at k=0
    else:
        x_f[0] = x_pred[0]
        P_f[0] = P_pred[0]

    # Forward KF (from second state k=1 up to K-1)
    for k in range(1, K):
        # Predict
        x_pred[k] = x_f[k - 1] + T * u[k]
        P_pred[k] = P_f[k - 1] + Q_steps[k - 1]

        # Update if using measurement at k
        if use_meas_mask[k]:
            S = P_pred[k] + r_var
            Kk = P_pred[k] / S
            innov = y_full[k] - x_pred[k]
            x_f[k] = x_pred[k] + Kk * innov
            P_f[k] = (1.0 - Kk) * P_pred[k]
        else:
            # if the measurement is skipped, the filtered state equals the prediction
            x_f[k] = x_pred[k]
            P_f[k] = P_pred[k]

    # Backward RTS smoothing pass
    x_s = np.copy(x_f)
    P_s = np.copy(P_f)

    # Loop backwards from the second-last state (k = K-2) down to the first state (k=0)
    for k in range(K - 2, -1, -1):
        # RTS smoother gain (for A=I)
        denom = P_pred[k + 1]

        if denom <= 1e-9: # guard against division by zero
            Ck = 0.0
        else:
            Ck = P_f[k] / denom

        # RTS recursions
        x_s[k] = x_f[k] + Ck * (x_s[k + 1] - x_pred[k + 1])
        P_s[k] = P_f[k] + (Ck**2) * (P_s[k + 1] - P_pred[k + 1])

    return x_s, P_s

#################################
# Build the LHS and plot its sparsity
#################################
def plot_normal_matrix_sparsity():

    # problem has K states
    # motion model has K-1 constraints
    # measurement model has K constraints

    # Motion differencing operator A^-1
    Ainv = diags([-1, 1], offsets=[0, 1], shape=(K - 1, K), format='csr')

    # Measurement selector C
    rows = np.arange(K)
    C = csr_matrix((np.ones(K), (rows, rows)), shape=(K, K))

    H = vstack([Ainv, C], format='csr')
    eps = np.finfo(float).tiny # Small epsilon to avoid division by zero
    
    # wights for K-1 motion terms and K measurement terms
    Q_inv_sqrt = 1.0 / np.sqrt(np.maximum(Q_steps, eps))
    R_inv_sqrt = np.full(K, 1.0 / np.sqrt(max(r_var, eps)))
    Winv_sqrt = np.concatenate([Q_inv_sqrt, R_inv_sqrt])
    
    Hw = H.multiply(Winv_sqrt[:, None])

    # normal matrix H^T W^-1 H
    N = (Hw.T @ Hw).tocsc()

    # sparsity plot
    plt.figure()
    plt.spy(N, markersize=0.5)
    plt.title(r"Sparsity of left-hand side")
    plt.xlabel("column")
    plt.ylabel("row")
    plt.tight_layout()
    plt.savefig(outdir / "sparsity_lhs.png", dpi=200)
    plt.close()

#################################
# Run KF + RTS for a given delta
#################################
def batch(delta):
    use_meas_mask = np.zeros(K, dtype=bool) # mask for Q5
    use_meas_mask[::delta] = True # mark every delta index as true, measurement available

    x_hat, P_hat = kf_rts_1d(u=v, y_full=y, use_meas_mask=use_meas_mask, Q_steps=Q_steps, r_var=r_var, T=T)

    err = x_hat - x_true
    sigma = np.sqrt(np.maximum(P_hat, 0.0)) # std
    upper = 3.0*sigma
    lower = -3.0*sigma

    # Error + envelope
    plt.figure()
    plt.plot(t, err, label="error: x* - x_true")
    plt.plot(t, upper, linestyle="--", label="+3σ")
    plt.plot(t, lower, linestyle="--", label="-3σ")
    plt.xlabel("time t [s]")
    plt.ylabel("position error [m]")
    plt.title(f"Error and ±3σ envelope (delta = {delta})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"error_envelope_delta_{delta}.png", dpi=150)
    plt.close()

    # Histogram of raw errors
    plt.figure()
    plt.hist(err, bins=20, density=True) # plot density
    plt.xlabel("error e_k = x*_k - x_true,k  [m]")
    plt.ylabel("density")
    plt.title(f"Histogram of errors (delta = {delta})")
    plt.tight_layout()
    plt.savefig(outdir / f"hist_error_delta_{delta}.png", dpi=150)
    plt.close()

#################################
# Run all the deltas
#################################
outdir = Path("figs")
outdir.mkdir(exist_ok=True)

# Make the sparsity plot for Q4
plot_normal_matrix_sparsity()

deltas = [1, 10, 100, 1000]

# Run the cases
for d in deltas:
    batch(d)

print("\nSaved figures to:", outdir.resolve())