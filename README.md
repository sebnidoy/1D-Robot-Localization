# 1D Robot Localization with Kalman Filtering and RTS Smoothing

An implementation of sequential state estimation for a mobile robot navigating a 1D rail. The robot's position is estimated by fusing wheel odometry and laser rangefinder measurements using a Kalman filter forward pass followed by a Rauch–Tung–Striebel (RTS) smoother backward pass. The solution is analyzed across varying measurement subsampling rates to study the effect of measurement frequency on estimation accuracy and uncertainty.

---

## Dataset

<video src="dataset1_4x.mov" controls width="100%"></video>

The video above (played at 4× speed) shows the physical experiment used to generate `dataset1.mat` - referred to as the **"Giant Glass of Milk"** dataset. A mobile robot traverses a straight rail back and forth for approximately 20 minutes while two sensors record data simultaneously: a wheel encoder producing noisy velocity readings, and a laser rangefinder measuring distance to a fixed cylindrical landmark. The ground-truth trajectory `x_true` was obtained via a high-accuracy reference system, enabling quantitative evaluation of the estimator.

Dataset courtesy of **Prof. Timothy D. Barfoot** (University of Toronto Institute for Aerospace Studies). 

---

## Problem Setup

A mobile robot drives back and forth along a straight rail for approximately 20 minutes (~12,709 timesteps at T = 0.1 s). Two sensor modalities are available:

- **Wheel odometry** - noisy velocity measurements `v_k`, used at every timestep
- **Laser rangefinder** - range `r_k` to a fixed cylindrical landmark at known position `x_c`

The range measurement is transformed into a direct (noisy) position measurement:

```
y_k = x_c - r_k ≈ x_k + n_k
```

The discrete-time motion and measurement models are:

```
x_k = x_{k-1} + T * u_k + w_k      (motion model)
y_k = x_k + n_k                     (measurement model)
```

where `w_k ~ N(0, Q_k)` and `n_k ~ N(0, R_k)` are zero-mean Gaussian noise terms.

The noise parameters used are:

| Parameter | Value |
|-----------|-------|
| Range std dev `σ_r` | 0.019155 m |
| Odometry std dev `σ_v` | 0.047554 m/s |
| Measurement variance `R = σ_r²` | 3.669 × 10⁻⁴ m² |
| Process variance `Q = T²σ_v²` | 2.261 × 10⁻⁵ m² |

---

## Method

### Batch Formulation (MAP / Weighted Least Squares)

Because both models are linear and Gaussian, the MAP trajectory estimate is the solution to a weighted least-squares problem:

```
J(x) = (1/2)(z - Hx)^T W^{-1} (z - Hx)
```

The normal equations yield:

```
x* = (H^T W^{-1} H)^{-1} H^T W^{-1} z
```

The left-hand matrix `H^T W^{-1} H` is **tridiagonal, symmetric, and positive definite** - its sparsity structure is visualized in `figs/sparsity_lhs.png`.

### Sequential Solution: Kalman Filter + RTS Smoother

Rather than inverting the full 12709×12709 batch matrix, the equivalent estimate is computed in O(K) time using:

1. **Kalman Filter (forward pass)** - propagates the state mean and covariance forward in time, applying measurement updates wherever range data is available.
2. **RTS Smoother (backward pass)** - refines the filtered estimates by propagating information backward through the trajectory.

This exploits the Markov structure of the problem and produces the same mean and covariance as the batch solution.

### Measurement Subsampling

The pipeline is evaluated under four subsampling rates `δ`:

| δ | Measurements used |
|---|-------------------|
| 1 | Every timestep |
| 10 | Every 10th timestep |
| 100 | Every 100th timestep |
| 1000 | Every 1000th timestep |

All odometry measurements are always used. Skipping range measurements corresponds to removing rows from the measurement selector matrix in the batch view, or skipping Kalman update steps in the sequential view.

---

## Repository Structure

```
.
├── main.py              # Full pipeline: data loading, KF+RTS, plotting
├── dataset1.mat         # Input dataset (not included - see below)
├── requirements.txt     # Python dependencies
├── figs/                # Output figures (generated on run)
│   ├── sparsity_lhs.png
│   ├── error_envelope_delta_1.png
│   ├── error_envelope_delta_10.png
│   ├── error_envelope_delta_100.png
│   ├── error_envelope_delta_1000.png
│   ├── hist_error_delta_1.png
│   ├── hist_error_delta_10.png
│   ├── hist_error_delta_100.png
│   └── hist_error_delta_1000.png
└── README.md
```

---

## Setup and Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Obtain the dataset

Place `dataset1.mat` (the "Giant Glass of Milk" dataset) in the project root. The dataset contains: `t`, `x_true`, `l` (cylinder center), `r`, `r_var`, `v`, `v_var`.

### 3. Run

```bash
python main.py
```

Output figures are saved to `figs/`.

---

## Outputs

For each subsampling rate `δ`, the pipeline generates:

- **Error + uncertainty envelope** - estimation error `x*_k - x_true,k` over time, overlaid with ±3σ bounds from the smoothed covariance
- **Error histogram** - distribution of estimation errors, validating the zero-mean Gaussian noise assumption

Additionally, the **sparsity pattern** of the batch normal matrix `H^T W^{-1} H` is plotted.

### Key Findings

- At `δ=1`, error is small but uncertainty bounds can be overly tight (overconfidence).
- As `δ` increases, odometry drift accumulates between sparse range updates; the covariance grows between updates and collapses when a range measurement arrives, producing visible "arches" in the uncertainty envelope.
- At `δ=1000`, uncertainty is wide but the RTS smoother still recovers a reasonable estimate by propagating future measurement information backward.
- Error histograms remain centered near zero across all `δ`, consistent with the zero-mean noise assumption.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations and linear algebra |
| `scipy` | `.mat` file loading, sparse matrix construction |
| `matplotlib` | Plotting and figure export |
