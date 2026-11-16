# HoBlunt

Python implementation of the Haar–Basis adaptive algorithm described by K. C. Ho and S. D. Blunt:

> K. C. Ho and S. D. Blunt, “Rapid Identification of a Sparse Impulse Response Using an Adaptive Algorithm in the Haar Domain,” *IEEE Transactions on Signal Processing*, vol. 51, no. 3, pp. 628–638, March 2003. DOI: [10.1109/TSP.2002.808077](https://doi.org/10.1109/TSP.2002.808077).

The repository also contains an NLMS reference algorithm, stochastic signal generators, ITU-T G.168 echo-path models, experiment runners, and exploratory Jupyter notebooks.

## Repository map

| Path | Responsibility |
|---|---|
| `Haar/Haar_matrix.py` | Builds the orthonormal Haar basis and provides level and tree-index operations. |
| `Haar_Basis_Algorithm.py` | Haar–Basis adaptive algorithm, Control-coefficient detection, and active Parent/Children selection. |
| `NLMS.py` | Conventional normalized LMS baseline. |
| `np_detector/np_detector.py` | Neyman–Pearson threshold decision for a Control coefficient. |
| `power_estimator/power_estimator.py` | Exponentially smoothed estimate of signal power. |
| `generators/regression_vector.py` | Time-domain regression vector. |
| `generators/generators.py` | AR, white Gaussian noise, and speech-file signal sources. |
| `line_echo/lineecho.py` | ITU-T G.168 Annex D echo-path models m1–m8. |
| `experiment_runner.py` | Sequential, Monte Carlo, and process-parallel experiment execution. |
| `testing/` | Jupyter notebooks for component checks and end-to-end experiments. |

## Haar transform conventions

For a filter length $N=2^M$, construct the matrix with:

```python
from Haar.Haar_matrix import Haar_matrix

haar = Haar_matrix(max_level=M)
```

**Basis vectors are stored in columns** of `transform_matrix`:

- `haar.transform_matrix[:, k]` is basis vector $k$.
- Transform to the Haar domain: `z = haar.transform_matrix.T @ x`.
- Transform coefficients back to the time domain: `w = haar.transform_matrix @ b`.

The coefficient indices are grouped into levels:

| Level | Global coefficient indices |
|---|---|
| $S_0$ | `0` |
| $S_i$, $i \geq 1$ | `2**(i-1)` through `2**i - 1` |

Thus, $S_0$ contains one scaling vector and each subsequent level contains the Haar wavelets at one scale. For an $N=2^M$ transform, the complete index range is $0,\ldots,N-1$.

Tree methods accept and return **global coefficient indices**:

- `find_indexes_level(level)`: indices belonging to a level.
- `find_first_index_in_level(level)`: first index of a level.
- `get_level(k)`: level containing coefficient $k$.
- `parents_of(k)`: Parent coefficient indices of $k$.
- `children_of(k)`: Children coefficient indices of $k$.

The Parent/Children relations are the temporal-hierarchy relations used by the Haar–Basis algorithm. They are not a general wavelet-packet tree interface.

## Haar–Basis algorithm

The current constructor is:

```python
Haar_Basis_Algorithm(
    Haar_matrix,
    filter_length,
    control_level,
    alpha,
    G,
    beta_fa,
    mu,
    force_initial_selection=False,
    zero_inactive_coefficients=True,
)
```

| Argument | Meaning |
|---|---|
| `Haar_matrix` | An initialized `Haar_matrix` instance. |
| `filter_length` | Time-domain filter length $N$; the implementation requires a power of two. |
| `control_level` | Haar level whose coefficients are tested for nonzero significance. |
| `alpha` | Smoothing factor for the input-power estimators, one per Haar level. |
| `G` | Positive factor used to determine the Adaptation Interval. |
| `beta_fa` | Multiplier used in the Control-coefficient detection threshold. |
| `mu` | Base adaptation factor. After initialization/reset and each adaptation event, the step size is $\mu/\lvert\mathcal{A}\rvert$. |
| `force_initial_selection` | Enables the first-interval fallback: when no Control coefficient is detected, the three largest-magnitude Control coefficients are used to expand the active set. |
| `zero_inactive_coefficients` | When enabled, coefficients removed from the active set are set to zero. Defaults to `True`. |

Here, $\mathcal{A}$ denotes the active coefficient-index set. The algorithm starts with the scaling/low-level coefficients and all Control coefficients at the selected level. At an Adaptation Interval boundary, `AI()` rebuilds the active set from the Control detections, adding the corresponding Parent and Children coefficients.

### Inactive-coefficient policy

The default `zero_inactive_coefficients=True` keeps the stored coefficient vector consistent with the active structure: coefficients that are no longer active are zeroed, while coefficients that remain active retain their current values. Newly activated coefficients therefore have value zero if they were inactive at the preceding selection.

Setting this option to `False` preserves values of coefficients after they leave the active set. Since `step()` computes the output using the dense product `b @ z`, preserved inactive values continue contributing to the output in this mode. The option is exposed to support controlled comparisons with the previous implementation.

The paper defines the estimated output using only the Control coefficients and the active Parent and Children coefficients (Section II-A, Eq. (1)). It does not prescribe a particular in-memory policy for values of coefficients after deactivation. Zeroing is the default implementation policy; it is not presented as a verbatim instruction from the paper.

### Adaptation Interval

The current implementation uses:

```python
int(G / (4 * step_size))
```

The step size is updated at each call to `AI()`, based on the newly selected active set. Therefore, the next Adaptation Interval may differ from the preceding one. The experiment runner queries the algorithm again after every adaptation event.

### Initialization and reset

Call `reset()` before starting a new realization. It clears the adaptive coefficients and restores the initial active set, resets the power estimators to their configured initial values, recalculates the step size, and restores the one-time initial-selection flag from `force_initial_selection`.

The current implementation initializes the MSE estimator with `initial_value=10.0` and each per-level input-power estimator with `initial_value=1.0`. These are implementation choices; the appropriate values depend on the signal scaling and experiment.

The algorithm exposes:

- `step(new_input, new_desired_signal)`: processes one sample and returns the instantaneous error.
- `AI()`: performs active-set selection and updates the step size.
- `get_adaptation_interval()`: returns the current interval in samples.
- `get_time_coefficients()`: returns the time-domain coefficient vector.
- `get_npea()`: returns the active-set size.
- `initialize_regressor(regressor)`: initializes the input history.

## Detector and power estimator

The detector in `np_detector/np_detector.py` implements the threshold test:

[
T = \beta_{fa}\sqrt{\frac{\mu_{step}\,\widehat\xi}{\widehat\lambda_\Delta}},
\qquad
\text{detected} \iff |b_k| > T.
]

The arguments `mu`, `xi`, and `lambda_delta` passed to `np_detector()` are, respectively, the current step size, the current MSE estimate, and the input-power estimate for the Control level. The function returns a Boolean and does not modify algorithm state.

`Power_Estimator` uses exponential averaging of squared samples:

[
\widehat P[n]=(1-\alpha)\widehat P[n-1]+\alpha u^2[n].
]

Its constructor accepts `alpha` and `initial_value`. `reset()` restores the configured initial value; `update_alpha()` changes the smoothing factor without resetting the stored estimate.

## Signal and noise generators

`SignalGenerator` defines the methods used by the experiment code:

- `generate(n_samples)`: generate a block of samples.
- `generate_sample()`: generate one sample.
- `autocorrelation(lag)`: provide an autocorrelation sequence/value for the source.

Implemented classes:

- `ARGenerator`: autoregressive Gaussian process driven by white Gaussian innovations. With denominator coefficients (a_i), the recurrence implemented is (x[n] = -\sum_i a_i x[n-i] + u[n]). `process_variance` specifies the desired stationary output variance; the innovation variance is computed from the Yule–Walker solution.
- `GaussianNoiseGenerator`: zero-mean white Gaussian noise.
- `SpeechFileGenerator`: reads a WAV file and returns its samples sequentially, wrapping to the beginning when the end is reached. It also accepts an AR-coefficient file for the inherited initialization/statistics interface.

`RegressionVector` stores the history as ([x[n],x[n-1],\ldots,x[n-N+1]]^T). A new sample is inserted at index zero.

## Echo-path models

`line_echo/lineecho.py` provides the G.168 Annex D models `m1()` through `m8()`. Each function returns a NumPy array containing the tabulated impulse-response samples multiplied by the model's scale factor.

A typical experiment embeds one of these responses in a longer zero-filled vector to introduce an echo delay. The model functions return the path samples themselves; delay placement and overall experiment configuration belong to the experiment script.

## Experiment runner

`experiment_runner.py` provides three entry points:

- `run_realization(...)`: runs one realization sequentially.
- `run_monte_carlo(...)`: averages sequential realizations.
- `run_parallel_monte_carlo(...)`: distributes realizations among worker processes and combines their accumulated results.

The runner expects each algorithm to implement the common interface:

```python
reset()
initialize_regressor(regressor)
step(new_input, new_desired_signal)
get_time_coefficients()
get_npea()
get_adaptation_interval()
AI()
```

Each algorithm has its own adaptation counter and current interval. The runner advances the counters once per input sample. When an interval expires, it calls that algorithm's `AI()` and then queries `get_adaptation_interval()` again. This permits the interval to change after an adaptation event without embedding algorithm-specific timing logic in the runner.

The returned dictionary is keyed by algorithm name. Each algorithm has three arrays, each of length `n_samples`:

| Metric | Value stored at each sample |
|---|---|
| `MSD` | Squared Euclidean norm of the difference between the estimated and true time-domain coefficient vectors. |
| `MSE` | Instantaneous squared error $e^2[n]$. |
| `NPEA` | Number of effectively adapted coefficients reported by the algorithm. |

The runner returns MSD and MSE in **linear scale**. The experiment notebook plots them in decibels using (10\log_{10}(\cdot)). Monte Carlo routines average each metric sample-by-sample over the requested realizations.

## Notebooks

| Notebook | Purpose |
|---|---|
| `testing/test_HB_algorithm.ipynb` | Basic inspection of the Haar–Basis algorithm and Haar-tree methods. |
| `testing/test_algorithms.ipynb` | End-to-end comparison of Haar–Basis and NLMS using an AR input and a sparse echo path; runs Monte Carlo experiments and plots MSD, MSE, and NPEA. |
| `testing/test_detector.ipynb` | Exploratory inspection of the Neyman–Pearson threshold decision. |
| `testing/test_haar_matrix.ipynb` | Visualization of the Haar matrix and inspection of level/Parent/Children indices. |
| `testing/test_power_estimator.ipynb` | Demonstrates the exponential power estimator on an AR(1) process. |

The notebooks are exploratory research material, not an automated unit-test suite. The end-to-end notebook currently embeds its experiment settings in notebook cells.

## Running the experiments

The notebooks add the repository root to `sys.path` based on the current working directory, and therefore expect to be launched from the `testing/` directory.

The repository does not currently provide a pinned dependency file or package installer configuration. The source code requires NumPy. The notebooks additionally use Matplotlib and SciPy; the speech-file source uses a WAV-loading library. Install the required packages in the active Python environment before running the corresponding notebook or source path.

## Research notes and scope

- This repository is an experimental research implementation, not a production echo canceller.
- The Haar–Basis algorithm is based on Ho and Blunt (2003); consult the paper for derivations, statistical assumptions, and parameter-selection rationale.
- The current Haar matrix is a conventional critically sampled Haar/DWT basis. It is not a Wavelet Packet Transform implementation.
- The notebook plots document particular experiment configurations; they are not performance guarantees.
- The current implementation contains no CI workflow or automated test runner. The notebooks provide manual exploratory checks.
- In `Haar/Haar_matrix.py`, `get_basis_vector()` currently references `max_level` without qualifying it as an instance attribute. Also, `find_indexes_level(max_level)` requests the first index of level `max_level + 1`, while `find_first_index_in_level()` rejects levels above `max_level`. These boundary/API issues should be fixed and covered by tests before relying on those methods at the deepest level.
- `SpeechFileGenerator` calls `librosa.load()`, but `generators/generators.py` currently does not import `librosa`. The speech-file generator therefore requires that missing import to be addressed before use.
