"""Nearest Normal Value (NNV) fault diagnosis module.

Provides helper functions for computing NNV-based SPE contributions
integrated with :class:`~bibmon._generic_model.GenericModel`.

Includes a pure-Python ADWIN drift detector implementation.
"""

import math
import numpy as np
import pandas as pd
import warnings
from typing import Optional, Tuple, Union
from collections import deque as _deque


class _Bucket:
    """Bucket data structure used internally by ADWIN."""

    __slots__ = ("_max_size", "current_idx", "_total", "_variance")

    def __init__(self, max_size: int = 5):
        self._max_size = max_size
        self.current_idx = 0
        self._total = [0.0] * (max_size + 1)
        self._variance = [0.0] * (max_size + 1)

    def insert_data(self, value: float, variance: float) -> None:
        self._total[self.current_idx] = value
        self._variance[self.current_idx] = variance
        self.current_idx += 1

    def remove(self) -> None:
        self.compress(1)

    def compress(self, n_elements: int) -> None:
        length = len(self._total)
        for i in range(n_elements, length):
            self._total[i - n_elements] = self._total[i]
            self._variance[i - n_elements] = self._variance[i]
        for i in range(length - n_elements, length):
            self._total[i] = 0.0
            self._variance[i] = 0.0
        self.current_idx -= n_elements

    def get_total_at(self, index: int) -> float:
        return self._total[index]

    def get_variance_at(self, index: int) -> float:
        return self._variance[index]


class ADWIN:
    """ADWIN (ADaptive WINdowing) drift detector.

    Pure-Python implementation based on the algorithm described by
    Bifet & Gavalda (2007).

    Parameters
    ----------
    delta : float
        Confidence parameter (smaller ⇒ fewer false positives).
    clock : int
        Check for change every *clock* samples (default 32).
    max_buckets : int
        Maximum buckets per level before merging (default 5).
    min_window_length : int
        Minimum sub-window length for cut evaluation (default 5).
    grace_period : int
        Minimum samples before any detection (default 10).
    """

    def __init__(self, delta: float = 0.002, clock: int = 32,
                 max_buckets: int = 5, min_window_length: int = 5,
                 grace_period: int = 10):
        self.delta = delta
        self.clock = clock
        self.max_buckets = max_buckets
        self.min_window_length = min_window_length
        self.grace_period = grace_period
        self._reset()

    def _reset(self) -> None:
        self._bucket_deque: _deque[_Bucket] = _deque(
            [_Bucket(max_size=self.max_buckets)]
        )
        self._total: float = 0.0
        self._variance: float = 0.0
        self._width: float = 0.0
        self._n_buckets: int = 0
        self._tick: int = 0
        self._n_detections: int = 0
        self.drift_detected: bool = False

    # --- Public properties --------------------------------------------

    @property
    def width(self) -> float:
        return self._width

    @property
    def n_detections(self) -> int:
        return self._n_detections

    @property
    def variance(self) -> float:
        return self._variance

    @property
    def total(self) -> float:
        return self._total

    @property
    def estimation(self) -> float:
        return self._total / self._width if self._width > 0 else 0.0

    @property
    def _variance_in_window(self) -> float:
        return self._variance / self._width if self._width > 0 else 0.0

    # --- Public update ------------------------------------------------

    def update(self, value: float) -> None:
        """Feed *value* and check for drift."""
        if self.drift_detected:
            self._reset()
        self._insert_element(value, 0.0)
        self.drift_detected = self._detect_change()

    # --- Internal helpers ---------------------------------------------

    def _insert_element(self, value: float, variance: float) -> None:
        bucket = self._bucket_deque[0]
        bucket.insert_data(value, variance)
        self._n_buckets += 1

        self._width += 1
        inc_var = 0.0
        if self._width > 1.0:
            inc_var = (
                (self._width - 1.0)
                * (value - self._total / (self._width - 1.0))
                * (value - self._total / (self._width - 1.0))
                / self._width
            )
        self._variance += inc_var
        self._total += value

        self._compress_buckets()

    @staticmethod
    def _bucket_size(row: int) -> float:
        return 2.0 ** row

    def _delete_element(self) -> float:
        bucket = self._bucket_deque[-1]
        n = self._bucket_size(len(self._bucket_deque) - 1)
        u = bucket.get_total_at(0)
        mu = u / n
        v = bucket.get_variance_at(0)

        self._width -= n
        self._total -= u
        mu_window = self._total / self._width if self._width > 0 else 0.0
        inc_var = v + n * self._width * (mu - mu_window) ** 2 / (n + self._width)
        self._variance -= inc_var

        bucket.remove()
        self._n_buckets -= 1

        if bucket.current_idx == 0:
            self._bucket_deque.pop()

        return n

    def _compress_buckets(self) -> None:
        idx = 0
        bucket = self._bucket_deque[0]
        while bucket is not None:
            k = bucket.current_idx
            if k == self.max_buckets + 1:
                if idx + 1 >= len(self._bucket_deque):
                    self._bucket_deque.append(
                        _Bucket(max_size=self.max_buckets)
                    )
                next_bucket = self._bucket_deque[idx + 1]
                n1 = self._bucket_size(idx)
                n2 = self._bucket_size(idx)
                mu1 = bucket.get_total_at(0) / n1
                mu2 = bucket.get_total_at(1) / n2
                total12 = bucket.get_total_at(0) + bucket.get_total_at(1)
                temp = n1 * n2 * (mu1 - mu2) ** 2 / (n1 + n2)
                v12 = (bucket.get_variance_at(0)
                       + bucket.get_variance_at(1) + temp)
                next_bucket.insert_data(total12, v12)
                self._n_buckets += 1
                bucket.compress(2)

                if next_bucket.current_idx <= self.max_buckets:
                    break
            else:
                break

            idx += 1
            if idx < len(self._bucket_deque):
                bucket = self._bucket_deque[idx]
            else:
                bucket = None

    def _detect_change(self) -> bool:
        change_detected = False
        self._tick += 1

        if (self._tick % self.clock != 0) or (self._width <= self.grace_period):
            return False

        reduce_width = True
        while reduce_width:
            reduce_width = False
            exit_flag = False
            n0 = 0.0
            n1 = self._width
            u0 = 0.0
            u1 = self._total
            v0 = 0.0
            v1 = self._variance

            for idx in range(len(self._bucket_deque) - 1, -1, -1):
                if exit_flag:
                    break
                bucket = self._bucket_deque[idx]

                for k_i in range(bucket.current_idx):
                    n2 = self._bucket_size(idx)
                    u2 = bucket.get_total_at(k_i)
                    mu2 = u2 / n2

                    if n0 > 0.0:
                        mu0 = u0 / n0
                        v0 += (
                            bucket.get_variance_at(k_i)
                            + n0 * n2 * (mu0 - mu2) ** 2 / (n0 + n2)
                        )

                    if n1 > 0.0:
                        mu1_val = u1 / n1
                        v1 -= (
                            bucket.get_variance_at(k_i)
                            + n1 * n2 * (mu1_val - mu2) ** 2 / (n1 + n2)
                        )

                    n0 += self._bucket_size(idx)
                    n1 -= self._bucket_size(idx)
                    u0 += bucket.get_total_at(k_i)
                    u1 -= bucket.get_total_at(k_i)

                    if idx == 0 and k_i == bucket.current_idx - 1:
                        exit_flag = True
                        break

                    if n1 < self.min_window_length:
                        exit_flag = True
                        break
                    if n0 < self.min_window_length:
                        continue

                    delta_mean = (u0 / n0) - (u1 / n1)
                    if self._evaluate_cut(n0, n1, delta_mean):
                        reduce_width = True
                        change_detected = True
                        if self._width > 0:
                            n0 -= self._delete_element()
                            exit_flag = True
                            break

        if change_detected:
            self._n_detections += 1

        return change_detected

    def _evaluate_cut(self, n0: float, n1: float,
                      delta_mean: float) -> bool:
        delta_prime = math.log(2.0 * math.log(self._width) / self.delta)
        m_recip = (
            1.0 / (n0 - self.min_window_length + 1)
            + 1.0 / (n1 - self.min_window_length + 1)
        )
        epsilon = (
            math.sqrt(2.0 * m_recip * self._variance_in_window * delta_prime)
            + 2.0 / 3.0 * delta_prime * m_recip
        )
        return abs(delta_mean) > epsilon


# Backward-compatibility alias
_ADWINPure = ADWIN


def compute_nnv_contributions(map_from_X, X, reference_values=None):
    """Compute NNV contributions for a batch of observations.

    For each observation and each variable the algorithm replaces the
    variable with its reference (normal) value, re-runs the model, and
    measures the absolute change in SPE.  The result is normalised so
    that each row sums to 1.

    Parameters
    ----------
    map_from_X : callable
        Model reconstruction/prediction function that receives a 2-D
        ``numpy.ndarray`` and returns a 2-D ``numpy.ndarray`` of the
        same shape (e.g. ``GenericModel.map_from_X``).
    X : numpy.ndarray
        Observation matrix (n_samples × n_variables), already
        pre-processed / normalised.
    reference_values : numpy.ndarray or None
        Reference value for each variable (length *n_variables*).
        If *None*, zeros are used (appropriate when the data have been
        normalised to zero mean).

    Returns
    -------
    contributions : numpy.ndarray
        Matrix (n_samples × n_variables) of normalised NNV
        contributions.
    """
    X = np.asarray(X, dtype=float)
    n_obs, n_vars = X.shape

    if reference_values is None:
        reference_values = np.zeros(n_vars)
    else:
        reference_values = np.asarray(reference_values, dtype=float)

    contributions = np.zeros_like(X)

    for i in range(n_obs):
        xi = X[i]
        pred_orig = map_from_X(xi.reshape(1, -1)).flatten()
        spe_orig = np.sum((xi - pred_orig) ** 2)

        for j in range(n_vars):
            modified = xi.copy()
            modified[j] = reference_values[j]
            pred_mod = map_from_X(modified.reshape(1, -1)).flatten()
            spe_mod = np.sum((modified - pred_mod.flatten()) ** 2)
            contributions[i, j] = abs(spe_orig - spe_mod)

        row_sum = contributions[i].sum()
        if row_sum > 0:
            contributions[i] /= row_sum

    return contributions


def run_nnv_analysis(model, dataset,
                     reference_values=None,
                     adwin_delta=0.01,
                     adwin_clock=1,
                     ema_alpha=0.05,
                     f_pp=None):
    """Execute a complete NNV analysis with adaptive reference updating.

    Uses ADWIN drift detection to adaptively update reference
    values during normal operation and reset them when concept
    drift is detected.

    Parameters
    ----------
    model : GenericModel
        A trained BibMon model (must expose ``map_from_X``, ``limSPE``,
        ``predict``, and pre-processing attributes).
    dataset : pandas.DataFrame
        Complete dataset for monitoring.  Each row is an observation.
    reference_values : pandas.Series or None
        Initial reference values for each variable.  When *None*, the
        training-set column means (normalised → zeros) are used.
    adwin_delta : float
        Sensitivity parameter for the ADWIN drift detector.
    adwin_clock : int
        How often ADWIN checks for drift. 1 means every sample
        (recommended for real-time monitoring), 32 is the default.
    ema_alpha : float
        Smoothing factor for the exponential moving average used to
        track the normal operating reference (0 < ema_alpha <= 1).
        Effective memory window ≈ 1/ema_alpha samples.
        Typical range: 0.01 – 0.05 (default 0.05 ≈ 20-sample window).
    f_pp : list or None
        Preprocessing functions passed to ``model.predict``.

    Returns
    -------
    contrib_NNV : pandas.DataFrame
        Normalised NNV contributions (n_samples × n_variables).
    spe_series : pandas.Series
        SPE value for each sample.
    final_reference : numpy.ndarray
        Reference values at the end of the analysis.  Can be passed
        back as ``reference_values`` to resume monitoring.
    reference_history : pandas.DataFrame
        Reference values at each time step (same index and columns
        as *dataset*), useful for diagnostics and visualisation.
    drift_indices : list of int
        Positional indices where ADWIN detected drift.

    Raises
    ------
    ValueError
        If the dataset is empty.
    """
    if dataset.empty:
        raise ValueError("Dataset cannot be empty")

    control_limit = float(model.limSPE)
    tags = model.tags_X
    n_vars = len(tags)

    # Resolve initial reference values
    if reference_values is None:
        ref = np.zeros(n_vars)
    else:
        ref = np.asarray(reference_values, dtype=float)
        if ref.shape[0] != n_vars:
            raise ValueError(
                f"reference_values has {ref.shape[0]} elements but model has "
                f"{n_vars} variables ({tags})."
            )

    spe_values = []
    spe_index = []
    preprocessed_values = []  # stores (X_preprocessed, spe) per sample
    contrib_NNV = pd.DataFrame(
        index=dataset.index, columns=tags, dtype=float
    )
    reference_history = pd.DataFrame(
        index=dataset.index, columns=tags, dtype=float
    )
    drift_indices = []

    adwin = ADWIN(delta=adwin_delta, clock=adwin_clock)

    for i in range(dataset.shape[0]):
        # Extract single observation
        df_now = pd.DataFrame(dataset.iloc[i]).T

        # Run prediction through the standard BibMon pipeline
        model.predict(df_now, f_pp=f_pp)
        spe_value = float(model.SPE_test.iloc[0])
        x_preprocessed = model.X_test.values.flatten().copy()

        spe_values.append(spe_value)
        spe_index.append(df_now.index[0])
        preprocessed_values.append((x_preprocessed, spe_value))

        # Update ADWIN
        adwin.update(spe_value)

        # NNV contributions use the reference *before* this sample is absorbed
        try:
            x_now = model.X_test.values
            contribs = compute_nnv_contributions(
                model.map_from_X, x_now, reference_values=ref
            )
            contrib_NNV.loc[df_now.index[0]] = contribs.flatten()
        except Exception:
            contrib_NNV.loc[df_now.index[0]] = np.zeros(dataset.shape[1])

        # Update reference if within normal limits (fixed-alpha EMA)
        if spe_value <= control_limit:
            ref = (1 - ema_alpha) * ref + ema_alpha * x_preprocessed

        # Drift detection — recompute reference from recent normal window
        if adwin.drift_detected:
            drift_indices.append(i)
            if spe_value <= control_limit:
                width = adwin.width
                recent_width = int(width) if np.isfinite(width) and width > 0 else i + 1
                recent_start = max(0, i - recent_width + 1)
                # Collect preprocessed values from the recent window
                # that were within the control limit (normal operation)
                normal_vals = [
                    xp for xp, sp in preprocessed_values[recent_start:i + 1]
                    if sp <= control_limit
                ]
                if normal_vals:
                    ref = np.mean(normal_vals, axis=0)
                # else: keep current ref (no normal samples in window)
            adwin = ADWIN(delta=adwin_delta, clock=adwin_clock)

        # Record reference snapshot (state after incorporating current sample,
        # i.e., the reference that will be used for the next observation)
        reference_history.iloc[i] = ref

    spe_series = pd.Series(spe_values, index=spe_index, name="SPE")
    contrib_NNV = contrib_NNV.fillna(0.0)

    return contrib_NNV, spe_series, ref.copy(), reference_history, drift_indices
