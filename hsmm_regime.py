#!/usr/bin/env python3
"""
Hidden Semi-Markov Model (HSMM) for Regime Detection in Returns
----------------------------------------------------------------

This script implements a practical HSMM via Viterbi (hard-EM) training using
explicit-duration modeling (a.k.a. semi-Markov Viterbi). It is designed for
quant finance regime detection (e.g., low-vol vs high-vol regimes) with
Gaussian emissions and Poisson duration distributions.

Key features
- Explicit state durations (semi-Markov): no geometric duration assumption.
- Poisson duration distributions per state (parameter lambda_k learned from data).
- Gaussian emissions per state (mu_k, sigma_k learned from data).
- Hard-EM training: alternate between semi-Markov Viterbi decoding and M-step
  parameter updates.
- Works out of the box on your CSV or a synthetic demo series.

Usage
-----
1) Install deps (Python 3.9+ recommended):
   pip install numpy scipy pandas matplotlib

2) Run on your own prices CSV with columns: Date, Close
   python hsmm_regime.py --csv path/to/prices.csv --date-col Date --price-col Close \
       --states 2 --dmax 60 --iters 15

3) Or run with a synthetic demo time series:
   python hsmm_regime.py --demo

Outputs
- Prints learned Gaussian parameters, Poisson duration parameters, and
  summary of decoded regimes.
- Saves `hsmm_regimes.csv` with columns: Date, Return, state, duration_left.

Notes
- This is a reference, transparent implementation prioritizing clarity over speed.
- You can switch the duration family (e.g., Negative Binomial/Weibull) by
  swapping the log-pmf in `_log_duration_pmf` and updating the M-step.
"""

import argparse
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import poisson, norm

# ---------------------------
# Data utilities
# ---------------------------

def generate_demo_prices(n: int = 2000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Two-regime rough volatility synthetic prices
    # Regime 0: low vol, Regime 1: high vol, with explicit durations
    p = 100.0
    prices = []
    dates = pd.date_range('2012-01-02', periods=n, freq='B')

    t = 0
    while t < n:
        # Sample a regime and duration
        state = 0 if rng.random() < 0.6 else 1
        lam = 30 if state == 0 else 8
        dur = max(1, poisson(mu=lam).rvs(random_state=rng))
        dur = int(min(dur, n - t))
        mu = 0.0002 if state == 0 else -0.00005
        sigma = 0.006 if state == 0 else 0.02
        for _ in range(dur):
            r = rng.normal(mu, sigma)
            p *= math.exp(r)
            prices.append(p)
        t += dur

    return pd.DataFrame({'Date': dates[:len(prices)], 'Close': prices})


def load_prices(csv_path: str, date_col: str, price_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if date_col not in df.columns or price_col not in df.columns:
        raise ValueError(f"CSV must contain columns '{date_col}' and '{price_col}'.")
    df = df[[date_col, price_col]].dropna()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df.rename(columns={date_col: 'Date', price_col: 'Close'}, inplace=True)
    return df


def compute_returns(df: pd.DataFrame) -> pd.Series:
    r = np.log(df['Close']).diff().dropna()
    return r

# ---------------------------
# HSMM via duration-expanded Viterbi (hard EM)
# ---------------------------

@dataclass
class HSMMParams:
    mu: np.ndarray      # shape (K,)
    sigma: np.ndarray   # shape (K,) > 0
    lam: np.ndarray     # Poisson duration lambda per state, shape (K,)
    A: np.ndarray       # transition probs between states, no self-transitions, shape (K,K)
    pi: np.ndarray      # initial probs over states, shape (K,)


def _init_params(x: np.ndarray, K: int, seed: int = 0) -> HSMMParams:
    rng = np.random.default_rng(seed)
    # K-means++-ish init for means
    centers = rng.choice(x, size=K, replace=False)
    mu = np.sort(centers)
    # Split by nearest center
    assign = np.argmin(np.abs(x[:, None] - mu[None, :]), axis=1)
    sigma = np.array([x[assign == k].std() if np.any(assign == k) else x.std() for k in range(K)])
    sigma[sigma <= 1e-6] = np.median(sigma[sigma > 0]) if np.any(sigma > 0) else 1e-3
    lam = np.linspace(10, 30, K)  # rough duration priors
    A = np.ones((K, K)) - np.eye(K)
    A = A / A.sum(axis=1, keepdims=True)  # uniform over other states
    pi = np.ones(K) / K
    return HSMMParams(mu=mu, sigma=sigma, lam=lam, A=A, pi=pi)


def _log_emission(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    # log N(x | mu, sigma^2)
    return norm.logpdf(x, loc=mu, scale=sigma)


def _cumulative_loglik(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    # prefix sums for efficient segment log-likelihoods
    ll = _log_emission(x, mu, sigma)
    csum = np.cumsum(ll)
    return csum


def _segment_loglik(csum: np.ndarray, start: int, end: int) -> float:
    # log-likelihood of x[start:end] inclusive
    if start == 0:
        return float(csum[end])
    else:
        return float(csum[end] - csum[start - 1])


def _log_duration_pmf(d: int, lam: float) -> float:
    # Poisson(d) but support starts at 1 (semi-Markov segment length >= 1),
    # so we use P(D=d) = Poisson(d) / (1 - Poisson(0)) = Poisson(d) / (1 - e^{-lam})
    if d < 1:
        return -np.inf
    base = poisson.logpmf(d, mu=lam)
    logZ = np.log1p(-math.exp(-lam))  # log(1 - e^{-lam})
    return base - logZ


def hsmm_viterbi(x: np.ndarray, params: HSMMParams, dmax: int) -> Tuple[np.ndarray, np.ndarray]:
    """Semi-Markov Viterbi decoding with explicit durations up to dmax.
    Returns (state_sequence, durations_left_per_timestep).
    """
    T = len(x)
    K = len(params.mu)

    # Precompute cumulative log-likelihoods for each state
    cums = [ _cumulative_loglik(x, params.mu[k], params.sigma[k]) for k in range(K) ]

    # DP arrays
    V = -np.inf * np.ones((T, K))   # best log score ending at t in state k
    D = np.zeros((T, K), dtype=int) # chosen duration for segment ending at t in state k
    P = -np.ones((T, K), dtype=int) # predecessor state index

    # Initialization: first segment ending at t with duration d in state k
    for t in range(T):
        for k in range(K):
            best = -np.inf
            best_d = 1
            start_min = max(0, t - dmax + 1)
            for s in range(start_min, t + 1):
                d = t - s + 1
                seg_ll = _segment_loglik(cums[k], s, t)
                dur_ll = _log_duration_pmf(d, params.lam[k])
                # initial prob pi_k for first segment only when s == 0
                if s == 0:
                    score = np.log(params.pi[k] + 1e-16) + seg_ll + dur_ll
                else:
                    # must have transitioned from some prev state j != k
                    prev_best = -np.inf
                    for j in range(K):
                        if j == k:
                            continue
                        prev = V[s - 1, j] + np.log(params.A[j, k] + 1e-16)
                        if prev > prev_best:
                            prev_best = prev
                    score = prev_best + seg_ll + dur_ll
                if score > best:
                    best = score
                    best_d = d
            V[t, k] = best
            D[t, k] = best_d
            # P is filled during backtrack since we don't store j here

    # Backtrack
    states = np.zeros(T, dtype=int)
    durations = np.ones(T, dtype=int)
    # pick best final state
    last_state = int(np.argmax(V[T - 1]))
    t = T - 1
    k = last_state
    while t >= 0:
        d = D[t, k]
        # determine predecessor state for this segment
        s = t - d + 1
        if s == 0:
            prev_k = -1
        else:
            # choose best predecessor
            prev_scores = [ V[s - 1, j] + np.log(params.A[j, k] + 1e-16) if j != k else -np.inf for j in range(K) ]
            prev_k = int(np.argmax(prev_scores))
        # fill segment
        states[s:t+1] = k
        # durations_left: countdown to segment end
        c = d
        for tt in range(s, t+1):
            durations[tt] = t - tt + 1
        # move
        t = s - 1
        k = prev_k if prev_k >= 0 else 0

    return states, durations


def m_step_from_path(x: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Update (mu, sigma, lam, A) from decoded state sequence z via hard counts."""
    K = int(z.max()) + 1
    mu = np.zeros(K)
    sigma = np.zeros(K)
    lam = np.zeros(K)
    A_counts = np.zeros((K, K))

    # Emissions
    for k in range(K):
        xs = x[z == k]
        if len(xs) == 0:
            mu[k], sigma[k] = 0.0, 1.0
        else:
            mu[k] = xs.mean()
            sigma[k] = xs.std(ddof=1) if xs.std(ddof=1) > 1e-6 else 1e-3

    # Durations (segment lengths per state)
    seg_lens = {k: [] for k in range(K)}
    t = 0
    T = len(x)
    while t < T:
        k = z[t]
        s = t
        while t + 1 < T and z[t + 1] == k:
            t += 1
        d = t - s + 1
        seg_lens[k].append(d)
        # transition
        if t + 1 < T:
            A_counts[k, z[t + 1]] += 1
        t += 1

    for k in range(K):
        if len(seg_lens[k]) == 0:
            lam[k] = 10.0
        else:
            lam[k] = float(np.mean(seg_lens[k]))  # MLE for Poisson mean

    # Transitions: no self-transitions in semi-Markov core
    for k in range(K):
        A_counts[k, k] = 0
        row = A_counts[k]
        s = row.sum()
        if s == 0:
            # uniform over others
            row[:] = 1.0
            row[k] = 0.0
            s = row.sum()
        A_counts[k] = row / s

    A = A_counts
    return mu, sigma, lam, A


def fit_hsmm(x: np.ndarray, K: int = 2, dmax: int = 60, iters: int = 15, seed: int = 0) -> Tuple[HSMMParams, np.ndarray, np.ndarray]:
    params = _init_params(x, K, seed=seed)
    # pi from first decoded state freq at init (rough)
    params.pi = np.ones(K) / K

    for it in range(1, iters + 1):
        z, dur_left = hsmm_viterbi(x, params, dmax)
        mu, sigma, lam, A = m_step_from_path(x, z)
        params.mu, params.sigma, params.lam, params.A = mu, sigma, lam, A
        # Update pi from first-state counts in segments
        first_state = z[0]
        pi = np.zeros(K)
        pi[first_state] = 1.0
        params.pi = 0.5 * params.pi + 0.5 * pi  # smooth
        # Diagnostics
        ll = np.mean([_log_emission(x[t], params.mu[z[t]], params.sigma[z[t]]) for t in range(len(x))])
        print(f"Iter {it:02d} | mu={params.mu.round(5)} | sigma={params.sigma.round(5)} | lam={params.lam.round(2)} | avg loglik per obs ~ {ll:.4f}")

    z, dur_left = hsmm_viterbi(x, params, dmax)
    return params, z, dur_left

# ---------------------------
# Main CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, default=None)
    ap.add_argument('--date-col', type=str, default='Date')
    ap.add_argument('--price-col', type=str, default='Close')
    ap.add_argument('--states', type=int, default=2)
    ap.add_argument('--dmax', type=int, default=60, help='Max duration per regime (days)')
    ap.add_argument('--iters', type=int, default=15)
    ap.add_argument('--demo', action='store_true')
    args = ap.parse_args()

    if args.demo:
        df = generate_demo_prices()
    else:
        if not args.csv:
            raise SystemExit('Provide --csv path or use --demo')
        df = load_prices(args.csv, args.date_col, args.price_col)

    r = compute_returns(df)
    x = r.values.astype(float)

    params, z, dur_left = fit_hsmm(x, K=args.states, dmax=args.dmax, iters=args.iters, seed=7)

    out = pd.DataFrame({
        'Date': df['Date'].iloc[-len(x):].values,
        'Return': x,
        'state': z,
        'duration_left': dur_left,
    })
    out.to_csv('hsmm_regimes.csv', index=False)

    print('\nLearned parameters:')
    print('mu      =', params.mu)
    print('sigma   =', params.sigma)
    print('lambda  =', params.lam)
    print('A (no self-transitions):\n', params.A)
    print('\nSaved decoded regimes to hsmm_regimes.csv')

if __name__ == '__main__':
    main()
