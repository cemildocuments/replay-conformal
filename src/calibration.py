"""Conformal thresholds for bounded replication of a fixed calibration pool."""
import math
import numpy as np


def conformal_rank(n, alpha):
    if n < 1 or not 0 < alpha < 1:
        raise ValueError('Positive calibration size and alpha in (0, 1) required.')
    # Prevent a representational epsilon from changing an exact integer rank.
    return int(math.ceil(np.nextafter((n + 1) * (1 - alpha), -np.inf)))


def rank_floor(k, n, total, cap):
    if not (1 <= k <= total and n <= total <= n * cap):
        raise ValueError('Inconsistent rank or replication metadata.')
    return max((k + cap - 1) // cap, n - total + k)


def replay_rank(n, total, cap, alpha):
    """Smallest replay-row rank that dominates the clean conformal rank."""
    if not (n >= 1 and cap >= 1 and n <= total <= n * cap):
        raise ValueError('Inconsistent replication metadata.')
    r = conformal_rank(n, alpha)
    if r > n:
        return total + 1
    return min(cap * (r - 1) + 1, total - n + r)


def order_statistic(scores, k):
    scores = np.asarray(scores, dtype=float)
    if k > len(scores):
        return float('inf')
    if k < 1 or not np.isfinite(scores).all():
        raise ValueError('Scores must be finite and rank positive.')
    return float(np.partition(scores, k - 1)[k - 1])


def thresholds(scores, counts, alpha, cap):
    scores, counts = np.asarray(scores), np.asarray(counts, dtype=int)
    n, total = len(scores), int(counts.sum())
    if len(counts) != n or np.any(counts < 1) or np.any(counts > cap):
        raise ValueError('Each source record must appear between one and cap times.')
    rows = np.repeat(scores, counts)
    r = conformal_rank(n, alpha)
    # ESS plug-in is a deliberately labelled heuristic comparator.
    neff = total ** 2 / np.square(counts).sum()
    ess_level = min(1.0, (1 - alpha) * (neff + 1) / neff)
    kess = int(math.ceil(np.nextafter(total * ess_level, -np.inf)))
    out = {
        'row': order_statistic(rows, conformal_rank(total, alpha)),
        'dedup': order_statistic(scores, r),
        'ess': order_statistic(rows, kess),
        'count': order_statistic(rows, total - n + r),
        'replay': order_statistic(rows, replay_rank(n, total, cap, alpha)),
    }
    return out


def replay_counts(scores, total, cap, mechanism, rng):
    n = len(scores)
    if not n <= total <= n * cap:
        raise ValueError('Infeasible total replay budget.')
    counts = np.ones(n, dtype=np.int64)
    extra = int(total - n)
    if mechanism == 'balanced':
        q, r = divmod(extra, n)
        counts += q
        if r:
            counts[rng.permutation(n)[:r]] += 1
    elif mechanism in ['low', 'high']:
        order = np.argsort(scores, kind='stable')
        if mechanism == 'high':
            order = order[::-1]
        for i in order:
            take = min(cap - 1, extra)
            counts[i] += take
            extra -= take
            if extra == 0:
                break
    elif mechanism == 'random':
        # Uniform subset of the n*(cap-1) optional replay slots.
        slots = rng.choice(n * (cap - 1), size=extra, replace=False)
        counts += np.bincount(slots // (cap - 1), minlength=n)
    else:
        raise ValueError(mechanism)
    assert counts.sum() == total and counts.max() <= cap
    return counts
