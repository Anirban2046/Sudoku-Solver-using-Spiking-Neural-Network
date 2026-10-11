"""Post-run analysis of a Sudoku SNN simulation.

``AnalysisAccumulator`` collects per-window data during the run (``add_window``)
and turns it into one JSON-serialisable report at the end (``finish``). Every
number the frontend's analysis report shows comes from here or from ``meta``.
"""

import math
from collections import deque

import numpy as np

from .helpers import PEER_PAIRS, solve_all


def count_violations(grid):
    """Number of unordered cell pairs sharing a row, column or box that show
    the same digit (the "energy" of the readout; 0 for a valid grid)."""
    g = np.asarray(grid).reshape(81)
    return int((g[PEER_PAIRS[:, 0]] == g[PEER_PAIRS[:, 1]]).sum())


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


def _inhibition_breakdown():
    """Target populations of one population, split by constraint type."""
    return {"row": 8, "column": 8, "box_only": 4, "same_cell": 8, "total": 8 + 8 + 4 + 8}


def neuron_quantities(meta):
    """Derived single-neuron quantities (Appendix 1 of the build guide),
    computed from the actual parameters in ``meta``."""
    p = meta["neuron_params"]
    w = meta["weights"]
    C_m, I_e, tau_m, t_ref = p["C_m"], p["I_e"], p["tau_m"], p["t_ref"]
    E_L, V_th, V_reset = p["E_L"], p["V_th"], p["V_reset"]
    tau_s = p["tau_syn_ex"]

    V_inf = E_L + I_e * tau_m / C_m
    if V_inf > V_th:
        T = tau_m * math.log((V_inf - V_reset) / (V_inf - V_th))
        rate = 1000.0 / (T + t_ref)
    else:
        T, rate = None, 0.0

    pref = tau_m * tau_s / (tau_m - tau_s)
    t_peak = pref * math.log(tau_m / tau_s)

    def psp(weight, t):
        return weight / C_m * pref * (np.exp(-t / tau_m) - np.exp(-t / tau_s))

    kernel_t = np.arange(0, 80.5, 0.5)
    psps = {}
    for name in ("inhibitory", "stimulus", "noise"):
        psps[name] = {
            "weight": w[name],
            "peak_mv": _r(psp(w[name], t_peak), 3),
            "t_peak_ms": _r(t_peak, 3),
            "kernel_mv": [round(float(v), 4) for v in psp(w[name], kernel_t)],
        }

    tau_s_s = tau_s / 1000.0
    return {
        "V_inf_mv": _r(V_inf, 3),
        "V_inf_above_threshold": V_inf > V_th,
        "tonic_isi_ms": _r(T, 3),
        "tonic_rate_hz": _r(rate, 2),
        "psp_prefactor_ms": _r(pref, 4),
        "t_peak_ms": _r(t_peak, 3),
        "psp": psps,
        "kernel_t_ms": kernel_t.tolist(),
        "mean_current_noise": _r(w["noise"] * tau_s_s * meta["noise_rate"], 4),
        "mean_current_stimulus": _r(w["stimulus"] * tau_s_s * meta["stim_rate"], 4),
        "formulas": {
            "V_inf": r"V_\infty = E_L + \frac{I_e\,\tau_m}{C_m}",
            "tonic_isi": r"T = \tau_m \ln\frac{V_\infty - V_{reset}}{V_\infty - V_{th}}",
            "tonic_rate": r"r = \frac{1}{T + t_{ref}}",
            "t_peak": r"t_{peak} = \frac{\tau_m\tau_s}{\tau_m-\tau_s}\ln\frac{\tau_m}{\tau_s}",
            "psp": r"\Delta V(t) = \frac{w}{C_m}\frac{\tau_m\tau_s}{\tau_m-\tau_s}\left(e^{-t/\tau_m} - e^{-t/\tau_s}\right)",
            "mean_current": r"\bar I = w\,\tau_s\,\nu",
        },
    }


class AnalysisAccumulator:
    def __init__(self, puzzle, meta):
        self.puzzle = np.asarray(puzzle, dtype=np.int64).reshape(9, 9)
        self.meta = meta
        self.pop_size = meta["pop_size"]
        self.n_neurons = meta["n_neurons"]
        self.chunk_ms = meta["chunk_ms"]
        self.windows = []
        self.pop_counts = []
        self.neuron_counts = np.zeros(self.n_neurons, dtype=np.int64)
        n_isi = max(1, int(math.ceil(1000.0 / self.chunk_ms)))
        self.recent_spikes = deque(maxlen=n_isi)  # (neuron_idx, times) per window

    def add_window(self, *, k, t_start, t_end, pop_counts, neuron_counts, neuron_idx, times, grid, tie, silent,
                   rows, cols, boxes, performance, cells_changed, givens_kept, violations):
        self.pop_counts.append(np.asarray(pop_counts, dtype=np.int64))
        self.neuron_counts += neuron_counts
        self.recent_spikes.append((np.asarray(neuron_idx), np.asarray(times)))
        self.windows.append({
            "k": k,
            "t_start_ms": t_start,
            "t_end_ms": t_end,
            "grid": np.asarray(grid).flatten().tolist(),
            "performance": round(float(performance), 4),
            "valid_rows": int(np.sum(rows)),
            "valid_cols": int(np.sum(cols)),
            "valid_boxes": int(np.sum(boxes)),
            "invalid_rows": [int(i) + 1 for i in np.flatnonzero(~np.asarray(rows))],
            "invalid_cols": [int(i) + 1 for i in np.flatnonzero(~np.asarray(cols))],
            "invalid_boxes": [int(i) + 1 for i in np.flatnonzero(~np.asarray(boxes).flatten())],
            "cells_changed": int(cells_changed),
            "ties": int(np.sum(tie)),
            "silent": int(np.sum(silent)),
            "violations": int(violations),
            "givens_kept": bool(givens_kept),
            "total_spikes": int(np.sum(pop_counts)),
        })

    # -- sections ---------------------------------------------------------

    def _network(self):
        m = self.meta
        c = m["connections"]
        n = m["n_neurons"]
        return {
            "n_populations": m["n_populations"],
            "pop_size": m["pop_size"],
            "n_neurons": n,
            "connections": c,
            "inhibitory_target_populations": _inhibition_breakdown(),
            "fan_out_inhibitory": c["inhibitory"] // n,
            "fan_in": {
                "inhibitory": c["inhibitory"] // n,
                "noise": c["noise"] // n,
                "stimulus": c["stimulus"] // n,
                "total": (c["inhibitory"] + c["noise"] + c["stimulus"]) // n,
            },
            "weights": m["weights"],
        }

    def _input(self):
        givens = []
        for r in range(9):
            for c in range(9):
                d = int(self.puzzle[r, c])
                if d:
                    givens.append({"row": r + 1, "col": c + 1, "digit": d, "pop": r * 81 + c * 9 + (d - 1),
                                   "rate_hz": self.meta["stim_rate"]})
        return {
            "givens": givens,
            "n_givens": len(givens),
            "stim_rate_hz": self.meta["stim_rate"],
            "noise_rate_hz": self.meta["noise_rate"],
            "stim_weight_given": self.meta["weights"]["stimulus"],
            "stim_weight_other": 0.0,
            "active_stimulus_connections": self.meta["connections"]["stimulus_active"],
        }

    def _firing(self, final_grid):
        counts = np.stack(self.pop_counts)  # (n_windows, 729)
        win_s = self.chunk_ms / 1000.0
        ps = self.pop_size
        winners = np.zeros(729, dtype=bool)
        for i, d in enumerate(final_grid):
            winners[i * 9 + int(d) - 1] = True
        n_win = int(winners.sum()) * ps
        n_lose = int((~winners).sum()) * ps

        rate_all = counts.sum(axis=1) / (self.n_neurons * win_s)
        rate_win = counts[:, winners].sum(axis=1) / (n_win * win_s)
        rate_lose = counts[:, ~winners].sum(axis=1) / (n_lose * win_s)

        last = counts[-1]
        total = counts.sum(axis=0)
        cells_last = last.reshape(81, 9)
        srt = np.sort(cells_last, axis=1)
        margin = (srt[:, -1] - srt[:, -2]) / (srt[:, -1] + 1e-9)

        # ISI coefficient of variation for winning neurons over the last ~1 s
        idx = np.concatenate([s[0] for s in self.recent_spikes])
        tms = np.concatenate([s[1] for s in self.recent_spikes])
        win_neuron = winners[idx // ps]
        idx, tms = idx[win_neuron], tms[win_neuron]
        order = np.lexsort((tms, idx))
        idx, tms = idx[order], tms[order]
        cvs = []
        for n in np.unique(idx):
            t = tms[idx == n]
            if len(t) >= 3:
                isi = np.diff(t)
                if isi.mean() > 0:
                    cvs.append(isi.std() / isi.mean())
        cvs = np.asarray(cvs)
        isi_window = len(self.recent_spikes) * self.chunk_ms

        sim_s = len(self.windows) * win_s
        neuron_rates = self.neuron_counts / sim_s
        hi = float(max(neuron_rates.max(), 1.0))
        hist, edges = np.histogram(neuron_rates, bins=30, range=(0.0, hi))

        return {
            "t_end_ms": [w["t_end_ms"] for w in self.windows],
            "rate_all_hz": [round(float(v), 2) for v in rate_all],
            "rate_winners_hz": [round(float(v), 2) for v in rate_win],
            "rate_losers_hz": [round(float(v), 2) for v in rate_lose],
            "n_winner_neurons": n_win,
            "n_loser_neurons": n_lose,
            "cell_counts_last": cells_last.tolist(),
            "cell_counts_total": total.reshape(81, 9).tolist(),
            "decision_margin": [round(float(v), 4) for v in margin],
            "decision_margin_mean": _r(margin.mean()),
            "isi_cv_mean": _r(cvs.mean()) if len(cvs) else None,
            "isi_cv_median": _r(np.median(cvs)) if len(cvs) else None,
            "isi_cv_neurons": int(len(cvs)),
            "isi_window_ms": isi_window,
            "neuron_rate_hist": {"edges_hz": [round(float(e), 2) for e in edges], "counts": hist.tolist()},
            "total_spikes": int(counts.sum()),
            "mean_rate_hz": _r(counts.sum() / (self.n_neurons * sim_s), 3),
            "max_neuron_rate_hz": _r(neuron_rates.max(), 2),
        }

    def _convergence(self):
        grids = np.array([w["grid"] for w in self.windows])  # (n, 81)
        prev = np.vstack([self.puzzle.reshape(1, 81), grids[:-1]])
        changed = grids != prev
        lock_k = np.zeros(81, dtype=np.int64)
        for c in range(81):
            nz = np.flatnonzero(changed[:, c])
            lock_k[c] = nz[-1] if len(nz) else 0
        lock_ms = (lock_k + 1) * self.chunk_ms
        order = np.lexsort((np.arange(81), lock_ms))
        return {
            "lock_in_ms": lock_ms.tolist(),
            "lock_in_order": order.tolist(),
            "changes_per_cell": changed.sum(axis=0).tolist(),
            "is_given": (self.puzzle.reshape(81) != 0).tolist(),
        }

    def _classical(self, final_grid, solved):
        sols = solve_all(self.puzzle, limit=2)
        n = len(sols)
        classical = sols[0].flatten().tolist() if n else None
        equal = classical is not None and list(final_grid) == classical
        if not solved:
            verdict = "not_solved"
        elif n == 0:
            verdict = "no_classical_solution"
        elif equal:
            verdict = "match"
        else:
            verdict = "valid_but_different"
        return {
            "n_solutions": n,
            "n_solutions_label": {0: "0", 1: "1", 2: "≥ 2"}[n],
            "unique": n == 1,
            "solution": classical,
            "snn_equals_classical": equal,
            "verdict": verdict,
        }

    def finish(self, *, solved, solved_at_ms, wall_seconds_total):
        if not self.windows:
            return {"windows": 0}
        final = self.windows[-1]
        final_grid = final["grid"]
        sim_ms = len(self.windows) * self.chunk_ms
        perf = np.array([w["performance"] for w in self.windows])
        best_k = int(perf.argmax())

        outcome = {
            "solved": bool(solved),
            "solved_at_ms": solved_at_ms,
            "sim_ms_total": sim_ms,
            "wall_seconds_total": _r(wall_seconds_total, 3),
            "speed_ratio_sim_ms_per_wall_s": _r(sim_ms / max(wall_seconds_total, 1e-9), 1),
            "seed": self.meta["seed"],
            "givens_kept": final["givens_kept"],
            "best_performance": float(perf[best_k]),
            "best_at_ms": self.windows[best_k]["t_start_ms"],
            "best_grid": self.windows[best_k]["grid"],
            "final_invalid_rows": final["invalid_rows"],
            "final_invalid_cols": final["invalid_cols"],
            "final_invalid_boxes": final["invalid_boxes"],
            "final_violations": final["violations"],
            "max_sim_ms": self.meta["max_sim_ms"],
            "build_seconds": self.meta["build_seconds"],
            "threads": self.meta["threads"],
            "nest_version": self.meta["nest_version"],
        }
        return {
            "outcome": outcome,
            "network": self._network(),
            "neuron": {
                "model": self.meta["neuron_model"],
                "params": self.meta["neuron_params"],
                "derived": neuron_quantities(self.meta),
            },
            "input": self._input(),
            "readout": {
                "chunk_ms": self.chunk_ms,
                "rule": "argmax over the 9 digit populations of each cell; ties broken uniformly at random",
                "trace": self.windows,
            },
            "firing": self._firing(final_grid),
            "convergence": self._convergence(),
            "classical": self._classical(final_grid, solved),
        }
