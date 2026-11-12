"""Classical power-flow baselines on a PYPOWER-format case: DC, XB fast-decoupled (FD) steps, Newton-Raphson.

Single-threaded numpy/scipy so wall-clock times compare across methods on one box. Methods (``METHODS``):

- ``dc``           DC power flow (angles; |V| = setpoint at PV/slack, 1.0 at PQ). Taps and phase shifters are modelled
                   (b = 1/(x*tap), shifter injection), as MATPOWER makeBdc; bus shunts are ignored.
- ``dc_fd``        DC, then k XB fast-decoupled iterations (k given).
- ``dc_fd_conv``   DC, then fast-decoupled to max mismatch < 1e-8 pu (cap 50).
- ``nr_cold``      flat-start Newton-Raphson to 1e-8 pu (cap 30), the comparison most papers use.
- ``nr_dc``        Newton-Raphson from the DC solution.

The fast-decoupled step is PYPOWER fdpf order (P-theta half step with B', then Q-V half step with B''), with XB
matrices as in the powerflow-surrogate classical bar: B' is 1/x with no tap, shift or charging; B'' = -imag(Ybus) on PQ
buses (shifters left in). Timing separates matrix build (Ybus, DC and FD matrices and their LU factors) from the
solve (DC solve, iterations, mismatch checks); ``Net`` parsing (ppc -> arrays) is not timed.

Input format of ``run_on_ppc``: see README ("Running your own case").
"""
from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

TOL = 1e-8
NR_MAX_IT = 30
FD_MAX_IT = 50
METHODS = ("dc", "dc_fd", "dc_fd_conv", "nr_cold", "nr_dc")

# PYPOWER column indices (0-based)
BUS_I, BUS_TYPE, PD, QD, GS, BS, VM, VA = 0, 1, 2, 3, 4, 5, 7, 8
GEN_BUS, PG, QG, VG, GEN_STATUS = 0, 1, 2, 5, 7
F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, SHIFT, BR_STATUS = 0, 1, 2, 3, 4, 8, 9, 10


@dataclass
class Net:
    """Arrays of one case (per unit, radians), internal 0-based bus order."""
    n: int
    base: float
    f: np.ndarray
    t: np.ndarray
    r: np.ndarray
    x: np.ndarray
    b: np.ndarray        # total line charging
    tap: np.ndarray      # magnitude, 1.0 where unset
    shift: np.ndarray    # radians
    on: np.ndarray
    ysh: np.ndarray      # bus shunt (Gs + jBs)/baseMVA
    sbus: np.ndarray     # specified injection (Pg - Pd) + j(Qg - Qd), pu
    vset: np.ndarray     # |V| setpoint at PV/slack, 1.0 at PQ
    ref: int
    pv: np.ndarray
    pq: np.ndarray
    vm_label: np.ndarray
    va_label: np.ndarray  # radians


def parse_ppc(ppc: dict, label: dict | None = None) -> Net:
    """PYPOWER case dict -> Net. ``label`` = {"vm": (n,), "va_deg": (n,)} in ppc bus order; default is ppc bus VM/VA."""
    base = float(ppc["baseMVA"])
    bus, gen, br = (np.asarray(ppc[k], float) for k in ("bus", "gen", "branch"))
    n = len(bus)
    pos = {int(i): k for k, i in enumerate(bus[:, BUS_I])}
    typ = bus[:, BUS_TYPE].astype(int).copy()
    g_on = gen[:, GEN_STATUS] > 0
    gb = np.array([pos[int(i)] for i in gen[:, GEN_BUS]], int)
    pg, qg, has_gen = np.zeros(n), np.zeros(n), np.zeros(n, bool)
    vg = np.ones(n)
    for k in np.where(g_on)[0]:
        pg[gb[k]] += gen[k, PG]
        qg[gb[k]] += gen[k, QG]
        if not has_gen[gb[k]]:
            vg[gb[k]] = gen[k, VG]
        has_gen[gb[k]] = True
    typ[(typ == 2) & ~has_gen] = 1                      # PV bus without an in-service generator is PQ (PYPOWER)
    refs = np.where(typ == 3)[0]
    assert len(refs) == 1, f"expected one slack bus, got {len(refs)}"
    tap = br[:, TAP].copy()
    tap[tap == 0] = 1.0
    lab_vm = np.asarray(label["vm"], float) if label else bus[:, VM]
    lab_va = np.radians(np.asarray(label["va_deg"], float)) if label else np.radians(bus[:, VA])
    return Net(
        n=n, base=base,
        f=np.array([pos[int(i)] for i in br[:, F_BUS]], int), t=np.array([pos[int(i)] for i in br[:, T_BUS]], int),
        r=br[:, BR_R], x=br[:, BR_X], b=br[:, BR_B], tap=tap, shift=np.radians(br[:, SHIFT]), on=br[:, BR_STATUS] > 0,
        ysh=(bus[:, GS] + 1j * bus[:, BS]) / base,
        sbus=((pg - bus[:, PD]) + 1j * (qg - bus[:, QD])) / base,
        vset=np.where(typ == 1, 1.0, vg), ref=int(refs[0]), pv=np.where(typ == 2)[0], pq=np.where(typ == 1)[0],
        vm_label=lab_vm, va_label=lab_va)


class Problem:
    """Matrices (built on demand, timed by the caller) and the iteration kernels for one Net."""

    def __init__(self, net: Net):
        self.net = net
        self.pvpq = np.r_[net.pv, net.pq]

    # ---- builds ----
    def build_ybus(self):
        """Bus admittance matrix with taps, shifters, line charging and bus shunts (PYPOWER makeYbus)."""
        m = self.net
        on = m.on
        ys = 1.0 / (m.r[on] + 1j * m.x[on])
        bc = m.b[on]
        tp = m.tap[on] * np.exp(1j * m.shift[on])
        ytt = ys + 1j * bc / 2
        yff = ytt / (tp * np.conj(tp))
        yft = -ys / np.conj(tp)
        ytf = -ys / tp
        f, t, n = m.f[on], m.t[on], m.n
        ii = np.arange(n)
        self.ybus = sp.csr_matrix((np.r_[yff, yft, ytf, ytt, m.ysh], (np.r_[f, f, t, t, ii], np.r_[f, t, f, t, ii])),
                                  shape=(n, n))

    def build_dc(self):
        """DC B matrix over in-service branches (b = 1/(x*tap)), shifter injection, LU of the slack-reduced matrix."""
        m = self.net
        n, on = m.n, m.on
        b = 1.0 / (m.x[on] * m.tap[on])
        f, t = m.f[on], m.t[on]
        bb = sp.coo_matrix((np.r_[b, b, -b, -b], (np.r_[f, t, f, t], np.r_[f, t, t, f])), shape=(n, n)).tocsr()
        self.pinj = np.bincount(f, -b * m.shift[on], n) + np.bincount(t, b * m.shift[on], n)
        self.keep = np.r_[0:m.ref, m.ref + 1:n]
        self.bref = np.asarray(bb[self.keep, :][:, [m.ref]].todense()).ravel()
        self.dc_lu = spla.splu(bb[self.keep, :][:, self.keep].tocsc())

    def build_fd(self):
        """XB fast-decoupled matrices and LU factors: B' (1/x, PV+PQ rows) and B'' (-imag Ybus, PQ rows)."""
        m = self.net
        on = m.on
        n = m.n
        f, t, bx = m.f[on], m.t[on], 1.0 / m.x[on]
        bp = sp.coo_matrix((np.r_[bx, bx, -bx, -bx], (np.r_[f, t, f, t], np.r_[f, t, t, f])), shape=(n, n)).tocsr()
        self.bp_lu = spla.splu(bp[self.pvpq, :][:, self.pvpq].tocsc())
        self.bpp_lu = spla.splu((-self.ybus.imag)[m.pq, :][:, m.pq].tocsc()) if len(m.pq) else None

    # ---- kernels ----
    def flat(self):
        """Flat start: |V| = setpoint (PV/slack) or 1.0 (PQ), all angles 0 except the slack at its label angle."""
        m = self.net
        va = np.zeros(m.n)
        va[m.ref] = m.va_label[m.ref]
        return m.vset * np.exp(1j * va)

    def dc_solve(self):
        """DC angles from the factorised B: B theta = P - Pinj, slack angle held at its label value."""
        m = self.net
        va = np.zeros(m.n)
        va[m.ref] = m.va_label[m.ref]
        va[self.keep] = self.dc_lu.solve((m.sbus.real - self.pinj)[self.keep] - self.bref * va[m.ref])
        return m.vset * np.exp(1j * va)

    def mismatch(self, v):
        """(P mismatch at PV+PQ, Q mismatch at PQ) = Sbus(V) - Sspec."""
        ds = v * np.conj(self.ybus @ v) - self.net.sbus
        return ds[self.pvpq].real, ds[self.net.pq].imag

    def maxmis(self, v):
        p, q = self.mismatch(v)
        return float(max(np.abs(p).max(initial=0), np.abs(q).max(initial=0)))

    def fd_step(self, v):
        """One XB fast-decoupled iteration (PYPOWER fdpf order): angle half step, then magnitude half step."""
        m = self.net
        va, vm = np.angle(v), np.abs(v)
        p, _ = self.mismatch(v)
        va[self.pvpq] -= self.bp_lu.solve(p / vm[self.pvpq])
        v = vm * np.exp(1j * va)
        if self.bpp_lu is not None:
            _, q = self.mismatch(v)
            vm[m.pq] -= self.bpp_lu.solve(q / vm[m.pq])
            v = vm * np.exp(1j * va)
        return v

    def fd_steps(self, v, k):
        """Exactly k fast-decoupled iterations (no convergence check)."""
        for _ in range(k):
            v = self.fd_step(v)
        return v, k

    def fd_until(self, v, tol=TOL, cap=FD_MAX_IT):
        """Fast-decoupled iterations until max mismatch < tol (checked before each step), at most cap."""
        it = 0
        while it < cap and self.maxmis(v) >= tol:
            v = self.fd_step(v)
            it += 1
            if not np.all(np.isfinite(v)):
                break
        return v, it

    def _jac_plan(self):
        """Index plan for the Jacobian: entry positions of Ybus (plus diagonal) mapped into the reduced matrix."""
        m = self.net
        n, k = m.n, len(self.pvpq)
        y = self.ybus
        self._yrows = np.repeat(np.arange(n), np.diff(y.indptr))
        R = np.r_[self._yrows, np.arange(n)]
        Cc = np.r_[y.indices, np.arange(n)]
        pa = -np.ones(n, int)
        pa[self.pvpq] = np.arange(k)
        qm = -np.ones(n, int)
        qm[m.pq] = np.arange(len(m.pq))
        self._blocks = []
        for rmap, cmap, ro, co, part in ((pa, pa, 0, 0, "a_re"), (pa, qm, 0, k, "m_re"),
                                         (qm, pa, k, 0, "a_im"), (qm, qm, k, k, "m_im")):
            sel = (rmap[R] >= 0) & (cmap[Cc] >= 0)
            self._blocks.append((sel, rmap[R][sel] + ro, cmap[Cc][sel] + co, part))
        self._jshape = (k + len(m.pq),) * 2

    def jacobian(self, v):
        """Polar Newton Jacobian [[dP/dva, dP/dvm], [dQ/dva, dQ/dvm]] on (PV+PQ angles, PQ magnitudes), built from the
        entries of Ybus: dS/dVa = j diag(V) conj(diag(I) - Y diag(V)), dS/dVm = diag(V) conj(Y diag(V/|V|)) + conj(diag(I)) diag(V/|V|)."""
        if not hasattr(self, "_blocks"):
            self._jac_plan()
        y = self.ybus
        ib = y @ v
        yc = np.conj(y.data) * v[self._yrows]                      # conj(Y_ij) * V_i
        vn = v / np.abs(v)
        dva = np.r_[-1j * yc * np.conj(v)[y.indices], 1j * v * np.conj(ib)]
        dvm = np.r_[yc * np.conj(vn)[y.indices], np.conj(ib) * vn]
        data, rows, cols = [], [], []
        for sel, r_, c_, part in self._blocks:
            src = dva if part[0] == "a" else dvm
            data.append((src.real if part.endswith("re") else src.imag)[sel])
            rows.append(r_)
            cols.append(c_)
        return sp.coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                             shape=self._jshape).tocsc()

    def newton(self, v0, tol=TOL, max_it=NR_MAX_IT):
        """Newton-Raphson from v0 to max mismatch < tol. Returns (v, iterations)."""
        m = self.net
        va, vm = np.angle(v0), np.abs(v0)
        v = v0.copy()
        k = len(self.pvpq)
        it = 0
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            f = np.r_[self.mismatch(v)]  # tuple -> concatenated (P, Q)
            while np.abs(f).max(initial=0) >= tol and it < max_it:
                it += 1
                try:
                    dx = spla.spsolve(self.jacobian(v), -f)
                except Exception:
                    break
                va[self.pvpq] += dx[:k]
                vm[m.pq] += dx[k:]
                v = vm * np.exp(1j * va)
                f = np.r_[self.mismatch(v)]
                if not np.all(np.isfinite(f)):
                    break
        return v, it


def _converged(p: Problem, v) -> bool:
    with np.errstate(all="ignore"):
        return bool(np.all(np.isfinite(v)) and np.abs(v).min() >= 0.5 and p.maxmis(v) < TOL)


def _once(net: Net, method: str, k: int | None):
    """One timed pass. Returns (v, converged, iters, t_build_s, t_solve_s). v is None if a factorisation fails."""
    p = Problem(net)
    pc = time.perf_counter
    try:
        t0 = pc()
        if method in ("dc_fd", "dc_fd_conv", "nr_cold", "nr_dc"):
            p.build_ybus()
        if method in ("dc", "dc_fd", "dc_fd_conv", "nr_dc"):
            p.build_dc()
        if method in ("dc_fd", "dc_fd_conv"):
            p.build_fd()
        t1 = pc()
        with np.errstate(all="ignore"):
            if method == "dc":
                v, it = p.dc_solve(), 0
            elif method == "dc_fd":
                v, it = p.fd_steps(p.dc_solve(), k)
            elif method == "dc_fd_conv":
                v, it = p.fd_until(p.dc_solve())
            elif method == "nr_cold":
                v, it = p.newton(p.flat())
            elif method == "nr_dc":
                v, it = p.newton(p.dc_solve())
            else:
                raise ValueError(method)
        t2 = pc()
    except RuntimeError:  # singular matrix (islanded network)
        return None, False, 0, 0.0, 0.0
    if method == "dc":
        # a DC angle solution has no AC mismatch to test; "converged" means the solve succeeded
        return v, bool(np.all(np.isfinite(v))), it, t1 - t0, t2 - t1
    q = Problem(net)
    q.build_ybus()
    return v, _converged(q, v), it, t1 - t0, t2 - t1


def score(net: Net, v, tol_vm=1e-4, tol_va_deg=1e-2) -> dict:
    """Errors against the label: |V| (pu) and angle (deg, both referenced to the slack bus); same_root flag."""
    if v is None or not np.all(np.isfinite(v)):
        return dict(vm_mae=np.nan, va_mae_deg=np.nan, vm_max=np.nan, va_max_deg=np.nan, same_root=False)
    dvm = np.abs(np.abs(v) - net.vm_label)
    dva = np.degrees(np.angle(np.exp(1j * ((np.angle(v) - np.angle(v)[net.ref]) - (net.va_label - net.va_label[net.ref])))))
    dva = np.abs(dva)
    return dict(vm_mae=float(dvm.mean()), va_mae_deg=float(dva.mean()), vm_max=float(dvm.max()),
                va_max_deg=float(dva.max()), same_root=bool(dvm.max() < tol_vm and dva.max() < tol_va_deg))


def label_mismatch(net: Net) -> float:
    """Max power mismatch (pu) of the label state itself in this network (sanity check of the label and the parse)."""
    p = Problem(net)
    p.build_ybus()
    return p.maxmis(net.vm_label * np.exp(1j * net.va_label))


def run_method(net: Net, method: str, k: int | None = None, reps: int = 3) -> dict:
    """Run one method ``reps`` times; result of the first pass, median build and solve time (ms)."""
    outs = [_once(net, method, k) for _ in range(reps)]
    v, conv, it, _, _ = outs[0]
    s = score(net, v)
    s["same_root"] = bool(s["same_root"] and conv)
    return dict(method=method, k=k, converged=conv, iters=it, v=v,
                t_build_ms=1e3 * float(np.median([o[3] for o in outs])),
                t_solve_ms=1e3 * float(np.median([o[4] for o in outs])), **s)


def run_on_ppc(ppc: dict, label: dict | None = None, ks=(1, 2, 5, 10), reps: int = 3) -> list[dict]:
    """Run every classical method on one PYPOWER-format case. Returns one dict per (method, k) with errors against
    the label (default: the ppc's own bus VM/VA), converged, iters, same_root, t_build_ms, t_solve_ms, load_avg."""
    net = parse_ppc(ppc, label)
    jobs = [("dc", None)] + [("dc_fd", k) for k in ks] + [("dc_fd_conv", None), ("nr_cold", None), ("nr_dc", None)]
    rows = []
    for m, k in jobs:
        r = run_method(net, m, k, reps)
        r.pop("v")
        r["load_avg"] = os.getloadavg()[0]
        rows.append(r)
    return rows


def _worker(args):
    net, method, k = args
    _once(net, method, k)


def throughput(nets: list[Net], method: str, k: int | None = None, procs: int = 1) -> float:
    """Samples per second for ``method`` over ``nets`` (build + solve per sample) with ``procs`` worker processes.
    Pool start-up and a warm-up pass are excluded."""
    jobs = [(n, method, k) for n in nets]
    if procs == 1:
        for j in jobs[:5]:
            _worker(j)
        t0 = time.perf_counter()
        for j in jobs:
            _worker(j)
        return len(jobs) / (time.perf_counter() - t0)
    import multiprocessing as mp
    with mp.get_context("fork").Pool(procs) as pool:
        pool.map(_worker, jobs[:procs * 2])
        t0 = time.perf_counter()
        pool.map(_worker, jobs, chunksize=max(1, len(jobs) // (4 * procs)))
        return len(jobs) / (time.perf_counter() - t0)
