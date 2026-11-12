"""Export datakit-format scenarios as PYPOWER cases (external format, 1-based bus ids).

Arrays per dataset, stacked over samples: bus [S,nb,13], gen [S,ng,21], branch [S,nl,13],
label_vm [S,nb], label_va_deg [S,nb] (as in the raw labels; slack at its reference angle),
baseMVA, sample_ids [S], condition [S] (str). Out-of-service branches/generators keep their row
with status 0. Bus row i has bus_i = i+1, so bus_i - 1 indexes the label arrays.
Bus Vm/Va columns hold a flat start (1.0, 0); generator Vg holds the label Vm of its bus.
"""
import numpy as np


def export_scenarios(raw_tables, scen_ids, base_mva):
    bus_df, gen_df, br_df = raw_tables
    S = len(scen_ids)
    bus_g = dict(tuple(bus_df[bus_df.scenario.isin(scen_ids)].groupby("scenario")))
    gen_g = dict(tuple(gen_df[gen_df.scenario.isin(scen_ids)].groupby("scenario")))
    br_g = dict(tuple(br_df[br_df.scenario.isin(scen_ids)].groupby("scenario")))
    nb, ng, nl = len(bus_g[scen_ids[0]]), len(gen_g[scen_ids[0]]), len(br_g[scen_ids[0]])
    bus = np.zeros((S, nb, 13)); gen = np.zeros((S, ng, 21)); br = np.zeros((S, nl, 13))
    lvm = np.zeros((S, nb)); lva = np.zeros((S, nb)); cond = []
    for k, s in enumerate(scen_ids):
        b = bus_g[s].sort_values("bus"); g = gen_g[s].sort_values("idx"); l = br_g[s].sort_values("idx")
        typ = np.where(b.REF.values == 1, 3, np.where(b.PV.values == 1, 2, 1))
        bus[k, :, 0] = b.bus.values + 1; bus[k, :, 1] = typ
        bus[k, :, 2] = b.Pd.values; bus[k, :, 3] = b.Qd.values
        bus[k, :, 4] = b.GS.values * base_mva; bus[k, :, 5] = b.BS.values * base_mva
        bus[k, :, 6] = 1; bus[k, :, 7] = 1.0; bus[k, :, 8] = 0.0; bus[k, :, 9] = b.vn_kv.values
        bus[k, :, 10] = 1; bus[k, :, 11] = b.max_vm_pu.values; bus[k, :, 12] = b.min_vm_pu.values
        lvm[k] = b.Vm.values; lva[k] = b.Va.values
        gb = g.bus.values.astype(int)
        gen[k, :, 0] = gb + 1; gen[k, :, 1] = g.p_mw.values; gen[k, :, 2] = g.q_mvar.values
        gen[k, :, 3] = g.max_q_mvar.values; gen[k, :, 4] = g.min_q_mvar.values
        gen[k, :, 5] = lvm[k][gb]; gen[k, :, 6] = base_mva; gen[k, :, 7] = g.in_service.values
        gen[k, :, 8] = g.max_p_mw.values; gen[k, :, 9] = g.min_p_mw.values
        br[k, :, 0] = l.from_bus.values + 1; br[k, :, 1] = l.to_bus.values + 1
        br[k, :, 2] = l.r.values; br[k, :, 3] = l.x.values; br[k, :, 4] = l.b.values
        br[k, :, 5] = l.rate_a.values
        br[k, :, 8] = l.tap.values; br[k, :, 9] = l.shift.values; br[k, :, 10] = l.br_status.values
        br[k, :, 11] = l.ang_min.values; br[k, :, 12] = l.ang_max.values
        bo = np.where(l.br_status.values == 0)[0]; go = np.where(g.in_service.values == 0)[0]
        parts = [f"branch_out={int(i)}" for i in bo] + [f"gen_out={int(i)}" for i in go]
        cond.append(";".join(parts) if parts else "base")
    return dict(bus=bus, gen=gen, branch=br, label_vm=lvm, label_va_deg=lva,
                baseMVA=np.array(base_mva), sample_ids=np.array(scen_ids), condition=np.array(cond))


def max_mismatch(case, k):
    """Max |S mismatch| (MVA) at non-slack buses for sample k at its label voltage, using pandapower's
    PYPOWER routines (Ybus/Sbus). A self-check that the exported case is consistent with its label."""
    from pandapower.pypower.makeYbus import makeYbus
    from pandapower.pypower.makeSbus import makeSbus
    from pandapower.pypower.idx_bus import BUS_TYPE
    bus = case["bus"][k].copy(); gen = case["gen"][k]; br = case["branch"][k].copy(); bm = float(case["baseMVA"])
    br[br[:, 10] == 0] = br[br[:, 10] == 0]  # status handled by makeYbus
    ids = bus[:, 0].astype(int) - 1
    bus_i = bus.copy(); bus_i[:, 0] = np.arange(len(bus)); gen_i = gen.copy(); gen_i[:, 0] -= 1
    br_i = br.copy(); br_i[:, 0] -= 1; br_i[:, 1] -= 1
    Ybus, _, _ = makeYbus(bm, bus_i, br_i)
    Sbus = makeSbus(bm, bus_i, gen_i)
    V = case["label_vm"][k] * np.exp(1j * np.deg2rad(case["label_va_deg"][k]))
    mis = V * np.conj(Ybus @ V) - Sbus
    keep = bus[:, 1] != 3
    return float(np.abs(mis[keep]).max() * bm)
