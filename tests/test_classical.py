import os

import numpy as np
import pytest

import pfsb  # noqa: F401  (threads pinned before numpy-heavy work)
from pfsb.baselines import classical as C
from pfsb.data import pfdelta

pytestmark = pytest.mark.skipif(not os.environ.get("PFSB_PFDELTA_ROOTS"), reason="PFSB_PFDELTA_ROOTS not set")


def test_newton_matches_pfdelta_label_case14():
    ids, ppcs = pfdelta.load_test_set("case14", "n", cache_dir=os.environ.get("PFSB_CACHE", "cache"), max_samples=5)
    assert len(ppcs) == 5
    for ppc in ppcs:
        net = C.parse_ppc(ppc)
        assert C.label_mismatch(net) < 1e-6          # label is a power-flow solution of this network
        for method in ("nr_cold", "nr_dc"):
            r = C.run_method(net, method, reps=1)
            assert r["converged"]
            assert r["vm_max"] < 1e-6 and r["va_max_deg"] < 1e-6 * 180 / np.pi, (method, r["vm_max"], r["va_max_deg"])
