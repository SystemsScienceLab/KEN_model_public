# =============================================================================
# Lead model conceptualization and development (theory, methodology, and
# implementation), software architecture & model coding (all components),
# data acquisition & processing, team supervision: Michael Miess
# Initial software architecture development, first coding, technical consulting:
# Joel Foramitti
# Lead of energy modeling: Ansir Ilyas
# Lead of water modeling: Dan Wang
# Model consulting & support: Asjad Naqvi
# Project conceptualization, funding & supervision: Yoshihide Wada
# =============================================================================

"""
Imports without industrial policy: sensitivity run outside the published scenarios.

Reproduces the table "Imports with and without industrial policy" of the Supplementary
Information and the sentence on imports without import substitution in the H1 results.

Method
------
Vision 2030 and Transformation are run twice, with Oil_exports_random_fluctuations off as for
the paper results: once as published, and once with import substitution switched off:

    Endogenous_excess_import_reduction         = False  (also disables the direct substitution
                                                         in the anchor sectors)
    Endogenous_import_reduction_cluster_growth = False

All other scenario settings (oil export path, non-oil export targets, energy transition, taxes,
water) stay the same. The runs are kept in memory, no results file is written. The published
runs are checked against the saved results file (GDP 2060 ratio must be 1.0).
Conventions: t = 1 is 2021, t = 40 is 2060; growth rates over 2021-2061 as in the growth table.

Usage (from the repository root, KEN_model conda environment):
    python -m model.import_counterfactual [saved_results/results_YYYYMMDD_HHMMSS.pkl]

Without an argument the newest file in saved_results is used. Writes
Tables/import_counterfactual.json and Tables/import_counterfactual.md.
"""

import glob
import json
import os
import pickle
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")

from model.experiment_runner import ExperimentRunner

warnings.filterwarnings("ignore")

SCS = ["Vision_2030", "Transformation"]
BASE = {"Oil_exports_random_fluctuations": False}
OFF = dict(BASE, Endogenous_excess_import_reduction=False, Endogenous_import_reduction_cluster_growth=False)


def S(r, n):
    return r["results_macro"][n].apply(lambda x: float(np.sum(x)))


def g(a, t1=1, t2=41):
    return ((a.loc[t2] / a.loc[t1]) ** (1.0 / (t2 - t1)) - 1) * 100


def first_year(cond):
    z = [t for t in cond.index if cond.loc[t]]
    return 2020 + z[0] if z else None


def describe(r):
    Y = S(r, "Y"); IM = S(r, "IM"); Yno = S(r, "Y_non_oil")
    d = {f"imports / GDP {2020 + t} %": round(100 * IM.loc[t] / Y.loc[t], 1) for t in (10, 25, 40)}
    d["cumulative imports / cumulative GDP 2021-2060 %"] = round(100 * IM.loc[1:40].sum() / Y.loc[1:40].sum(), 1)
    d["non-oil share of GDP 2060 %"] = round(100 * Yno.loc[40] / Y.loc[40], 1)
    d["imports / non-oil GDP 2060 %"] = round(100 * IM.loc[40] / Yno.loc[40], 1)
    d["first year imports above non-oil GDP"] = first_year((IM / Yno).loc[1:40] > 1)
    d["GDP growth nominal %"] = round(g(Y), 2)
    d["GDP growth real %"] = round(g(S(r, "y")), 2)
    d["import growth %"] = round(g(IM), 2)
    return d, Y, IM


def main(path=None):
    if path is None:
        path = sorted(glob.glob("saved_results/results_*.pkl"))[-1]
    with open(path, "rb") as f:
        P = pickle.load(f)
    er = ExperimentRunner()
    out = {"file": path}
    for sc in SCS:
        pol = er.single_run(sc, 0, BASE)
        cf = er.single_run(sc, 0, OFF)
        check = S(pol, "Y").loc[40] / S(P[sc][0], "Y").loc[40]
        dp, Yp, IMp = describe(pol)
        dc, Yc, IMc = describe(cf)
        out[sc] = {"with industrial policy": dp, "without industrial policy": dc, "comparison": {
            "check: GDP 2060 published run / saved results": round(float(check), 6),
            "cumulative imports 2021-2060, with vs without %": round(float(100 * (IMp.loc[1:40].sum() / IMc.loc[1:40].sum() - 1)), 1),
            "imports 2025, 2035, 2045 with vs without %": [round(float(100 * (IMp.loc[t] / IMc.loc[t] - 1)), 1) for t in (5, 15, 25)],
            "imports 2060 with vs without %": round(float(100 * (IMp.loc[40] / IMc.loc[40] - 1)), 1),
            "GDP 2060 with / without": round(float(Yp.loc[40] / Yc.loc[40]), 2),
        }}
    os.makedirs("Tables", exist_ok=True)
    with open(os.path.join("Tables", "import_counterfactual.json"), "w", encoding="utf8") as f:
        json.dump(out, f, indent=1)
    L = ["# Imports without industrial policy", "", f"Published runs checked against `{path}`", ""]
    for sc in SCS:
        L += [f"## {sc}", "", "| Quantity | with IP | without IP |", "|---|---|---|"]
        L += [f"| {k} | {v} | {out[sc]['without industrial policy'][k]} |" for k, v in out[sc]["with industrial policy"].items()]
        L += [""] + [f"- {k}: {v}" for k, v in out[sc]["comparison"].items()] + [""]
    with open(os.path.join("Tables", "import_counterfactual.md"), "w", encoding="utf8") as f:
        f.write("\n".join(L))
    print("\n".join(L))
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
