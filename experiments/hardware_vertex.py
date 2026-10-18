"""Per delay: incentive from the full quadratic-form test versus the test restricted to C and D, and the
Q-component of the best deviation (hardware data, readout mitigated)."""
import sys
import numpy as np
import common as C
import hardware_analyze as H
from ewlcert import game as G
from ewlcert import stats as S

if __name__ == "__main__":
    out = []
    for f_name in sys.argv[1:]:
        rec = C.load(f_name)
        data = H.by_delay(rec)
        for f in rec["fractions"]:
            g, c0, c1 = data[f]
            P = H.probs(g, (c0, c1))
            forms = S.forms_from_probs(P)
            qc = []
            for M in forms.values():
                val, x = G.max_on_orthant(M)
                qc.append(float(x[2]) if val - M[2, 2] > 1e-9 else 1.0)
            out.append(dict(file=f_name, tau_us=f * rec["tc_pred"] * 1e6, eps=S.eps_from_probs(P),
                            eps_vertex=S.vertex_eps_from_probs(P), dgain=S.mean_D_gain(P),
                            offdiag=S.max_offdiag(P), q_component_best=min(qc)))
            r = out[-1]
            print(f"{f_name[-11:-5]} tau={r['tau_us']:6.1f} eps={r['eps']:.3f} eps_CD={r['eps_vertex']:.3f} "
                  f"Dgain={r['dgain']:+.3f} offdiag={r['offdiag']:.3f} Qcomp={r['q_component_best']:.2f}")
    C.save("hardware_vertex.json", out)
