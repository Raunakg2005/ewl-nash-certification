"""Scheduled durations (without measurement) of the 11 certification circuits at zero idle delay."""
import warnings
import common as C
from ewlcert import stats as S

warnings.filterwarnings("ignore")
if __name__ == "__main__":
    out = {}
    for name in C.BACKENDS:
        be = C.Dv.backend_by_name(name)
        picks, _ = C.select_pairs(be)
        for p in picks:
            dp = C.Dv.DevicePair(be, p["pair"])
            d = [dp.scheduled(a, b, 0.0).estimate_duration(be.target, unit="s") for a, b in S.circuit_strategies()]
            out[f"{name}:{p['label']}"] = max(d)
    C.save("durations.json", out)
    print(min(out.values()), max(out.values()))
