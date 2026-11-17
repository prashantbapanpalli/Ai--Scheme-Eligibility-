"""Compare the rule parser with hand labels in eval/parser_labels.csv."""
import sys, pandas as pd
sys.path.insert(0, ".")
from scheme_finder.parse import parse_eligibility

def score(df, labels):
    out = []
    for _, r in labels.iterrows():
        p = parse_eligibility(df["Eligibility Criteria"][r.row])
        out.append(dict(row=r.row, min_t=r.min_age, min_p=p.min_age, max_t=r.max_age, max_p=p.max_age,
                        inc_t=r.income_cap, inc_p=p.income_cap, w_t=r.women_only, w_p=int(p.women_only)))
    R = pd.DataFrame(out)
    def same(a, b):
        if pd.isna(a) and pd.isna(b): return True
        if pd.isna(a) or pd.isna(b): return False
        return abs(float(a) - float(b)) < 1
    summary = {}
    for f, name in [("min", "min age"), ("max", "max age"), ("inc", "income cap")]:
        t, p = R[f + "_t"], R[f + "_p"]
        ok = [same(a, b) for a, b in zip(t, p)]
        has_t, has_p = t.notna(), p.notna()
        correct_found = sum(o and h for o, h in zip(ok, has_t))
        summary[name] = dict(
            exact_match=f"{sum(ok)}/{len(ok)}",
            precision=f"{correct_found}/{int(has_p.sum())}",   # of values the parser gave, how many right
            recall=f"{correct_found}/{int(has_t.sum())}",      # of real rules, how many found
        )
    tp = int(((R.w_t == 1) & (R.w_p == 1)).sum())
    summary["women only"] = dict(exact_match=f"{int((R.w_t == R.w_p).sum())}/{len(R)}",
                                precision=f"{tp}/{int(R.w_p.sum())}", recall=f"{tp}/{int(R.w_t.sum())}")
    return pd.DataFrame(summary).T, R

if __name__ == "__main__":
    df = pd.read_csv("data/gov_myscheme_data.csv")
    s, R = score(df, pd.read_csv("eval/parser_labels.csv"))
    print(s.to_string())
    def diff(t, q): return [not ((pd.isna(a) and pd.isna(b)) or (not pd.isna(a) and not pd.isna(b) and abs(a - b) < 1)) for a, b in zip(t, q)]
    import numpy as np
    m = np.array(diff(R.inc_t, R.inc_p)) | np.array(diff(R.min_t, R.min_p)) | np.array(diff(R.max_t, R.max_p)) | (R.w_t != R.w_p).values
    print(R[m].to_string())
