# تدريب نهائي للنموذجين المتعلمين وحفظهما للخدمة
# يشغل مرة واحدة (أو عند تحديث الداتا) - والـ Pipeline يحملهما جاهزين

import os
import sys
import csv
import joblib
import numpy as np
from datetime import datetime
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(HERE, "pipeline", "artifacts")
os.makedirs(ART, exist_ok=True)

SEV = ["low", "medium", "high", "critical"]
SEVI = {s: i for i, s in enumerate(SEV)}
SUB = {"full_form": 0, "voice": 1, "sos": 2}
ANCH = {"at_home": 0, "seeing_it": 1, "unknown": 2}
ZONES = ["residential", "commercial", "industrial", "rural"]
TYPES = ["medical", "road", "fire", "police", "rescue", "collapse", "disaster"]

# ---------- 1) القاضي الزوجي ----------
rows = []
with open(os.path.join(HERE, "dataset", "sim_pairs.csv"), encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    PAIR_FEATS = [c for c in reader.fieldnames
                    if c not in ("label", "source")]
    for row in reader:
        rows.append([float(row[c]) for c in PAIR_FEATS]
                    + [int(row["label"])])
data = np.array(rows)
judge = RandomForestClassifier(n_estimators=200, max_depth=8,
                                class_weight="balanced",
                                random_state=42)
judge.fit(data[:, :-1], data[:, -1].astype(int))
joblib.dump({"model": judge, "features": PAIR_FEATS},
            os.path.join(ART, "pairwise_judge.joblib"))
print(f"القاضي الزوجي: تدرب على {len(data)} زوجاً وحفظ")

# ---------- 2) المصداقية ----------
raw = list(csv.DictReader(open(
    os.path.join(HERE, "dataset", "sim_reports.csv"),
    encoding="utf-8-sig")))
inc_users = {}
for r in raw:
    if r["incident_id"]:
        inc_users.setdefault(r["incident_id"],
                                set()).add(r["user_id"])
inc_counts = {k: len(v) for k, v in inc_users.items()}

X, y = [], []
for r in raw:
    ts = datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S")
    inj = -1 if r["injured_count"] == "" else float(r["injured_count"])
    feats = [inc_counts.get(r["incident_id"], 1),
                float(r["user_true_report_ratio"]),
                int(r["is_witness"] == "True"),
                int(r["has_media"] == "True"),
                int(r["injured_unknown"] == "True"),
                inj, ts.hour, ts.weekday(),
                SEVI[r["citizen_severity"]],
                SUB[r["submission_method"]],
                ANCH[r["incident_anchor"]]]
    feats += [int(r["incident_type"] == t) for t in TYPES]
    X.append(feats)
    y.append(1 - int(r["is_fake"]))
cred = RandomForestClassifier(n_estimators=200, max_depth=10,
                                class_weight="balanced",
                                random_state=42)
cred.fit(np.array(X, float), np.array(y))
joblib.dump({"model": cred},
            os.path.join(ART, "credibility.joblib"))
print(f"المصداقية: تدرب على {len(X)} بلاغاً وحفظ")

# ---------- 3) الخطورة ----------
by_inc = {}
for r in raw:
    if r["incident_id"]:
        by_inc.setdefault(r["incident_id"], []).append(r)
X, y = [], []
for inc in csv.DictReader(open(
        os.path.join(HERE, "dataset", "sim_incidents.csv"),
        encoding="utf-8-sig")):
    reps = by_inc.get(inc["incident_id"], [])
    if not reps:
        continue
    ts = datetime.strptime(inc["timestamp"], "%Y-%m-%d %H:%M:%S")
    n = len(reps)
    witness = len({r["user_id"] for r in reps})
    votes = Counter(r["citizen_severity"] for r in reps)
    vote_share = [votes.get(s, 0) / n for s in SEV]
    mode = votes.most_common(1)[0][0]
    inj = [float(r["injured_count"]) for r in reps
            if r["injured_count"] != ""]
    inj_mean = float(np.mean(inj)) if inj else -1
    inj_max = max(inj) if inj else -1
    credm = float(np.mean([float(r["user_true_report_ratio"])
                            for r in reps]))
    media = sum(r["has_media"] == "True" for r in reps)
    feats = [witness, inj_mean, int(inj_mean < 0), credm, media,
                ts.hour, int(ts.hour >= 23 or ts.hour < 7),
                ts.weekday(), ts.month,
                SEVI[mode]] + vote_share + [inj_max]
    feats += [int(inc["zone_type"] == z) for z in ZONES]
    feats += [int(inc["incident_type"] == t) for t in TYPES]
    X.append(feats)
    y.append(SEVI[inc["actual_severity"]])
sev_m = XGBClassifier(n_estimators=400, max_depth=6,
                        learning_rate=0.06,
                        objective="multi:softprob", num_class=4,
                        eval_metric="mlogloss", random_state=42)
sev_m.fit(np.array(X, float), np.array(y))
joblib.dump({"model": sev_m},
            os.path.join(ART, "severity.joblib"))
print(f"الخطورة: تدرب على {len(X)} حادثاً وحفظ")
print(f"\nكل النماذج جاهزة في: pipeline/artifacts/")