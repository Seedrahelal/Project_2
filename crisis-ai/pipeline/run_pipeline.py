# الـ Pipeline الكامل: بلاغات خام -> قرارات جاهزة للمشغل
# التسلسل (ملف المتطلبات): تجميع -> مصداقية -> خطورة -> توزيع

# اللحامات الأربعة المؤجلة لهذا اليوم:
# 1) قلب الحكم الزوجي = النموذج المتعلم (قواعد الأمان فوقه + fallback)
# 2) قاعدة أمان المصداقية في المسار الحي
# 3) witness_count الحقيقي من التجميع يغذي المصداقية والخطورة
# 4) بوابة الألوان: الأحمر لا يكمل السلسلة تلقائياً

import os
import sys
import joblib
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import model_clustering as mc
from models.model_dispatch import greedy_dispatch
from data.data_clustering import create_report

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(HERE, "pipeline", "artifacts")

SEV = ["low", "medium", "high", "critical"]
SEVI = {s: i for i, s in enumerate(SEV)}
SUB = {"full_form": 0, "voice": 1, "sos": 2}
ANCH = {"at_home": 0, "seeing_it": 1, "unknown": 2}
ZONES = ["residential", "commercial", "industrial", "rural"]
TYPES = ["medical", "road", "fire", "police", "rescue", "collapse", "disaster"]

# ---------- تحميل النماذج الجاهزة ----------
J = joblib.load(os.path.join(ART, "pairwise_judge.joblib"))
CRED = joblib.load(os.path.join(ART, "credibility.joblib"))["model"]
SEVM = joblib.load(os.path.join(ART, "severity.joblib"))["model"]

# ---------- اللحام 1: قلب الحكم الزوجي = النموذج المتعلم ----------
_rules = mc.same_incident_possible      # القواعد تبقى fallback

def learned_judge(r1, r2, eps_s, eps_t=mc.EPS_TEMPORAL_S):
    # قواعد الأمان الصلبة أولاً (لا تترك لاحتمال)
    if r1["latitude"] is None or r2["latitude"] is None:
        return False
    if r1["user_id"] == r2["user_id"]:
        return _rules(r1, r2, eps_s, eps_t)
    try:
        dist = mc.effective_distance_m(r1, r2)
        dt = abs((r1["timestamp"] - r2["timestamp"]).total_seconds())
        t1, t2 = r1["incident_type"], r2["incident_type"]
        feats = {
            "eff_distance_m": dist, "time_diff_s": dt,
            "same_type": int(t1 == t2),
            "causally_linked": int(
                frozenset((t1, t2)) in mc.CAUSALLY_LINKED
                and t1 != t2),
            "both_person_type": int(
                mc.PROPAGATION.get(t1) == "person"
                and mc.PROPAGATION.get(t2) == "person"),
            "both_at_home": int(r1["incident_anchor"] == "at_home"
                                and r2["incident_anchor"] == "at_home"),
            "any_unknown_anchor": int(
                "unknown" in (r1["incident_anchor"],
                                r2["incident_anchor"])),
            "same_user": 0,
            "sum_gps_error": r1["gps_accuracy"] + r2["gps_accuracy"],
            "min_user_ratio": min(
                r1.get("user_true_report_ratio", 0.8),
                r2.get("user_true_report_ratio", 0.8)),
        }
        x = np.array([[feats[c] for c in J["features"]]])
        p = J["model"].predict_proba(x)[0, 1]
        if p >= 0.65:
            return True
        if p <= 0.35:
            return False
        # المنطقة الرمادية: قرار القواعد + وسم للمشغل
        r1["judge_gray"] = r2["judge_gray"] = True
        return _rules(r1, r2, eps_s, eps_t)
    except Exception:
        return _rules(r1, r2, eps_s, eps_t)   # fallback دائم

mc.same_incident_possible = learned_judge     # التبديل الفعلي


# ---------- اللحامان 2 و3: المصداقية بالشهود الحقيقيين ----------
def credibility_score(r, witness_count):
    ts = r["timestamp"]
    inj = -1 if r["injured_count"] is None else float(r["injured_count"])
    ratio = float(r.get("user_true_report_ratio", 0.5))  # cold start
    feats = [witness_count, ratio,
                int(bool(r["is_witness"])), int(bool(r["has_media"])),
                int(r["injured_unknown"]), inj, ts.hour, ts.weekday(),
                SEVI[r["citizen_severity"]],
                SUB[r["submission_method"]],
                ANCH[r["incident_anchor"]]]
    feats += [int(r["incident_type"] == t) for t in TYPES]
    score = CRED.predict_proba(np.array([feats], float))[0, 1] * 100
    # قاعدة الأمان: منفرد بسجل نظيف لا ينزل تحت الأصفر
    if score < 40 and witness_count == 1 and ratio >= 0.6:
        score = 40.0
    level = "green" if score >= 70 else \
            "yellow" if score >= 40 else "red"
    return round(score, 1), level


def severity_predict(members, zone_type):
    n = len(members)
    witness = len({m["user_id"] for m in members})
    ts = min(m["timestamp"] for m in members)
    from collections import Counter
    votes = Counter(m["citizen_severity"] for m in members)
    vote_share = [votes.get(s, 0) / n for s in SEV]
    mode = votes.most_common(1)[0][0]
    inj = [float(m["injured_count"]) for m in members
            if m["injured_count"] is not None]
    inj_mean = float(np.mean(inj)) if inj else -1
    inj_max = max(inj) if inj else -1
    credm = float(np.mean([m.get("user_true_report_ratio", 0.5)
                            for m in members]))
    media = sum(bool(m["has_media"]) for m in members)
    feats = [witness, inj_mean, int(inj_mean < 0), credm, media,
                ts.hour, int(ts.hour >= 23 or ts.hour < 7),
                ts.weekday(), ts.month,
                SEVI[mode]] + vote_share + [inj_max]
    feats += [int(zone_type == z) for z in ZONES]
    feats += [int(members[0]["incident_type"] == t) for t in TYPES]
    proba = SEVM.predict_proba(np.array([feats], float))[0]
    return SEV[int(proba.argmax())], proba, mode


# ---------- السلسلة الكاملة ----------
def process(reports, units, zone_type="residential"):
    reports, summary = mc.cluster_reports(reports)
    groups = {}
    for r in reports:
        groups.setdefault(r.get("cluster_id", -1), []).append(r)

    incidents_out, log = [], []
    iid = 0
    for cid, members in groups.items():
        units_of = [members] if cid != -1 else [[m] for m in members]
        for mem in units_of:
            iid += 1
            witness = len({m["user_id"] for m in mem})
            # المصداقية: أفضل درجة بين البلاغات تمثل الحادث
            scores = [credibility_score(m, witness) for m in mem]
            best_score, best_level = max(scores)
            gray = any(m.get("judge_gray") for m in mem)
            flagged = any(m.get("status") == "flagged" for m in mem)
            entry = {"incident_id": f"LIVE-{iid:03d}",
                        "n_reports": len(mem), "witnesses": witness,
                        "type": mem[0]["incident_type"],
                        "credibility": best_score, "level": best_level,
                        "flags": ("قاض رمادي " if gray else "")
                        + ("عصبة معلمة" if flagged else "")}
            # بوابة الألوان: الأحمر لا يكمل تلقائياً
            if best_level == "red":
                entry["decision"] = "قائمة مراجعة - لا توزيع تلقائي"
                log.append(entry)
                continue
            sev, proba, citizen = severity_predict(mem, zone_type)
            entry.update({
                "citizen_says": citizen, "predicted": sev,
                "proba": " | ".join(
                    f"{s}:{100*p:.0f}%" for s, p in zip(SEV, proba)),
                "victims": int(max(
                    [float(m["injured_count"]) for m in mem
                        if m["injured_count"] is not None] or [0])),
                "lat": float(np.mean([m["latitude"] for m in mem])),
                "lng": float(np.mean([m["longitude"] for m in mem])),
            })
            entry["decision"] = ("بانتظار موافقة المشغل"
                                    if best_level == "yellow"
                                    else "جاهز للتوزيع")
            incidents_out.append(entry)
            log.append(entry)

    ready = [{"incident_id": e["incident_id"],
                "incident_type": e["type"],
                "predicted_severity": e["predicted"],
                "victims": e["victims"],
                "credibility": e["credibility"],
                "lat": e["lat"], "lng": e["lng"]}
                for e in incidents_out if e["decision"] == "جاهز للتوزيع"]
    plan = greedy_dispatch(ready, units)
    return log, plan


# ---------- عرض حي: موقف مركب يختبر كل الوصلات ----------
if __name__ == "__main__":
    B = datetime(2026, 7, 20, 21, 30, 0)
    reports = []
    # حريق بناء: 4 شهود حقيقيون (مستخدم مختلف لكل بلاغ)
    for i in range(4):
        reports.append(create_report(
            f"R{i}", 33.5138 + i * 0.0002, 36.2765,
            B + timedelta(seconds=40 * i), "fire",
            citizen_severity=["high", "critical", "high", "medium"][i],
            incident_anchor="seeing_it",
            injured_count=[2, 4, None, 3][i],
            has_media=(i % 2 == 0), gps_accuracy=15,
            user_id=f"USR-1{i:02d}"))
        reports[-1]["user_true_report_ratio"] = 0.85
    # إصابات من نفس الحريق (نوع مترابط - مستخدم خامس)
    reports.append(create_report(
        "R10", 33.5139, 36.2766, B + timedelta(seconds=200),
        "medical", citizen_severity="critical",
        incident_anchor="seeing_it", injured_count=3,
        gps_accuracy=20, user_id="USR-201"))
    reports[-1]["user_true_report_ratio"] = 0.9
    # جاران طبيان منفصلان (اختبار الفيزياء - مستخدمان مختلفان)
    for i in range(2):
        reports.append(create_report(
            f"R2{i}", 33.5250 + i * 0.00003, 36.3100,
            B + timedelta(seconds=30 * i), "medical",
            citizen_severity="high", incident_anchor="at_home",
            injured_count=1, gps_accuracy=25,
            user_id=f"USR-3{i}0"))
        reports[-1]["user_true_report_ratio"] = 0.8
    # كاذب كلاسيكي منفرد بعيد (سجل ملطخ)
    reports.append(create_report(
        "R30", 33.4700, 36.2200, B + timedelta(seconds=100),
        "fire", citizen_severity="critical",
        incident_anchor="unknown", injured_count=25,
        submission_method="sos", gps_accuracy=40,
        user_id="FKR-999"))
    reports[-1]["user_true_report_ratio"] = 0.15
    # بلاغ منفرد حقيقي بسجل نظيف (اختبار قاعدة الأمان)
    reports.append(create_report(
        "R40", 33.5400, 36.2500, B + timedelta(seconds=150),
        "road", citizen_severity="medium",
        incident_anchor="seeing_it", injured_count=None,
        gps_accuracy=18, user_id="USR-400"))
    reports[-1]["user_true_report_ratio"] = 0.88
    
    units = [
        {"unit_id": "AMB-1", "unit_type": "ambulance",
            "status": "available", "lat": 33.520, "lng": 36.290},
        {"unit_id": "AMB-2", "unit_type": "ambulance",
            "status": "available", "lat": 33.530, "lng": 36.310},
        {"unit_id": "FIRE-1", "unit_type": "fire",
            "status": "available", "lat": 33.510, "lng": 36.270},
        {"unit_id": "POL-1", "unit_type": "police",
            "status": "available", "lat": 33.535, "lng": 36.255},
    ]

    log, plan = process(reports, units)
    print("=" * 64)
    print("الحوادث بعد السلسلة الكاملة:")
    print("=" * 64)
    for e in log:
        print(f"\n{e['incident_id']} | {e['type']} | "
                f"بلاغات {e['n_reports']} | شهود {e['witnesses']}")
        print(f"  مصداقية {e['credibility']}% ({e['level']}) "
                f"{e['flags']}")
        if "predicted" in e:
            print(f"  المواطن: {e['citizen_says']} | "
                    f"النموذج: {e['predicted']} [{e['proba']}]")
        print(f"  القرار: {e['decision']}")
    print("\n" + "=" * 64)
    print("خطة التوزيع:")
    print("=" * 64)
    for a in plan:
        print(f"  {a['incident']} <- {a['unit']} "
                f"(بديل {a['alternative']}) | {a['reason']}")