# مولد أزواج التدريب للحكم الزوجي - نسخة 2 (عينة ممثلة)
# = أزواج عضوية من sim_reports.csv
# + 11 سيناريو التباس متعمد على 4 محاور (hard example mining)

# معيار نجاح معلن مسبقا: بعد إعادة التدريب يجب أن ترتفع أهمية
# خصائص فيزياء الأنواع والمراسي من الصفر - وإلا فالإثراء فشل

import os
import sys
import csv
import random
from datetime import datetime, timedelta
from itertools import combinations

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_clustering import (haversine_m, effective_distance_m, CAUSALLY_LINKED, PROPAGATION)

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
SRC = os.path.join(PROJECT, "dataset", "sim_reports.csv")
OUT = os.path.join(PROJECT, "dataset", "sim_pairs.csv")

random.seed(2026)
BASE = datetime(2026, 7, 14, 20, 0, 0)
M = 0.0000090   # درجة عرض تقابل ~1 متر (تقريب محلي)


def pair_features(a, b):
    dist = effective_distance_m(a, b)
    dt = abs((a["timestamp"] - b["timestamp"]).total_seconds())
    t1, t2 = a["incident_type"], b["incident_type"]
    return {
        "eff_distance_m": round(dist, 1),
        "time_diff_s": round(dt, 1),
        "same_type": int(t1 == t2),
        "causally_linked": int(frozenset((t1, t2)) in CAUSALLY_LINKED
                                and t1 != t2),
        "both_person_type": int(PROPAGATION.get(t1) == "person"
                                and PROPAGATION.get(t2) == "person"),
        "both_at_home": int(a["incident_anchor"] == "at_home"
                            and b["incident_anchor"] == "at_home"),
        "any_unknown_anchor": int("unknown" in (a["incident_anchor"],
                                                b["incident_anchor"])),
        "same_user": int(a["user_id"] == b["user_id"]),
        "min_user_ratio": round(min(
            float(a.get("user_true_report_ratio", 0.8)),
            float(b.get("user_true_report_ratio", 0.8))), 3),
        "sum_gps_error": round(a["gps_accuracy"] + b["gps_accuracy"], 1),
    }


def mk(inc, lat, lng, ts, itype, anchor="seeing_it", acc=None, user=None, ratio=None):
    # بلاغ سيناريو مصغر - الحقول التي يراها الحكم الزوجي فقط
    return {"incident_id": inc, "latitude": lat, "longitude": lng,
            "gps_accuracy": acc or random.uniform(8, 35),
            "timestamp": ts, "incident_type": itype,
            "incident_anchor": anchor,
            "user_id": user or f"SC-{random.randint(0, 99999):05d}",
            "user_true_report_ratio": ratio if ratio is not None
                else round(random.uniform(0.60, 0.95), 3),}


def group_pairs(reports):
    """كل الأزواج داخل مجموعة سيناريو - الجواب بالوراثة من incident_id"""
    out = []
    for a, b in combinations(reports, 2):
        out.append((pair_features(a, b),
                    int(a["incident_id"] == b["incident_id"])))
    return out


# ---------- سيناريوهات الالتباس المتعمد ----------
def scenarios():
    all_pairs, counts = [], {}

    def add(name, prs):
        all_pairs.extend((f, l, name) for f, l in prs)
        counts[name] = counts.get(name, 0) + len(prs)

    for k in range(70):   # 1) جيران طبيون متلاصقون
        lat, lng = 33.51 + random.uniform(0, .02), 36.27 + random.uniform(0, .02)
        t0 = BASE + timedelta(minutes=random.randint(0, 300))
        g = [mk(f"A{k}", lat + i * random.uniform(3, 30) * M, lng, 
                t0 + timedelta(seconds=random.randint(0, 120)),
                "medical", "at_home", acc=random.uniform(15, 40))
                for i in range(random.randint(2, 3))]
        # كل واحد حادث مستقل
        for i, r in enumerate(g):
            r["incident_id"] = f"A{k}-{i}"
        add("جيران طبيون", group_pairs(g))

    for k in range(60):   # 2) حريق + طبي مستقل جنبه
        lat, lng = 33.52 + random.uniform(0, .02), 36.28 + random.uniform(0, .02)
        t0 = BASE + timedelta(minutes=random.randint(0, 300))
        d = random.uniform(150, 350) * M
        g = [mk(f"F{k}", lat + random.uniform(-40, 40) * M,
                lng + random.uniform(-40, 40) * M,
                t0 + timedelta(seconds=random.randint(0, 300)), "fire",
                random.choice(["at_home", "seeing_it"]))
                for _ in range(random.randint(2, 3))]
        g.append(mk(f"F{k}X", lat + d, lng,
                    t0 + timedelta(seconds=random.randint(0, 300)),
                    "medical", "at_home"))
        add("حريق+طبي مستقلان", group_pairs(g))

    for k in range(50):   # 3) انفجار مركب: أنواع مترابطة = نفس الحادث
        lat, lng = 33.50 + random.uniform(0, .02), 36.30 + random.uniform(0, .02)
        t0 = BASE + timedelta(minutes=random.randint(0, 300))
        types = ["fire", "medical", "collapse", "fire", "rescue"]
        g = [mk(f"EX{k}", lat + random.uniform(-60, 60) * M,
                lng + random.uniform(-60, 60) * M,
                t0 + timedelta(seconds=random.randint(0, 400)),
                random.choice(types),
                random.choice(["seeing_it", "at_home"]))
                for _ in range(random.randint(3, 5))]
        add("انفجار مركب", group_pairs(g))

    for k in range(60):   # 4) نفس المكان بفارق 15-40 دقيقة
        lat, lng = 33.53 + random.uniform(0, .02), 36.26 + random.uniform(0, .02)
        t0 = BASE + timedelta(minutes=random.randint(0, 200))
        gap = timedelta(minutes=random.uniform(15, 40))
        g = [mk(f"T{k}a", lat + random.uniform(-30, 30) * M, lng, 
                t0 + timedelta(seconds=random.randint(0, 240)), "fire")
                for _ in range(2)]
        g += [mk(f"T{k}b", lat + random.uniform(-30, 30) * M, lng,
                    t0 + gap + timedelta(seconds=random.randint(0, 240)),
                    "fire") for _ in range(2)]
        add("نفس المكان بفاصل زمني", group_pairs(g))

    for k in range(60):   # 5) مبلغ متأخر عن نفس الحادث (موجب صعب)
        lat, lng = 33.49 + random.uniform(0, .02), 36.29 + random.uniform(0, .02)
        t0 = BASE + timedelta(minutes=random.randint(0, 200))
        g = [mk(f"L{k}", lat + random.uniform(-50, 50) * M, lng,
                t0 + timedelta(seconds=random.randint(0, 300)), "fire")
                for _ in range(random.randint(2, 3))]
        g.append(mk(f"L{k}", lat + random.uniform(-50, 50) * M, lng,
                    t0 + timedelta(minutes=random.uniform(12, 25)),
                    "fire", "seeing_it"))
        add("مبلغ متأخر", group_pairs(g))

    for k in range(50):   # 6) حريقان بنفس الحي بنفس الساعة (وضع أزمة)
        lat, lng = 33.54 + random.uniform(0, .02), 36.31 + random.uniform(0, .02)
        t0 = BASE + timedelta(minutes=random.randint(0, 120))
        d = random.uniform(250, 600) * M
        g = [mk(f"C{k}a", lat + random.uniform(-40, 40) * M, lng,
                t0 + timedelta(seconds=random.randint(0, 400)), "fire")
                for _ in range(2)]
        g += [mk(f"C{k}b", lat + d + random.uniform(-40, 40) * M, lng,
                    t0 + timedelta(seconds=random.randint(0, 400)), "fire")
                for _ in range(2)]
        add("حريقان متجاوران", group_pairs(g))

    for k in range(50):   # 7) حادثا سير متتاليان بنفس الطريق
        lat, lng = 33.48 + random.uniform(0, .02), 36.25 + random.uniform(0, .02)
        t0 = BASE + timedelta(minutes=random.randint(0, 200))
        d = random.uniform(80, 250) * M
        g = [mk(f"R{k}a", lat, lng + random.uniform(-20, 20) * M, 
                t0, "road", "seeing_it"),
             mk(f"R{k}b", lat + d, lng + random.uniform(-20, 20) * M,
                t0 + timedelta(minutes=random.uniform(5, 25)),
                "road", "seeing_it")]
        add("سلسلة حوادث سير", group_pairs(g))

    for k in range(50):   # 8) تشتت GPS واسع - نفس الحادث (موجب صعب)
        lat, lng = 33.55 + random.uniform(0, .02), 36.27 + random.uniform(0, .02)
        t0 = BASE + timedelta(minutes=random.randint(0, 200))
        g = [mk(f"W{k}", lat + random.uniform(-350, 350) * M,
                lng + random.uniform(-350, 350) * M,
                t0 + timedelta(seconds=random.randint(0, 400)),
                "disaster", acc=random.uniform(30, 50))
            for _ in range(random.randint(4, 6))]
        add("تشتت واسع", group_pairs(g))

    for k in range(40):   # 9) مشاهدون من بعيد بمراس ضعيفة - نفس الحادث
        lat, lng = 33.52 + random.uniform(0, .02), 36.24 + random.uniform(0, .02)
        t0 = BASE + timedelta(minutes=random.randint(0, 200))
        g = [mk(f"D{k}", lat, lng, t0, "fire", "at_home")]
        g += [mk(f"D{k}", lat + random.uniform(200, 450) * M,
                 lng + random.uniform(-100, 100) * M,
                    t0 + timedelta(seconds=random.randint(30, 400)), "fire",
                    random.choice(["seeing_it", "unknown"]))
                for _ in range(random.randint(2, 3))]
        add("مشاهدون بعيدون", group_pairs(g))

    for k in range(40):   # 10) نفس المستخدم - حادثان مختلفان
        lat, lng = 33.50 + random.uniform(0, .02), 36.26 + random.uniform(0, .02)
        t0 = BASE + timedelta(minutes=random.randint(0, 200))
        u = f"SC-U{k:04d}"
        g = [mk(f"U{k}a", lat, lng, t0, "road", "seeing_it", user=u),
             mk(f"U{k}b", lat + random.uniform(300, 900) * M, lng,
                t0 + timedelta(minutes=random.uniform(20, 60)),
                "medical", "seeing_it", user=u)]
        add("نفس المستخدم حادثان", group_pairs(g))

    for k in range(35):   # 11) كاذب انتهازي ملاصق لحادث حقيقي
        lat, lng = 33.51 + random.uniform(0, .02), 36.32 + random.uniform(0, .02)
        t0 = BASE + timedelta(minutes=random.randint(0, 200))
        g = [mk(f"OP{k}", lat + random.uniform(-40, 40) * M, lng,
                t0 + timedelta(seconds=random.randint(0, 300)), "fire")
                for _ in range(2)]
        g.append(mk(f"OP{k}-FAKE",
                    lat + random.uniform(30, 150) * M, lng,
                    t0 + timedelta(seconds=random.randint(0, 400)),
                    "fire", "unknown",
                    ratio=round(random.uniform(0.05, 0.35), 3)))
        add("كاذب انتهازي", group_pairs(g))

    return all_pairs, counts


# ---------- الأزواج العضوية من الملف المولد ----------
def organic_pairs():
    reports = []
    with open(SRC, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            row["latitude"] = float(row["latitude"])
            row["longitude"] = float(row["longitude"])
            row["gps_accuracy"] = float(row["gps_accuracy"])
            row["timestamp"] = datetime.strptime(
                row["timestamp"], "%Y-%m-%d %H:%M:%S")
            row["is_fake"] = int(row["is_fake"])
            row["user_true_report_ratio"] = float(
                row["user_true_report_ratio"])
            reports.append(row)
    by_inc = {}
    for r in reports:
        if r["incident_id"]:
            by_inc.setdefault(r["incident_id"], []).append(r)
    pairs = []
    for members in by_inc.values():        # موجبة طبيعية
        for a, b in combinations(members, 2):
            pairs.append((pair_features(a, b), 1, "عضوي موجب"))
    n_pos = len(pairs)
    real = [r for r in reports if not r["is_fake"]]
    easy = []
    while len(easy) < n_pos // 2:          # سالبة سهلة (للتنويع فقط)
        a, b = random.sample(real, 2)
        if a["incident_id"] != b["incident_id"]:
            easy.append((pair_features(a, b), 0, "عضوي سالب"))
    for fk in [r for r in reports if r["is_fake"]]:
        easy.append((pair_features(fk, random.choice(real)), 0, "عضوي مع كاذب"))
    return pairs + easy


org = organic_pairs()
sc, counts = scenarios()
pairs = org + sc
random.shuffle(pairs)




with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
    cols = list(pairs[0][0].keys()) + ["label", "source"]
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for feats, lb, src in pairs:
        row = dict(feats)
        row["label"] = lb
        row["source"] = src
        w.writerow(row)

n_all = len(pairs)
n_p = sum(lb for _, lb, _ in pairs)
print(f"الأزواج الكلية: {n_all} "
        f"(موجبة {n_p} | سالبة {n_all - n_p})")
print(f"منها عضوية: {len(org)} | سيناريوهات متعمدة: {len(sc)}")
print("\nتفصيل السيناريوهات:")
for name, c in counts.items():
    print(f"  {name}: {c} زوجاً")
print(f"\nحفظ: dataset/sim_pairs.csv")