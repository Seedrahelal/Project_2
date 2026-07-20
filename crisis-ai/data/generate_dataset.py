# مولد الـ Dataset الموثق - ثلاث مراحل:
# 1) توليد الحقيقة الأرضية (حوادث فعلية بخطورة فعلية)
# 2) توليد البلاغات عن الحوادث (شهود + تشويه الذعر)
# 3) حقن البلاغات الكاذبة بالنسبة الموثقة

# كل قاعدة label موثقة بمصدرها في التعليقات.
# المعايرة الزمنية والنوعية من dataset/calibration.json
# (مستخرج من 662,831 بلاغ 911 حقيقي - Montgomery County PA)

# المخرجات في مجلد dataset:
# - sim_reports.csv   : بلاغات فردية (تدريب نموذج المصداقية + أزواج الحكم)
# - sim_incidents.csv : حوادث موحدة (تدريب نموذج الخطورة)

import os
import json
import random
import csv
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
CALIB = os.path.join(PROJECT, "dataset", "calibration.json")
OUT_REPORTS = os.path.join(PROJECT, "dataset", "sim_reports.csv")
OUT_INCIDENTS = os.path.join(PROJECT, "dataset", "sim_incidents.csv")

random.seed(2026)
N_INCIDENTS = 1500           # عدد الحوادث الحقيقية المولدة
FAKE_RATE = 0.10           # NFPA/USFA: الكاذب 9-12% من الحجم الكلي
SEVERITIES = ["low", "medium", "high", "critical"]

with open(CALIB, encoding="utf-8") as f:
    CAL = json.load(f)

# --- نسب الأنواع: الثلاثة المعايرة من الواقع تحمل 90% من الكتلة،
#     والأربعة غير المغطاة بالداتا الحقيقية نسب صغيرة معلنة كافتراض ---
TYPE_PROPS = {t: p * 0.90 for t, p in CAL["type_proportions"].items()}
TYPE_PROPS.update({"police": 0.05, "rescue": 0.02, "disaster": 0.015, "collapse": 0.015})

# --- توزيع الخطورة الأساسي لكل نوع (افتراض موثق قابل للنقاش) ---
SEV_BASE = {
    "medical":  [0.25, 0.40, 0.25, 0.10],
    "fire":     [0.20, 0.35, 0.30, 0.15],
    "road":     [0.45, 0.35, 0.15, 0.05],
    "police":   [0.30, 0.40, 0.20, 0.10],
    "rescue":   [0.15, 0.35, 0.35, 0.15],
    "collapse": [0.05, 0.20, 0.40, 0.35],
    "disaster": [0.05, 0.15, 0.35, 0.45],
}

# --- مدينة افتراضية: مناطق بأنواعها (لميزة zone_type) ---
ZONES = [
    ("residential", 33.510, 36.270, 0.012),
    ("residential", 33.535, 36.300, 0.012),
    ("commercial",  33.520, 36.285, 0.008),
    ("industrial",  33.495, 36.320, 0.010),
    ("rural",       33.560, 36.240, 0.025),
]

# --- تشويه الذعر: احتمال اختيار المواطن مقابل الخطورة الفعلية ---
#     (التحيز الذاتي من ملف المتطلبات - ما سيتعلم XGBoost تصحيحه)
PANIC = {"same": 0.55, "up1": 0.25, "down1": 0.15, "extreme": 0.05}

USERS_N, FAKERS_N = 800, 60


def pick_weighted(options, weights):
    return random.choices(options, weights=weights, k=1)[0]


def sample_timestamp():
    """وقت واقعي: الشهر واليوم والساعة من الأوزان الحقيقية المعايرة"""
    month = pick_weighted(range(1, 13), CAL["month_weights"])
    day = random.randint(1, 28)
    hour = pick_weighted(range(24), CAL["hour_weights"])
    return datetime(2026, month, day, hour,
                    random.randint(0, 59), random.randint(0, 59))


def shift_sev(sev, steps):
    i = max(0, min(3, SEVERITIES.index(sev) + steps))
    return SEVERITIES[i]


def distort(true_sev):
    """اختيار المواطن = الفعلية مشوهة بالذعر"""
    mode = pick_weighted(list(PANIC), list(PANIC.values()))
    if mode == "same":
        return true_sev
    if mode == "up1":
        return shift_sev(true_sev, +1)
    if mode == "down1":
        return shift_sev(true_sev, -1)
    return shift_sev(true_sev, random.choice([-2, +2]))


# ---------- سجل المستخدمين ----------
# مستخدم عادي: نسبة بلاغات صحيحة عالية / معتاد الكذب: منخفضة
users = {f"USR-{i:04d}": round(random.betavariate(8, 2), 3)
            for i in range(USERS_N)}
fakers = {f"FKR-{i:04d}": round(random.betavariate(2, 6), 3)
            for i in range(FAKERS_N)}

# ---------- المرحلة 1: الحقيقة الأرضية ----------
incidents = []
for i in range(N_INCIDENTS):
    itype = pick_weighted(list(TYPE_PROPS), list(TYPE_PROPS.values()))
    zone, zlat, zlng, zr = random.choice(ZONES)
    ts = sample_timestamp()
    sev = pick_weighted(SEVERITIES, SEV_BASE[itype])

    # قاعدة NFPA: الحريق الليلي (23-07) ~20% من الحرائق
    # لكنه ~نصف الوفيات => رفع مستوى بنصف الاحتمال
    if itype == "fire" and (ts.hour >= 23 or ts.hour < 7) \
            and random.random() < 0.5:
        sev = shift_sev(sev, +1)
    # افتراض موثق: حريق/انهيار بمنطقة صناعية أخطر
    if itype in ("fire", "collapse") and zone == "industrial" \
            and random.random() < 0.4:
        sev = shift_sev(sev, +1)

    sev_i = SEVERITIES.index(sev)
    injured = [random.randint(0, 1), random.randint(0, 3),
                random.randint(1, 6), random.randint(3, 15)][sev_i]
    incidents.append({
        "incident_id": f"INC-{i:04d}", "incident_type": itype,
        "latitude": round(zlat + random.uniform(-zr, zr), 6),
        "longitude": round(zlng + random.uniform(-zr, zr), 6),
        "timestamp": ts, "zone_type": zone,
        "actual_severity": sev, "injured_true": injured,
    })

# ---------- المرحلة 2: البلاغات عن الحوادث ----------
# عدد المبلغين حسب ظهور النوع (فيزياء الانتشار):
# الطبي خاص => قليل | الحريق/الكارثة مرئي => كثير ويزيد مع الخطورة
VISIBILITY = {"medical": 0.6, "police": 1.2, "road": 1.5,
                "rescue": 1.8, "fire": 2.5, "collapse": 3.0,
                "disaster": 4.0}
ANCHOR_P = {   # توزيع المرساة حسب النوع (افتراض معلن)
    "medical": [("at_home", .70), ("seeing_it", .25), ("unknown", .05)],
    "fire":    [("at_home", .30), ("seeing_it", .60), ("unknown", .10)],
}
ANCHOR_DEFAULT = [("at_home", .10), ("seeing_it", .70), ("unknown", .20)]

reports, rid = [], 1
for inc in incidents:
    lam = VISIBILITY[inc["incident_type"]] \
        * (1 + 0.5 * SEVERITIES.index(inc["actual_severity"]))
    n_rep = min(15, 1 + int(random.expovariate(1 / lam)))
    chosen_users = random.sample(list(users), n_rep)
    for u in chosen_users:
        acc = round(random.uniform(5, 50), 1)     # دقة GPS هاتف واقعية
        anch_opts = ANCHOR_P.get(inc["incident_type"], ANCHOR_DEFAULT)
        anchor = pick_weighted([a for a, _ in anch_opts],
                                [p for _, p in anch_opts])
        # المواطن قد يجهل عدد المصابين (25%) أو يقدره حول الحقيقي
        injured = None if random.random() < 0.25 else \
            max(0, inc["injured_true"] + random.randint(-2, 2))
        sev_choice = distort(inc["actual_severity"])
        sub = pick_weighted(["full_form", "sos", "voice"],
                            [.6, .25, .15]
                            if sev_choice != "critical"
                            else [.35, .5, .15])
        reports.append({
            "report_id": f"RPT-{rid:05d}",
            "incident_id": inc["incident_id"], "is_fake": 0,
            "latitude": round(inc["latitude"]
                                + random.uniform(-0.0008, 0.0008), 6),
            "longitude": round(inc["longitude"]
                                + random.uniform(-0.0008, 0.0008), 6),
            "gps_accuracy": acc,
            "timestamp": inc["timestamp"] + timedelta(
                seconds=random.randint(0, 480)),
            "incident_type": inc["incident_type"],
            "citizen_severity": sev_choice,
            "incident_anchor": anchor,
            "injured_count": injured,
            "injured_unknown": injured is None,
            "is_witness": random.random() < (.35 if anchor == "seeing_it" else .05),
            "has_media": random.random() < {"fire": .45, "road": .35,
                                            "medical": .10}.get(
                inc["incident_type"], .25),
            "submission_method": sub,
            "user_id": u, "user_true_report_ratio": users[u],
            "true_severity": inc["actual_severity"],
            "fake_style": "",
        })
        rid += 1
        # ذعر واقعي: 6% يعيد الإرسال (يختبر إسقاط التكرار لاحقاً)
        if random.random() < 0.06:
            dup = dict(reports[-1])
            dup["report_id"] = f"RPT-{rid:05d}"
            dup["timestamp"] += timedelta(seconds=random.randint(30, 300))
            reports.append(dup)
            rid += 1

# ---------- المرحلة 3: حقن الكذب ----------
# أنماط موثقة: تركز بالحريق (FIRE ALARM أعلى فئة حقيقية عندنا)،
# مستخدمون بسجل ضعيف، SOS بلا تفاصيل، بلا وسائط، مواقع عشوائية
n_fake = int(FAKE_RATE * len(reports) / (1 - FAKE_RATE))
for _ in range(n_fake):
    # 40% كاذب ذكي يقلد الحقيقي - 60% كلاسيكي بأنماط NFPA
    smart = random.random() < 0.4
    if smart:
        u = random.choice(list(users))          # سجل نظيف!
        ratio = users[u]
        sev = pick_weighted(SEVERITIES, [.10, .40, .40, .10])
        sub = pick_weighted(["full_form", "sos", "voice"],
                            [.50, .30, .20])
        media = random.random() < .15
        anchor = pick_weighted(["at_home", "seeing_it", "unknown"], [.25, .55, .20])
        injured = None if random.random() < .3 \
            else random.randint(1, 6)
    else:
        u = random.choice(list(fakers)) if random.random() < .8 \
            else random.choice(list(users))
        ratio = fakers.get(u, users.get(u))
        sev = pick_weighted(SEVERITIES, [.05, .15, .30, .50])
        sub = pick_weighted(["full_form", "sos", "voice"],
                            [.25, .55, .20])
        media = random.random() < .05
        anchor = pick_weighted(["at_home", "seeing_it", "unknown"], [.2, .3, .5])
        injured = None if random.random() < .6 \
            else random.randint(5, 30)
    itype = pick_weighted(["fire", "medical", "police", "road"], [.4, .3, .2, .1])
    ts = sample_timestamp()
    zone, zlat, zlng, zr = random.choice(ZONES)
    reports.append({
        "report_id": f"RPT-{rid:05d}",
        "incident_id": "", "is_fake": 1,
        "latitude": round(zlat + random.uniform(-zr, zr), 6),
        "longitude": round(zlng + random.uniform(-zr, zr), 6),
        "gps_accuracy": round(random.uniform(5, 50), 1),
        "timestamp": ts,
        "incident_type": itype,
        "citizen_severity": sev,
        "incident_anchor": anchor,
        "injured_count": injured,
        "injured_unknown": injured is None,
        "is_witness": random.random() < .05,
        "has_media": media,
        "submission_method": sub,
        "user_id": u, "user_true_report_ratio": ratio,
        "true_severity": "",
        "fake_style": "smart" if smart else "classic",
    })
    rid += 1

random.shuffle(reports)

# ---------- الكتابة ----------
with open(OUT_REPORTS, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=reports[0].keys())
    w.writeheader()
    for r in reports:
        r = dict(r)
        r["timestamp"] = r["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        r["injured_count"] = "" if r["injured_count"] is None \
            else r["injured_count"]
        w.writerow(r)

with open(OUT_INCIDENTS, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(incidents[0].keys()) + ["n_reports"])
    w.writeheader()
    counts = {}
    for r in reports:
        if r["incident_id"]:
            counts[r["incident_id"]] = counts.get(r["incident_id"], 0) + 1
    for inc in incidents:
        row = dict(inc)
        row["timestamp"] = row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        row["n_reports"] = counts.get(inc["incident_id"], 0)
        w.writerow(row)

# ---------- ملخص وفحوص عقلانية ----------
n_all = len(reports)
n_fk = sum(r["is_fake"] for r in reports)
match = sum(1 for r in reports if not r["is_fake"]
            and r["citizen_severity"] == r["true_severity"])
print(f"الحوادث المولدة: {len(incidents)}")
print(f"البلاغات الكلية: {n_all} "
      f"(كاذبة: {n_fk} = {100 * n_fk / n_all:.1f}%)")
print("توزيع الخطورة الفعلية للحوادث:")
for s in SEVERITIES:
    c = sum(1 for i in incidents if i["actual_severity"] == s)
    print(f"  {s}: {c} ({100 * c / len(incidents):.0f}%)")
print(f"تطابق اختيار المواطن مع الفعلية: "
      f"{100 * match / (n_all - n_fk):.0f}% "
        f"(المتوقع ~55% حسب مصفوفة الذعر)")
print(f"\nالملفات: sim_reports.csv و sim_incidents.csv في dataset/")