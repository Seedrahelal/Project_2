# النموذج 3 - تقييم مصداقية البلاغات (Random Forest)
# المدخل: بلاغ بكل حقول عقد البيانات + عدد الشهود (من التجميع)
# المخرج: درجة 0-100% + مستوى (green >=70 / yellow 40-69 / red <=39)

# مقياس النجاح الأساسي (خطة التقييم): Recall لفئة "حقيقي" > 0.85
# - تفويت بلاغ حقيقي أخطر من قبول بلاغ كاذب

import os
import sys
import csv
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, confusion_matrix, accuracy_score)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "dataset", "sim_reports.csv")

SEV = {"low": 0, "medium": 1, "high": 2, "critical": 3}
SUB = {"full_form": 0, "voice": 1, "sos": 2}
ANCH = {"at_home": 0, "seeing_it": 1, "unknown": 2}
TYPES = ["medical", "road", "fire", "police", "rescue", "collapse", "disaster"]

# ---------- التحميل + حساب عدد الشهود من الحقيقة الأرضية ----------
rows = []
with open(SRC, encoding="utf-8-sig") as f:
    raw = list(csv.DictReader(f))
# عد المستخدمين الفريدين لا البلاغات - نفس تعريف التجميع
# في التشغيل (اتساق تدريب/تشغيل)
inc_users = {}
for r in raw:
    if r["incident_id"]:
        inc_users.setdefault(r["incident_id"], set()).add(r["user_id"])
inc_counts = {k: len(v) for k, v in inc_users.items()}

FEATURES = ["witness_count", "user_true_report_ratio", "is_witness",
            "has_media", "injured_unknown", "injured_count_filled",
            "hour", "day_of_week", "citizen_severity",
            "submission_method", "anchor"] \
            + [f"type_{t}" for t in TYPES]

X, y, styles = [], [], []
for r in raw:
    ts = datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S")
    witness = inc_counts.get(r["incident_id"], 1)
    inj = -1 if r["injured_count"] == "" else float(r["injured_count"])
    feats = [witness,
                float(r["user_true_report_ratio"]),
                int(r["is_witness"] == "True"),
                int(r["has_media"] == "True"),
                int(r["injured_unknown"] == "True"),
                inj, ts.hour, ts.weekday(),
                SEV[r["citizen_severity"]],
                SUB[r["submission_method"]],
                ANCH[r["incident_anchor"]]]
    feats += [int(r["incident_type"] == t) for t in TYPES]
    X.append(feats)
    y.append(1 - int(r["is_fake"]))     # label=1 حقيقي | 0 كاذب
    styles.append(r["fake_style"])

X, y = np.array(X, dtype=float), np.array(y)

styles = np.array(styles)
X_tr, X_te, y_tr, y_te, st_tr, st_te = train_test_split(
    X, y, styles, test_size=0.2, random_state=42, stratify=y)


print(f"تدريب: {len(X_tr)} بلاغاً | اختبار: {len(X_te)} "
        f"(حقيقي {int(y_te.sum())} / كاذب {int((1-y_te).sum())})")

# ---------- التدريب ----------
model = RandomForestClassifier(
    n_estimators=200, max_depth=10,
    class_weight="balanced", random_state=42)
model.fit(X_tr, y_tr)
prob = model.predict_proba(X_te)[:, 1]      # احتمال "حقيقي" = الدرجة
score = (prob * 100).round(1)
pred = (prob >= 0.5).astype(int)
# قاعدة أمان صلبة فوق النموذج (نفس معمارية التجميع):
# بلاغ منفرد بسجل نظيف لا ينزل تحت الأصفر - يذهب لموافقة
# المشغل لا لقائمة المراجعة (تفويت حقيقي أخطر من قبول كاذب)
floor_mask = (score < 40) & (X_te[:, 0] == 1) & (X_te[:, 1] >= 0.6)
score[floor_mask] = 40.0



# ---------- التقييم حسب خطة التقييم ----------
rec_real = recall_score(y_te, pred, pos_label=1)
prec_real = precision_score(y_te, pred, pos_label=1)
rec_fake = recall_score(y_te, pred, pos_label=0)
tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
print("\nالتقييم (label=1 حقيقي):")
print(f"  Recall  حقيقي: {rec_real:.3f}  "
        f"(الحد المعتمد > 0.85) {'نجح' if rec_real > 0.85 else 'فشل'}")
print(f"  Precision حقيقي: {prec_real:.3f}")
print(f"  Recall  كاذب : {rec_fake:.3f}  (كم كاذباً التقطنا)")
print(f"  Accuracy: {accuracy_score(y_te, pred):.3f}")
print(f"  بلاغ حقيقي صنف كاذباً FN: {fn} | كاذب مر كحقيقي FP: {fp}")

# ---------- المستويات الثلاثة (من ملف المتطلبات) ----------
green = np.sum(score >= 70)
yellow = np.sum((score >= 40) & (score < 70))
red = np.sum(score < 40)
n = len(score)
print(f"\nتوزيع المستويات على الاختبار:")
print(f"  أخضر >=70 (يتابع تلقائياً): {green} ({100*green/n:.0f}%)")
print(f"  أصفر 40-69 (موافقة المشغل): {yellow} ({100*yellow/n:.0f}%)")
print(f"  أحمر <=39 (قائمة مراجعة)  : {red} ({100*red/n:.0f}%)")
# فحص أمان: كم بلاغاً حقيقياً وقع بالأحمر؟ (الخسارة الأخطر)
real_in_red = np.sum((score < 40) & (y_te == 1))
fake_in_green = np.sum((score >= 70) & (y_te == 0))
print(f"  حقيقي وقع بالأحمر: {real_in_red} | "
        f"كاذب مر بالأخضر: {fake_in_green}")
print(f"  بلاغات رفعتها قاعدة الأمان للأصفر: {int(floor_mask.sum())}")


# ---------- كشف الكاذب حسب نمطه ----------
print("\nكشف الكاذب حسب النمط:")
for st in ("classic", "smart"):
    m = st_te == st
    if m.sum():
        caught = int(np.sum(pred[m] == 0))
        in_green = int(np.sum(score[m] >= 70))
        print(f"  {st}: كشف {caught}/{m.sum()} "
                f"({100*caught/m.sum():.0f}%) | مر بالأخضر: {in_green}")


# ---------- أهم الأدلة ----------
print("\nأهم 8 أدلة في قرار المصداقية:")
order = np.argsort(model.feature_importances_)[::-1]
for i in order[:8]:
    print(f"  {FEATURES[i]}: {model.feature_importances_[i]:.3f}")