# الحكم الزوجي المتعلم - Random Forest يتنبأ باحتمال "نفس الحادث"
# + المقارنة العادلة مع قاضي القواعد على نفس أزواج الاختبار

# المخرج التشغيلي المستقبلي: احتمال 0-1 يغذي طبقة الحسم
# (عالي=دمج واثق | منخفض=فصل | متوسط=علم أحمر للمشغل)

import os
import sys
import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                                recall_score, confusion_matrix)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "dataset", "sim_pairs.csv")



# ---------- تحميل ----------
rows, sources = [], []
with open(SRC, encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    FEATURES = [c for c in reader.fieldnames
                if c not in ("label", "source")]
    for row in reader:
        rows.append([float(row[c]) for c in FEATURES]
                    + [int(row["label"])])
        sources.append(row["source"])
data = np.array(rows)
X, y = data[:, :-1], data[:, -1].astype(int)
sources = np.array(sources)

# تقسيم 80/20 قبل أي تدريب - بيانات الاختبار لا تلمس إلا للتقييم
X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
    X, y, sources, test_size=0.2, random_state=42, stratify=y)
print(f"أزواج التدريب: {len(X_tr)} | الاختبار: {len(X_te)}")



# ---------- القاضي المتعلم ----------
judge = RandomForestClassifier(
    n_estimators=200,       # عدد الأشجار المصوتة
    max_depth=8,            # عمق محدود = مقاومة overfitting
    class_weight="balanced",
    random_state=42)
judge.fit(X_tr, y_tr)
prob = judge.predict_proba(X_te)[:, 1]
pred_ml = (prob >= 0.5).astype(int)

# ---------- قاضي القواعد على نفس أزواج الاختبار ----------
def rules_judge(row):
    """نسخة القواعد بدلالة خصائص الزوج (نفس منطق النموذج 1)"""
    f = dict(zip(FEATURES, row))
    if f["time_diff_s"] > 600:
        return 0
    if f["eff_distance_m"] > 200:
        return 0
    if f["same_type"]:
        if f["both_person_type"] and f["both_at_home"] \
                and not f["same_user"]:
            return 0
        return 1
    return int(f["causally_linked"])

pred_rules = np.array([rules_judge(r) for r in X_te])

# ---------- المقارنة العادلة ----------
def report(name, pred):
    acc = accuracy_score(y_te, pred)
    prec = precision_score(y_te, pred)
    rec = recall_score(y_te, pred)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    print(f"\n{name}")
    print(f"  Accuracy : {acc:.3f}")
    print(f"  Precision: {prec:.3f}  (من قال عنهم نفس الحادث - كم صح؟)")
    print(f"  Recall   : {rec:.3f}  (من هم فعلاً نفس الحادث - كم التقط؟)")
    print(f"  دمج خاطئ FP: {fp} | فصل خاطئ FN: {fn}")
    return acc

print("=" * 58)
print("المقارنة العادلة على نفس أزواج الاختبار")
print("=" * 58)
acc_ml = report("القاضي المتعلم (Random Forest)", pred_ml)
acc_rules = report("قاضي القواعد (النموذج 1)", pred_rules)

# ---------- المنطقة الرمادية: أين النموذج غير واثق؟ ----------
gray = np.sum((prob > 0.35) & (prob < 0.65))
print(f"\nالمنطقة الرمادية (احتمال 0.35-0.65): {gray} زوجاً "
      f"({100 * gray / len(prob):.1f}%) - ترفع للمشغل معلمة")

# ---------- أهم الأدلة عند القاضي المتعلم ----------
print("\nأهم الخصائص في قرار القاضي المتعلم:")
order = np.argsort(judge.feature_importances_)[::-1]
for i in order:
    print(f"  {FEATURES[i]}: {judge.feature_importances_[i]:.3f}")


# ---------- التشخيص: الدقة لكل سيناريو على حدة ----------
print("\nالدقة حسب مصدر الزوج (متعلم | قواعد | عدد):")
for src in sorted(set(s_te)):
    m = s_te == src
    a_ml = accuracy_score(y_te[m], pred_ml[m])
    a_ru = accuracy_score(y_te[m], pred_rules[m])
    mark = "  <-- هنا يتفوق التعلم" if a_ml - a_ru > 0.15 else ""
    print(f"  {src}: {a_ml:.2f} | {a_ru:.2f} | n={m.sum()}{mark}")


# ---------- فحص الكاذب الانتهازي: حسم خاطئ أم تصعيد سليم؟ ----------
m = s_te == "كاذب انتهازي"
in_gray = np.sum((prob[m] > 0.35) & (prob[m] < 0.65))
print(f"\nالكاذب الانتهازي (n={m.sum()}): "
        f"{in_gray} في المنطقة الرمادية (تصعيد سليم) | "
        f"{m.sum() - in_gray} حسم واثق")



print("\n" + "=" * 58)
verdict = "المتعلم تفوق" if acc_ml > acc_rules else \
            "القواعد صمدت - نوثق النتيجة بصدق"
print(f"الخلاصة: {verdict}")