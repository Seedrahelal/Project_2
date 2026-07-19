# استكشاف وتنظيف بيانات Montgomery 911 الحقيقية
# واستخراج ملف المعايرة calibration.json الذي يقرؤه مولد الـ dataset

# المصدر: kaggle.com/datasets/mchirico/montcoalert
# ملاحظة منهجية: هذه مكالمات موزعة بلا labels مصداقية أو خطورة -
# تستخدم للمعايرة والاختبار الخارجي فقط، والـ labels من قواعدنا الموثقة

import os
import json
import pandas as pd
import matplotlib.pyplot as plt


HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
RAW = os.path.join(PROJECT, "dataset", "montco_911.csv")
OUT = os.path.join(PROJECT, "dataset", "calibration.json")


# حدود مقاطعة Montgomery تقريبياً - لفلترة الإحداثيات الفاسدة
LAT_MIN, LAT_MAX = 39.9, 40.5
LNG_MIN, LNG_MAX = -75.8, -74.9

# ترجمة فئاتهم الثلاث إلى أنواعنا (قرار موثق)
CATEGORY_MAP = {"EMS": "medical", "Fire": "fire", "Traffic": "road"}

print("قراءة الملف الخام (الأعمدة المطلوبة فقط)...")
df = pd.read_csv(RAW, usecols=["lat", "lng", "title", "timeStamp"])
n_raw = len(df)
print(f"عدد الصفوف الخام: {n_raw:,}")

# ---- التنظيف ----
# 1) فصل الفئة عن النوع الفرعي: "EMS: RESPIRATORY EMERGENCY"
parts = df["title"].str.split(":", n=1, expand=True)
df["category"] = parts[0].str.strip()
df["subtype"] = parts[1].str.strip().str.rstrip(" -")
df = df[df["category"].isin(CATEGORY_MAP)]
df["our_type"] = df["category"].map(CATEGORY_MAP)

# 2) فلترة الإحداثيات الفاسدة
before = len(df)
df = df[df["lat"].between(LAT_MIN, LAT_MAX)
        & df["lng"].between(LNG_MIN, LNG_MAX)]
print(f"صفوف حذفت لإحداثيات فاسدة: {before - len(df):,}")

# 3) تحويل الوقت واستخراج مركباته
df["timeStamp"] = pd.to_datetime(df["timeStamp"], errors="coerce")
df = df.dropna(subset=["timeStamp"])
df["hour"] = df["timeStamp"].dt.hour
df["dow"] = df["timeStamp"].dt.dayofweek      # 0=اثنين .. 6=أحد
df["month"] = df["timeStamp"].dt.month
df["date"] = df["timeStamp"].dt.date

n_clean = len(df)
print(f"الصفوف النظيفة: {n_clean:,} "
      f"({100 * n_clean / n_raw:.1f}% من الخام)")
print(f"المدى الزمني: {df['timeStamp'].min()} إلى "
        f"{df['timeStamp'].max()}")

# ---- الاستخراج ----
type_props = (df["our_type"].value_counts(normalize=True)
                .round(4).to_dict())
hour_w = (df["hour"].value_counts(normalize=True)
            .sort_index().round(5).tolist())
dow_w = (df["dow"].value_counts(normalize=True)
            .sort_index().round(5).tolist())
month_w = (df["month"].value_counts(normalize=True)
            .sort_index().round(5).tolist())
per_day = df.groupby("date").size()

calibration = {
    "source": "montcoalert (Kaggle) - Montgomery County PA 911",
    "n_records_used": int(n_clean),
    "date_range": [str(df["timeStamp"].min().date()),
                    str(df["timeStamp"].max().date())],
    "type_proportions": type_props,
    "hour_weights": hour_w,          # 24 وزناً - الفهرس = الساعة
    "dow_weights": dow_w,            # 7 أوزان - 0=اثنين
    "month_weights": month_w,        # 12 وزناً - الفهرس+1 = الشهر
    "calls_per_day_mean": round(float(per_day.mean()), 1),
    "calls_per_day_max": int(per_day.max()),
    "category_mapping": CATEGORY_MAP,
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(calibration, f, ensure_ascii=False, indent=2)
print(f"\nحفظ ملف المعايرة: {OUT}")

# ---- ملخص للفهم ----
print("\nنسب الأنواع الحقيقية:")
for t, p in type_props.items():
    print(f"  {t}: {100 * p:.1f}%")
print(f"\nمتوسط البلاغات باليوم: {calibration['calls_per_day_mean']}"
        f" | ذروة يوم واحد: {calibration['calls_per_day_max']}")
print("\nأكثر 5 أنواع فرعية بكل فئة:")
for cat in CATEGORY_MAP:
    top = (df[df['category'] == cat]['subtype']
            .value_counts().head(5))
    print(f"  {cat}: {', '.join(top.index)}")

busiest = per_day.idxmax()
print(f"\nأزحم يوم (لاختبار التجميع لاحقاً): {busiest} "
        f"بـ {per_day.max()} بلاغاً")

# ---- رسم فوري بلا حفظ ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(range(24), hour_w, color="#0F6E56")
axes[0].set_title("Real hourly distribution of 911 calls")
axes[0].set_xlabel("Hour of day")
axes[1].bar(type_props.keys(), type_props.values(),
            color=["#C0392B", "#2471A3", "#B7950B"])
axes[1].set_title("Real type proportions")
plt.tight_layout()
plt.show()