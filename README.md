نظام إدارة الأزمات والاستجابة للطوارئ — قسم الذكاء الاصطناعي
نظام ذكي لتحليل بلاغات الطوارئ يعتمد على أربعة نماذج متسلسلة: تجميع البلاغات المتكررة (DBSCAN)، تقييم المصداقية (Random Forest)، توقع الخطورة الفعلية (XGBoost)، وتوزيع الموارد الأمثل (Greedy Algorithm).
هيكل المشروع

data — البيانات ومولدات الـ datasets
models — النماذج الأربعة، كل نموذج بملف مستقل
pipeline — ربط النماذج بسلسلة واحدة
tests — اختبارات كل نموذج
docs — التوثيق وعقد البيانات وخطة التقييم

التشغيل لأول مرة
المتطلبات: Python 3.10 أو أحدث
الخطوة 1 — تنزيل المشروع:
git clone https://github.com/Seedrahelal/Project_2.git
cd Project_2
الخطوة 2 — إنشاء البيئة المعزولة:
python -m venv venv
الخطوة 3 — تفعيل البيئة على Windows:
venv\Scripts\activate
تفعيل البيئة على Linux أو Mac:
source venv/bin/activate
الخطوة 4 — تثبيت المكتبات:
pip install -r requirements.txt
التشغيل في كل مرة
فتح terminal داخل مجلد المشروع ثم تفعيل البيئة:
venv\Scripts\activate
حالة المشروع

 عقد البيانات وخطة التقييم (docs)
 النموذج 1 — DBSCAN
 النموذج 2 — Random Forest
 النموذج 3 — XGBoost
 النموذج 4 — Greedy
 الـ Pipeline الكامل

