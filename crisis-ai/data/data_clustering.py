# عقد البيانات + سيناريوهات الاختبار لمنظومة تجميع البلاغات
# ملف البيانات الوحيد للنموذج الأول

import random
from datetime import datetime, timedelta

INCIDENT_TYPES = ["fire", "medical", "police", "rescue", "disaster", "collapse", "road"]
SEVERITY_LEVELS = ["low", "medium", "high", "critical"]
SUBMISSION_METHODS = ["full_form", "sos", "voice"]
# المرساة: وين الحادث بالنسبة للمبلغ (سؤال بضغطة واحدة في التطبيق)
ANCHORS = ["at_home", "seeing_it", "unknown"]

BASE_TIME = datetime(2026, 7, 14, 22, 0, 0)


def create_report(
                report_id,
                latitude,
                longitude,
                timestamp,
                incident_type,
                citizen_severity="high",
                incident_anchor="unknown",
                injured_count=None,
                description="",
                is_witness=False,
                has_media=False,
                submission_method="full_form",
                gps_accuracy=10.0,
                user_id="USR-000"):

    # بلاغ واحد حسب عقد البيانات.
    # latitude/longitude تقبلان None (رفض إذن الموقع أو فشل GPS)
    # الحقول is_witness / has_media / severity يستهلكها نموذج المصداقية لاحقاً

    assert incident_type in INCIDENT_TYPES
    assert citizen_severity in SEVERITY_LEVELS
    assert submission_method in SUBMISSION_METHODS
    assert incident_anchor in ANCHORS

    return {
        "report_id": report_id,
        "latitude": latitude,
        "longitude": longitude,
        "gps_accuracy": gps_accuracy,
        "timestamp": timestamp,
        "incident_type": incident_type,
        "citizen_severity": citizen_severity,
        "incident_anchor": incident_anchor,
        "injured_count": injured_count,
        "injured_unknown": injured_count is None,
        "description": description,
        "is_witness": is_witness,
        "has_media": has_media,
        "submission_method": submission_method,
        "user_id": user_id,
    }


def _burst(rid_start, lat, lon, itype, count, anchor="seeing_it", start_s=0, spread_deg=0.0012, user_start=None, seed=None):
    
    # دفعة بلاغات حول نقطة واحدة خلال 10 دقائق - مستخدم مختلف لكل بلاغ

    if seed is not None:
        random.seed(seed)
    out = []
    for i in range(count):
        uid = user_start + i if user_start else random.randint(1, 500)
        out.append(create_report(
            report_id=f"RPT-{rid_start + i:04d}",
            latitude=lat + random.uniform(-spread_deg, spread_deg),
            longitude=lon + random.uniform(-spread_deg, spread_deg),
            timestamp=BASE_TIME + timedelta(
                seconds=start_s + random.randint(0, 600)),
            incident_type=itype,
            incident_anchor=anchor,
            user_id=f"USR-{uid:03d}",
        ))
    return out


def get_all_scenarios():

    # كل سيناريو: اسم + بلاغات + حوادث مفتوحة (إن وجدت) + التوقعات.
    # التوقعات = مواصفات السلوك الصحيح، تُفحص في ملف الاختبار.

    scenarios = []

    # 1) الأساسي: 3 حوادث متباعدة + 5 بلاغات متفرقة
    random.seed(42)
    reports = (_burst(1, 33.5138, 36.2765, "fire", 12, user_start=1)
                + _burst(100, 33.5250, 36.3100, "medical", 11, user_start=50)
                + _burst(200, 33.4900, 36.2400, "collapse", 10, user_start=100))
    for k in range(5):
        reports.append(create_report(
            f"RPT-N{k:03d}",
            33.45 + random.uniform(0, 0.15),
            36.15 + random.uniform(0, 0.25),
            BASE_TIME + timedelta(seconds=random.randint(0, 3600)),
            random.choice(INCIDENT_TYPES),
            user_id=f"USR-{400 + k:03d}"))
    scenarios.append({
                    "name": "الأساسي: 3 حوادث + 5 متفرقة",
                    "reports": reports, "open_incidents": [],
                    "expect": {"n_clusters": 3, "n_solo": 5, "n_flagged": 0}
                    })

    # 2) مثال الجيران: 3 حالات طبية من 3 بيوت متلاصقة بنفس اللحظة
    #    فيزياء النوع: الحالة الطبية لا تنتشر بين الشقق => ثلاثة حوادث
    reports = [create_report(
        f"RPT-{i:04d}", 33.5138 + i * 0.00003, 36.2765,
        BASE_TIME + timedelta(seconds=i * 20), "medical",
        incident_anchor="at_home", gps_accuracy=25.0,
        user_id=f"USR-{i:03d}") for i in range(1, 4)]
    scenarios.append({
                    "name": "جيران متلاصقون - 3 بلاغات طبية من 3 بيوت",
                    "reports": reports, "open_incidents": [],
                    "expect": {"n_clusters": 0, "n_solo": 3}
                    })

    # 3) حريق بناء: جاران من بيتيهما + مار في الشارع => حادث واحد
    reports = [
        create_report("RPT-0001", 33.51380, 36.27650, BASE_TIME,
                        "fire", incident_anchor="at_home",
                        gps_accuracy=25.0, user_id="USR-001"),
        create_report("RPT-0002", 33.51382, 36.27652,
                        BASE_TIME + timedelta(seconds=45), "fire",
                        incident_anchor="at_home",
                        gps_accuracy=25.0, user_id="USR-002"),
        create_report("RPT-0003", 33.51370, 36.27640,
                        BASE_TIME + timedelta(seconds=90), "fire",
                        incident_anchor="seeing_it", user_id="USR-003"),
    ]
    scenarios.append({
                    "name": "حريق بناء - جاران + مار",
                    "reports": reports, "open_incidents": [],
                    "expect": {"n_clusters": 1, "n_solo": 0,
                    "n_flagged": 0}
                    })

    # 4) انفجار: حريق + طبي + انهيار بنفس النقطة => عصبة واحدة معلمة
    random.seed(9)
    reports = (_burst(1, 33.5138, 36.2765, "fire", 3, user_start=1)
                + _burst(50, 33.5138, 36.2765, "medical", 2, user_start=10)
                + _burst(80, 33.5139, 36.2766, "collapse", 1, user_start=20))
    scenarios.append({
                    "name": "انفجار - أنواع مترابطة سببياً",
                    "reports": reports, "open_incidents": [],
                    "expect": {"n_clusters": 1, "n_flagged": 1}
                })

    # 5) تكرار نفس المستخدم: 4 بلاغات ذعر من شخص واحد => بلاغ واحد
    reports = [create_report(
        f"RPT-{i:04d}", 33.5138, 36.2765,
        BASE_TIME + timedelta(seconds=i * 60), "fire",
        user_id="USR-001") for i in range(4)]
    scenarios.append({
                    "name": "نفس المستخدم أرسل 4 مرات",
                    "reports": reports, "open_incidents": [],
                    "expect": {"n_clusters": 0, "n_solo": 1}
                    })

    # 6) حادثان متقاربان (300م) بنوعين مترابطين:
    #    قد يندمجان (حد فيزيائي) لكن الاختلاط يرفع علماً - لا قرار صامت
    reports = (
            _burst( 1, 33.5138, 36.2765, "fire", 10, user_start=1, seed=7)
            + _burst(100, 33.5165, 36.2765, "medical", 10, user_start=50, seed=8))
    scenarios.append({
                    "name": "حادثان على 300م بنوعين مترابطين",
                    "reports": reports, "open_incidents": [],
                    "expect": {"no_silent_merge": True}
                    })

    # 7) نفس المكان بفارق ساعة => حادثان (الشرط الزمني المنفصل)
    reports = (
            _burst(1, 33.5138, 36.2765, "fire", 8, user_start=1, seed=3)
            + _burst(100, 33.5138, 36.2765, "fire", 8, start_s=3600, user_start=50, seed=4))
    scenarios.append({
                    "name": "نفس المكان بفارق ساعة",
                    "reports": reports, "open_incidents": [],
                    "expect": {"n_clusters": 2}
                    })

    # 8) بلاغ بلا إحداثيات => للمشغل منفرداً، لا تخمين
    reports = [create_report("RPT-0001", None, None, BASE_TIME, "fire", user_id="USR-001")]
    scenarios.append({
                    "name": "بلاغ بلا إحداثيات",
                    "reports": reports, "open_incidents": [],
                    "expect": {"n_no_location": 1}
                    })

    # 9) بلاغ متأخر وحادث مفتوح مؤكد => ينسب للحادث لا عصبة جديدة
    reports = [create_report(
        "RPT-0001", 33.5140, 36.2767,
        BASE_TIME + timedelta(minutes=40), "fire",
        incident_anchor="seeing_it", user_id="USR-099")]
    open_inc = [{"incident_id": "INC-0007",
                "latitude": 33.5138, "longitude": 36.2765,
                "incident_type": "fire", "is_open": True}]
    scenarios.append({
                    "name": "بلاغ متأخر لحادث مفتوح",
                    "reports": reports, "open_incidents": open_inc,
                    "expect": {"n_assigned": 1}
                    })

    # 10) حادث واحد بتشتت GPS كبير (~350م)
    reports = _burst(1, 33.5138, 36.2765, "fire", 12, spread_deg=0.0032, user_start=1, seed=11)
    scenarios.append({
                    "name": "حادث واحد بتشتت GPS كبير",
                    "reports": reports, "open_incidents": [],
                    "expect": {"min_clusters": 1, "same_type_only": True}
                    })

    return scenarios


if __name__ == "__main__":
    for sc in get_all_scenarios():
        print(f"{sc['name']}: {len(sc['reports'])} بلاغاً")