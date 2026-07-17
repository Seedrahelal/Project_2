# النموذج 1 - منظومة تجميع البلاغات (ثلاث طبقات)
# الطبقة 1: ترشيح مكاني-زماني بدلالات ST-DBSCAN (شرطان منفصلان)
# الطبقة 2: تحقق دلالي (فيزياء انتشار النوع + المرساة + الترابط السببي)
# الطبقة 3: حسم بالسياسة (واثق / معلم للمشغل / منفرد) - لا قرار خاطئ صامت


import sys
import os
import math
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_clustering import get_all_scenarios

EARTH_RADIUS_M = 6371000

# ---------- بارامترات موثقة المصدر ----------
EPS_TEMPORAL_S = 600   # نافذة موجة الإبلاغ: منطق مجال، تعاير مستقبلاً من التشغيل
MIN_SAMPLES = 3        # قرار سياسة معلن: عدد الشهود الأدنى لتوثيق حادث
DEFAULT_EPS_S = 200.0  # احتياطي إذا تعذر الاستخراج من البيانات
CONFIDENT_USERS = 5    # عصبة بهذا العدد من الشهود: الصدفة مستبعدة رياضياً
ASSIGN_FACTOR = 2.0    # نصف قطر نسب البلاغ المتأخر لحادث مفتوح = المعامل × eps

# فيزياء انتشار الحادث حسب نوعه (الطبقة الدلالية)
PROPAGATION = {
                "medical": "person",
                "fire": "structure",
                "collapse": "structure",
                "rescue": "structure",
                "disaster": "area",
                "road": "area",
                "police": "area"}

# الأزواج المترابطة سببياً: حادث واحد يمكن أن يولد النوعين معاً
CAUSALLY_LINKED = {frozenset(p) for p in [
    ("fire", "medical"), ("fire", "rescue"), ("fire", "collapse"),
    ("collapse", "medical"), ("collapse", "rescue"),
    ("road", "medical"), ("police", "medical"),
    ("disaster", "fire"), ("disaster", "medical"),
    ("disaster", "collapse"), ("disaster", "rescue"),
    ("disaster", "road"),
]}


# ---------- أدوات المسافة ----------
def haversine_m(lat1, lon1, lat2, lon2):
    # المسافة الدقيقة على سطح الكرة الأرضية بالمتر - رياضيات كروية
    r1, r2 = math.radians(lat1), math.radians(lat2)
    dlat, dlon = r2 - r1, math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(r1) * math.cos(r2) * math.sin(dlon / 2) ** 2)
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def effective_distance_m(r1, r2):
    # حصر المسافة بالمجالات: أسوأ حالة = المقاسة - مجموع خطأي الجهازين.
    # خطأ GPS محسوب صراحة لا مبلوع في التقدير

    d = haversine_m(r1["latitude"], r1["longitude"],
                    r2["latitude"], r2["longitude"])
    return max(0.0, d - (r1["gps_accuracy"] + r2["gps_accuracy"]))


# ---------- الحكم الزوجي (المكون القابل للاستبدال) ----------
def same_incident_possible(r1, r2, eps_s, eps_t=EPS_TEMPORAL_S):
    
    # القاضي: هل يمكن أن يكون البلاغان لنفس الحادث؟
    # النسخة الحالية: قواعد. النسخة القادمة: نموذج متعلم بنفس التوقيع

    if r1["latitude"] is None or r2["latitude"] is None:
        return False
    # الشرط الزمني منفصل: لا مقايضة بين المكان والزمن (ST-DBSCAN)
    dt = abs((r1["timestamp"] - r2["timestamp"]).total_seconds())
    if dt > eps_t:
        return False
    if effective_distance_m(r1, r2) > eps_s:
        return False

    t1, t2 = r1["incident_type"], r2["incident_type"]
    if t1 == t2:
        # فيزياء النوع: الحالة الطبية ملتصقة بشخص واحد -
        # بلاغان طبيان من بيتين لمستخدمين مختلفين = مريضان مختلفان
        if (PROPAGATION[t1] == "person"
                and r1["incident_anchor"] == "at_home"
                and r2["incident_anchor"] == "at_home"
                and r1["user_id"] != r2["user_id"]):
            return False
        return True
    # نوعان مختلفان: جيران فقط إن كانا مترابطين سببياً
    return frozenset((t1, t2)) in CAUSALLY_LINKED


def can_found_cluster(r):
    
    # أهلية تأسيس العصبات: مرساة 'لا أعرف بدقة' تنضم لكنها
    # لا تؤسس ولا تمد جسوراً (إحداثياتها الأضعف دلالة)

    return r["latitude"] is not None and r["incident_anchor"] != "unknown"


# ---------- ما قبل التجميع ----------
def deduplicate_same_user(reports, eps_t=EPS_TEMPORAL_S):
    
    # الشاهد الواحد يعد مرة: بلاغات نفس المستخدم لنفس النوع
    # ضمن النافذة الزمنية = تحديثات لبلاغ واحد (إغلاق ثغرة التلاعب)

    by_key = {}
    for r in sorted(reports, key=lambda x: x["timestamp"]):
        key = (r["user_id"], r["incident_type"])
        chain = by_key.setdefault(key, [])
        if chain and (r["timestamp"] - chain[-1]["timestamp"]).total_seconds() <= eps_t:
            chain[0]["updates_merged"] = \
                chain[0].get("updates_merged", 0) + 1
            chain.append(r)          # يسجل ضمن السلسلة ولا يدخل التجميع
        else:
            r.setdefault("updates_merged", 0)
            chain.clear()
            chain.append(r)
            r["_active"] = True
    return [r for r in reports if r.pop("_active", False)]


def assign_to_open_incidents(reports, open_incidents, eps_s):

    # البعد السادس: البلاغ المتأخر يقارن مع الحوادث المفتوحة أولاً -
    # بدون هذا كل بلاغ متأخر يصنع حادثاً وهمياً جديداً

    remaining = []
    for r in reports:
        r["assigned_incident"] = None
        if r["latitude"] is not None:
            for inc in open_incidents:
                if not inc.get("is_open"):
                    continue
                d = haversine_m(r["latitude"], r["longitude"],
                                inc["latitude"], inc["longitude"])
                t_ok = (r["incident_type"] == inc["incident_type"]
                        or frozenset((r["incident_type"], inc["incident_type"]))
                        in CAUSALLY_LINKED)
                if d <= ASSIGN_FACTOR * eps_s and t_ok:
                    r["assigned_incident"] = inc["incident_id"]
                    r["status"] = "assigned_existing"
                    break
        if r["assigned_incident"] is None:
            remaining.append(r)
    return remaining


# ---------- استخراج eps من البيانات (الورقة الأصلية + Kneedle) ----------
def suggest_eps_spatial(reports, k=MIN_SAMPLES, eps_t=EPS_TEMPORAL_S):

    # منحنى k-distance: مسافة كل بلاغ لجاره الثالث الأقرب مرتبة تصاعدياً.
    # الركبة (أسلوب Kneedle) = eps المستخرجة من البيانات نفسها.
    # تعاد دورياً فيتكيف البارامتر مع كل مدينة ونمط تبليغ

    # قاعدة اتساق: الجار هنا بنفس تعريف الجار في التجميع -
    # متوافق زمنياً ضمن النافذة. بدونها موجتان منفصلتان زمنياً
    # متراكبتان مكانياً تضاعفان الكثافة الظاهرية فتنكمش eps خطأً

    pts = [r for r in reports if r["latitude"] is not None]
    if len(pts) < 2 * k + 2:
        return DEFAULT_EPS_S, "بيانات غير كافية - القيمة الاحتياطية"
    kth = []
    for i, a in enumerate(pts):
        ds = sorted(
            effective_distance_m(a, b)
            for j, b in enumerate(pts)
            if j != i and abs((a["timestamp"] - b["timestamp"]).total_seconds()) <= eps_t)
        if len(ds) >= k:      # نقطة بلا جيران زمنيين كافين =
            kth.append(ds[k - 1])   # مرشحة solo، لا تدخل المنحنى
    if len(kth) < k + 2:
        return DEFAULT_EPS_S, "بيانات غير كافية - القيمة الاحتياطية"
    kth.sort()
    lo, hi = kth[0], kth[-1]
    if hi - lo < 1e-9:
        return DEFAULT_EPS_S, "منحنى مسطح - القيمة الاحتياطية"
    n = len(kth)
    best_i, best_gap = 0, -1.0
    for i, v in enumerate(kth):          # Kneedle مبسطة:
        x = i / (n - 1)                  # أقصى ابتعاد للمنحنى
        y = (v - lo) / (hi - lo)         # عن قطر المستطيل
        if x - y > best_gap:
            best_gap, best_i = x - y, i
    eps = kth[best_i]
    eps = min(max(eps, 50.0), 500.0)     # حدود عقلانية فيزيائياً
    return eps, f"مستخرجة من منحنى k-distance عند الركبة ({eps:.0f} م)"

# ---------- الطبقة 1: التجميع (ST-DBSCAN بحكم زوجي مخصص) ----------
def _run_clustering(reports, eps_s, eps_t, min_samples):
    n = len(reports)
    nbrs = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if same_incident_possible(reports[i], reports[j], eps_s, eps_t):
                nbrs[i].append(j)
                nbrs[j].append(i)

    def is_core(i):
        return (can_found_cluster(reports[i])
                and len(nbrs[i]) + 1 >= min_samples)

    labels = [-1] * n
    cid = 0
    for i in range(n):
        if labels[i] != -1 or not is_core(i):
            continue
        labels[i] = cid
        queue = deque([i])
        while queue:
            p = queue.popleft()
            if not is_core(p):
                continue        # الحدودي ينضم ولا يوسع (لا جسور)
            for q in nbrs[p]:
                if labels[q] == -1:
                    labels[q] = cid
                    queue.append(q)
        cid += 1
    return labels


# ---------- الطبقتان 2 و 3 + الواجهة الكاملة ----------
def cluster_reports(reports, open_incidents=None,
                    eps_s=None, eps_t=EPS_TEMPORAL_S,
                    min_samples=MIN_SAMPLES):
    open_incidents = open_incidents or []
    summary = {
                "eps_note": "",
                "n_clusters": 0,
                "n_solo": 0,
                "n_no_location": 0,
                "n_assigned": 0, 
                "clusters": {}}
    if not reports:
        return reports, summary

    reports = deduplicate_same_user(reports, eps_t)

    no_loc = [r for r in reports if r["latitude"] is None]
    for r in no_loc:
        r.update(
                cluster_id=-1,
                witness_count=1,
                status="solo_no_location",
                flag_reason="بلا إحداثيات - عرض منفرد للمشغل")
    located = [r for r in reports if r["latitude"] is not None]

    if eps_s is None:
        eps_s, note = suggest_eps_spatial(located)
        summary["eps_note"] = note
    summary["eps_spatial_used"] = round(eps_s, 1)

    to_cluster = assign_to_open_incidents(located, open_incidents,eps_s)
    for r in located:
        if r.get("assigned_incident"):
            r.update(cluster_id=-1, witness_count=1, flag_reason="")

    labels = _run_clustering(to_cluster, eps_s, eps_t, min_samples)

    groups = {}
    for r, lb in zip(to_cluster, labels):
        r["cluster_id"] = int(lb)
        groups.setdefault(int(lb), []).append(r)

    for cid, members in groups.items():
        if cid == -1:
            for r in members:
                r.update(witness_count=1, status="solo",flag_reason="")
            continue
        users = {m["user_id"] for m in members}
        types = {m["incident_type"] for m in members}

        # الطبقة 3: السياسة - كبيرة أو صافية النوع = واثقة،
        # صغيرة مختلطة (ولو مترابطة) = معلمة للمشغل
        if len(types) == 1:
            status, reason = "confident", ""
        else:
            status = "flagged"
            reason = (
                        "أنواع مترابطة ضمن عصبة صغيرة "
                        f"{sorted(types)} - تعرض نصوصها للمشغل")
        for m in members:
            m.update(witness_count=len(users), status=status,flag_reason=reason)
        summary["clusters"][cid] = {
            "size": len(members), "users": len(users),
            "types": sorted(types), "status": status}

    summary["n_clusters"] = len(summary["clusters"])
    summary["n_solo"] = sum(1 for r in to_cluster
                            if r["cluster_id"] == -1)
    summary["n_no_location"] = len(no_loc)
    summary["n_assigned"] = sum(1 for r in located
                                if r.get("assigned_incident"))
    summary["n_flagged"] = sum(1 for c in summary["clusters"].values()
                                if c["status"] == "flagged")
    return reports, summary


if __name__ == "__main__":
    for sc in get_all_scenarios()[:4]:   # فحص سريع لأول 4 سيناريوهات
        reports = [dict(r) for r in sc["reports"]]
        _, s = cluster_reports(reports, sc["open_incidents"])
        print(f"\n{sc['name']}")
        print(f"  eps={s.get('eps_spatial_used')} م | "
                f"عصبات={s['n_clusters']} | منفرد={s['n_solo']} | "
                f"معلم={s['n_flagged']}")
        for cid, c in s["clusters"].items():
            print(f"    عصبة {cid}: {c['size']} بلاغاً، "
                    f"{c['users']} شاهداً، أنواع {c['types']}، "
                    f"حالة {c['status']}")