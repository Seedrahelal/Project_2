# النموذج 5 - توزيع الموارد الأمثل (Greedy)
# معادلة الأولوية من ملف المتطلبات مع تطبيع كل العوامل لمجال 0-1:
# Priority = 0.45*severity + 0.30*victims + 0.15*credibility + 0.10*eta_factor

# التقييم (خطة التقييم): Simulation مقارن ضد العشوائي وضد الأقرب-أولاً
# المقياس الحاسم: زمن الاستجابة الموزون بالخطورة (الحرج يخدم أولاً)

import math
import random

random.seed(2026)
SEV_W = {"low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}
UNIT_FOR = {"medical": "ambulance", "fire": "fire", "road": "police",
            "police": "police", "rescue": "rescue",
            "collapse": "rescue", "disaster": "rescue"}
SPEED_MPS = 11.0        # ~40 كم/س سرعة استجابة حضرية (افتراض معلن)
VICTIMS_CAP = 20        # سقف تطبيع المصابين


def eta_seconds(unit, inc):
    dx = (unit["lat"] - inc["lat"]) * 111320
    dy = (unit["lng"] - inc["lng"]) * 92000
    return math.hypot(dx, dy) / SPEED_MPS


def priority_score(inc, unit, min_eta):
    # المعادلة المعتمدة - كل عامل مطبع على 0-1 (شرط عقد البيانات)
    sev = SEV_W[inc["predicted_severity"]]
    vic = min(inc["victims"], VICTIMS_CAP) / VICTIMS_CAP
    cred = inc["credibility"] / 100.0
    eta = eta_seconds(unit, inc)
    eta_f = min_eta / eta if eta > 0 else 1.0   # الأقرب = 1
    return (0.45 * sev + 0.30 * vic + 0.15 * cred
            + 0.10 * eta_f), eta


def greedy_dispatch(incidents, units):
    # يكرر: اختيار أعلى زوج (حادث، وحدة متاحة متوافقة النوع) أولوية،
    # تخصيصه، وإعادة الحساب - حتى نفاد الوحدات أو الحوادث.
    # يعيد: قائمة تخصيصات + اقتراح بديل لكل تخصيص + التبرير

    free = [u for u in units if u["status"] == "available"]
    pending = list(incidents)
    plan = []
    while pending and free:
        best = None
        for inc in pending:
            need = UNIT_FOR[inc["incident_type"]]
            match = [u for u in free if u["unit_type"] == need]
            if not match:
                continue
            min_eta = min(eta_seconds(u, inc) for u in match)
            for u in match:
                s, eta = priority_score(inc, u, min_eta)
                if best is None or s > best[0]:
                    best = (s, eta, inc, u, match)
        if best is None:
            break
        s, eta, inc, u, match = best
        alt = sorted((x for x in match if x is not u),
                        key=lambda x: eta_seconds(x, inc))
        plan.append({
            "incident": inc["incident_id"], "unit": u["unit_id"],
            "eta_s": round(eta), "score": round(s, 3),
            "alternative": alt[0]["unit_id"] if alt else None,
            "reason": (f"خطورة {inc['predicted_severity']} | "
                        f"مصابون {inc['victims']} | "
                        f"مصداقية {inc['credibility']:.0f}% | "
                        f"وصول {round(eta)} ث"),
        })
        pending.remove(inc)
        free.remove(u)
    return plan


# ---------- استراتيجيات المقارنة ----------
def nearest_first(incidents, units):
    """الأقرب أولاً: بترتيب وصول البلاغات، كل حادث يأخذ أقرب وحدة"""
    free = [u for u in units if u["status"] == "available"]
    plan = []
    for inc in incidents:                       # ترتيب الوصول
        need = UNIT_FOR[inc["incident_type"]]
        match = [u for u in free if u["unit_type"] == need]
        if not match:
            continue
        u = min(match, key=lambda x: eta_seconds(x, inc))
        plan.append((inc, u, eta_seconds(u, inc)))
        free.remove(u)
    return plan


def random_dispatch(incidents, units):
    free = [u for u in units if u["status"] == "available"]
    plan = []
    for inc in incidents:
        need = UNIT_FOR[inc["incident_type"]]
        match = [u for u in free if u["unit_type"] == need]
        if not match:
            continue
        u = random.choice(match)
        plan.append((inc, u, eta_seconds(u, inc)))
        free.remove(u)
    return plan


# ---------- مولد مواقف المحاكاة ----------
def make_situation():
    n_inc = random.randint(3, 8)
    n_unit = random.randint(2, 6)
    incidents = [{
        "incident_id": f"I{i}",
        "incident_type": random.choice(list(UNIT_FOR)),
        "predicted_severity": random.choices(
            list(SEV_W), weights=[.28, .37, .25, .10])[0],
        "victims": random.randint(0, 15),
        "credibility": random.uniform(40, 100),
        "lat": 33.50 + random.uniform(0, 0.06),
        "lng": 36.25 + random.uniform(0, 0.09),
    } for i in range(n_inc)]
    types = list(set(UNIT_FOR.values()))
    units = [{
        "unit_id": f"U{i}", "unit_type": random.choice(types),
        "status": "available",
        "lat": 33.50 + random.uniform(0, 0.06),
        "lng": 36.25 + random.uniform(0, 0.09),
    } for i in range(n_unit)]
    return incidents, units


def weighted_response(assignments):
    """زمن الاستجابة الموزون بالخطورة - المقياس الحاسم"""
    tot_w, tot = 0.0, 0.0
    for inc, u, eta in assignments:
        w = SEV_W[inc["predicted_severity"]]
        tot += w * eta
        tot_w += w
    return tot / tot_w if tot_w else 0.0


if __name__ == "__main__":
    N = 500
    scores = {"greedy": [], "nearest": [], "random": []}
    crit_first = {"greedy": 0, "nearest": 0}
    n_crit_cases = 0
    for _ in range(N):
        incidents, units = make_situation()
        g = greedy_dispatch(incidents, units)
        g_pairs = [(next(i for i in incidents
                            if i["incident_id"] == a["incident"]),
                    next(u for u in units
                            if u["unit_id"] == a["unit"]),
                    a["eta_s"]) for a in g]
        nf = nearest_first(incidents, units)
        rd = random_dispatch(incidents, units)
        scores["greedy"].append(weighted_response(g_pairs))
        scores["nearest"].append(weighted_response(nf))
        scores["random"].append(weighted_response(rd))
        # هل خُدم أخطر حادث ممكن خدمته أولاً؟
        crit = [i for i in incidents
                if i["predicted_severity"] == "critical"]
        served_g = {a["incident"] for a in g}
        served_n = {i["incident_id"] for i, _, _ in nf}
        servable = [c for c in crit
                    if c["incident_id"] in served_g | served_n]
        if servable:
            n_crit_cases += 1
            if all(c["incident_id"] in served_g for c in servable):
                crit_first["greedy"] += 1
            if all(c["incident_id"] in served_n for c in servable):
                crit_first["nearest"] += 1

    import statistics as st
    print(f"محاكاة {N} موقف ندرة (حوادث أكثر من وحدات غالباً)")
    print("\nزمن الاستجابة الموزون بالخطورة (ثوانٍ - الأقل أفضل):")
    for k in scores:
        print(f"  {k:8}: {st.mean(scores[k]):7.1f}")
    print(f"\nخدمة الحوادث الحرجة (من {n_crit_cases} موقفاً فيه حرج):")
    print(f"  greedy خدمها : {crit_first['greedy']} "
            f"({100*crit_first['greedy']/n_crit_cases:.0f}%)")
    print(f"  nearest خدمها: {crit_first['nearest']} "
            f"({100*crit_first['nearest']/n_crit_cases:.0f}%)")

    # مثال مخرج للمشغل
    incidents, units = make_situation()
    plan = greedy_dispatch(incidents, units)
    print("\nمثال مخرج للمشغل (موقف واحد):")
    for a in plan[:3]:
        print(f"  {a['incident']} <- {a['unit']} "
                f"(بديل: {a['alternative']}) | {a['reason']}")