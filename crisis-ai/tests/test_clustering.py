# الاختبار الموحد للنموذج 1 - منظومة تجميع البلاغات
# يفحص السيناريوهات العشرة مقابل توقعاتها المعتمدة
# ثم يعرض الرسم البصري في نافذة فورية (بدون حفظ ملفات)

import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_clustering import get_all_scenarios
from models.model_clustering import cluster_reports


def check(sc, summary, reports):
    # يقارن نتيجة السيناريو مع توقعاته - يعيد قائمة الإخفاقات
    exp, fails = sc["expect"], []

    def eq(key, actual):
        if key in exp and actual != exp[key]:
            fails.append(f"{key}: متوقع {exp[key]} - نتج {actual}")

    eq("n_clusters", summary["n_clusters"])
    eq("n_solo", summary["n_solo"])
    eq("n_flagged", summary.get("n_flagged", 0))
    eq("n_no_location", summary["n_no_location"])
    eq("n_assigned", summary["n_assigned"])

    if "min_clusters" in exp and \
            summary["n_clusters"] < exp["min_clusters"]:
        fails.append(f"عصبات أقل من {exp['min_clusters']}")

    if exp.get("same_type_only"):
        for cid, c in summary["clusters"].items():
            if len(c["types"]) != 1:
                fails.append(f"عصبة {cid} خلطت أنواعاً: {c['types']}")

    if exp.get("no_silent_merge"):
        # الضمانة: أي عصبة مختلطة الأنواع يجب أن تكون معلمة لا واثقة
        for cid, c in summary["clusters"].items():
            if len(c["types"]) > 1 and c["status"] == "confident":
                fails.append(f"عصبة {cid} مختلطة ومرت صامتة!")
    return fails


def main():
    scenarios = get_all_scenarios()
    print("=" * 62)
    print("الاختبار الموحد - منظومة تجميع البلاغات (10 سيناريوهات)")
    print("=" * 62)

    passed, results_for_plot = 0, []
    for sc in scenarios:
        reports = [dict(r) for r in sc["reports"]]
        reports, summary = cluster_reports(
            reports, open_incidents=sc["open_incidents"])
        fails = check(sc, summary, reports)
        ok = not fails
        passed += ok
        print(f"[{'نجح' if ok else 'فشل'}] {sc['name']}")
        if summary.get("eps_note"):
            print(f"        eps = {summary['eps_spatial_used']} م "
                    f"({summary['eps_note']})")
        for f in fails:
            print(f"        -> {f}")
        results_for_plot.append((sc["name"], reports, summary))

    print("=" * 62)
    print(f"النتيجة: نجح {passed} من {len(scenarios)}")
    print("=" * 62)

    # ----- الرسم البصري: 4 سيناريوهات مختارة، نافذة فورية -----
    chosen = [0, 3, 5, 9]   # الأساسي، الانفجار، 300م، التشتت الكبير
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    colors = plt.cm.tab10.colors

    for ax, idx in zip(axes.flat, chosen):
        name, reports, summary = results_for_plot[idx]
        located = [r for r in reports if r["latitude"] is not None]
        for r in located:
            cid = r.get("cluster_id", -1)
            if cid == -1:
                ax.scatter(r["longitude"], r["latitude"], c="gray",
                            marker="x", s=70)
            else:
                flagged = r.get("status") == "flagged"
                ax.scatter(r["longitude"], r["latitude"],
                            color=colors[cid % 10], s=60,
                            edgecolors="red" if flagged else "none",
                            linewidths=2 if flagged else 0)
        ax.set_title(f"{name}\n"
                        f"clusters={summary['n_clusters']} "
                        f"solo={summary['n_solo']} "
                        f"flagged={summary.get('n_flagged', 0)}",
                        fontsize=9)
        ax.ticklabel_format(useOffset=False)
        ax.tick_params(labelsize=7)

    fig.suptitle("Clustering System - Visual Check "
                    "(gray x = solo | red edge = flagged)", fontsize=12)
    plt.tight_layout()
    plt.show()      


if __name__ == "__main__":
    main()