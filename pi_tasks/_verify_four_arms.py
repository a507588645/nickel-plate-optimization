import json

s = {a["arm_id"]: a for a in json.load(open("dianji_grade_detector_compare_v1/four_arms_summary.json", encoding="utf-8"))}
m3 = json.load(open("dianji_grade_detector_compare_v1/arm3_slf640/metrics_slf_texture33.json", encoding="utf-8"))
m4 = json.load(open("dianji_grade_detector_compare_v1/arm4_ryw640/metrics_v3_texture_only.json", encoding="utf-8"))


def g(m):
    lgb = m["models"]["lightgbm"] if "models" in m else m
    return round(lgb["test"]["accuracy"], 4), round(lgb["test"]["macro_f1"], 4)


a3 = g(m3)
a4 = g(m4)
print("arm3 metrics json:", a3, " summary:", s["arm3"]["texture_only"]["test_accuracy"], s["arm3"]["texture_only"]["test_macro_f1"])
print("arm4 metrics json:", a4, " summary:", s["arm4"]["texture_only"]["test_accuracy"], s["arm4"]["texture_only"]["test_macro_f1"])
ok3 = abs(a3[0] - s["arm3"]["texture_only"]["test_accuracy"]) < 5e-4 and abs(a3[1] - s["arm3"]["texture_only"]["test_macro_f1"]) < 5e-4
ok4 = abs(a4[0] - s["arm4"]["texture_only"]["test_accuracy"]) < 5e-4 and abs(a4[1] - s["arm4"]["texture_only"]["test_macro_f1"]) < 5e-4
print("MATCH arm3:", ok3, " arm4:", ok4)
