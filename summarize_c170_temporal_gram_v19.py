"""Old-validation-only V19 model/family-vote summary (never reads test data)."""
from __future__ import annotations

import csv, json, math
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT / "outputs/c170_temporal_gram_v19_formal_20260930"
V6 = PROJECT / "outputs/c170_temporal_phase_invariant_v6_formal_cuda_20260929"
V7 = PROJECT / "outputs/c170_c4_kernel_ridge_v7_formal_20260929/validation_predictions.csv"
OUT = ROOT / "v19_oldval_summary.json"
LABELS = ("UP", "DOWN", "LEFT", "RIGHT")
LABEL_ID = {label: i for i, label in enumerate(LABELS)}
SEEDS = (17, 29, 43, 61, 89)


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def metrics(y, p):
    cm = [[0]*4 for _ in range(4)]
    for a,b in zip(y,p): cm[a][b] += 1
    per = {}
    f1s=[]; recalls=[]
    for i,label in enumerate(LABELS):
        tp=cm[i][i]; fn=sum(cm[i])-tp; fp=sum(cm[r][i] for r in range(4))-tp
        recall=tp/(tp+fn) if tp+fn else 0.0
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
        per[label]={"recall":recall,"f1":f1,"support":tp+fn,"predicted":tp+fp}
        f1s.append(f1); recalls.append(recall)
    return {"n":len(y),"accuracy":sum(cm[i][i] for i in range(4))/len(y) if y else 0.0,
            "macro_f1":sum(f1s)/4,"per_class":per,"confusion_true_rows_predicted_columns":cm}


def positions(y,p,rows):
    out={}
    for pos in sorted({int(r["distance_mm"]) for r in rows}):
        indices=[i for i,r in enumerate(rows) if int(r["distance_mm"])==pos]
        result=metrics([y[i] for i in indices],[p[i] for i in indices])
        out[str(pos)]={"n":result["n"],"macro_f1":result["macro_f1"],
                       "per_class_recall":{k:v["recall"] for k,v in result["per_class"].items()}}
    return out


def vote(predictions):
    return [min(Counter(row).items(), key=lambda kv:(-kv[1],kv[0]))[0] for row in zip(*predictions)]


def val_rows(run):
    rows=read_csv(run/"validation_predictions.csv")
    if len(rows)!=320: raise RuntimeError(f"old-val row count is not 320: {run}")
    return rows


def main():
    driver=json.loads((ROOT/"launcher_manifest.json").read_text(encoding="utf-8-sig"))
    if driver.get("status")!="completed" or driver.get("completed_count")!=36 or driver.get("failed_count")!=0:
        raise RuntimeError("V19 36-run queue is not complete and failure-free")
    v6_family=[]; base_keys=None; y=None; base_rows=None
    for seed in SEEDS:
        rows=val_rows(V6/f"learned_power__seed-{seed}")
        keys=[(r["sample_index"],r["distance_mm"],r["block"],r["true_direction"]) for r in rows]
        if base_keys is None: base_keys=keys; base_rows=rows; y=[LABEL_ID[r["true_direction"]] for r in rows]
        elif keys!=base_keys: raise RuntimeError("V6 validation sample IDs/order mismatch")
        v6_family.append([LABEL_ID[r["prediction"]] for r in rows])
    v6_vote=vote(v6_family)
    v7_rows=[r for r in read_csv(V7) if r["preprocessor"]=="bandpass_0p5_40" and r["view"]=="bcd" and r["mode"]=="group_true"]
    v7_rows.sort(key=lambda r:int(r["row"]))
    if len(v7_rows)!=320 or [int(r["row"]) for r in v7_rows]!=list(range(320)):
        raise RuntimeError("V7 fixed group_true validation arm is not 320 canonical rows")
    v7_y=[int(r["y_true"]) for r in v7_rows]; v7_vote=[int(r["y_pred"]) for r in v7_rows]
    if v7_y!=y: raise RuntimeError("V7 validation truth/order does not match V6/V19 canonical validation")

    report={"format":"c170-temporal-gram-v19-oldval-summary","validation_count":320,
            "validation_positions_mm":[5,40,90,130],"class_order":LABELS,"test_metrics_computed":False,
            "source_policy":"reads only V19/V6/V7 old-validation prediction CSVs and V19 launcher manifest; no dataset or test split",
            "configs":{},"frozen_v6_power_family_vote":metrics(y,v6_vote),"frozen_v7_group_true":metrics(y,v7_vote),
            "fixed_three_family_fusion":{}}
    for arch in ("gz","gzn","hybrid"):
        for resolution in (64,128):
            key=f"{arch}_L{resolution}"; true_runs=[]; controls=[]; run_metrics=[]; rows0=None
            for seed in SEEDS:
                rows=val_rows(ROOT/f"{arch}_L{resolution}__seed-{seed}")
                keys=[(r["sample_index"],r["distance_mm"],r["block"],r["true_direction"]) for r in rows]
                if keys!=base_keys: raise RuntimeError(f"V19 validation sample IDs/order mismatch for {key} seed={seed}")
                if rows0 is None: rows0=rows
                prediction=[LABEL_ID[r["prediction"]] for r in rows]
                true_runs.append(prediction); run_metrics.append(metrics(y,prediction))
            control_rows=val_rows(ROOT/f"{arch}_L{resolution}_shuffle__seed-17")
            control_keys=[(r["sample_index"],r["distance_mm"],r["block"],r["true_direction"]) for r in control_rows]
            if control_keys!=base_keys: raise RuntimeError(f"V19 shuffled validation sample IDs/order mismatch for {key}")
            control_pred=[LABEL_ID[r["prediction"]] for r in control_rows]
            family=vote(true_runs)
            fixed=vote([family,v6_vote,v7_vote])
            recall_mean={label:mean(x["per_class"][label]["recall"] for x in run_metrics) for label in LABELS}
            recall_sd={label:(stdev(x["per_class"][label]["recall"] for x in run_metrics) if len(run_metrics)>1 else 0.0) for label in LABELS}
            report["configs"][key]={
                "five_seed_run_macro_f1_mean":mean(x["macro_f1"] for x in run_metrics),
                "five_seed_run_macro_f1_sample_sd":stdev(x["macro_f1"] for x in run_metrics),
                "five_seed_mean_class_recall":recall_mean,"five_seed_class_recall_sample_sd":recall_sd,
                "five_seed_family_vote":metrics(y,family),"family_vote_by_validation_position_mm":positions(y,family,rows0),
                "shuffle_seed17":metrics(y,control_pred),"shuffle_seed17_by_validation_position_mm":positions(y,control_pred,control_rows),
                "fixed_v19_v6_power_v7_three_family_vote":metrics(y,fixed),
                "fixed_three_family_by_validation_position_mm":positions(y,fixed,rows0)}
            report["fixed_three_family_fusion"][key]={"macro_f1":metrics(y,fixed)["macro_f1"],"accuracy":metrics(y,fixed)["accuracy"]}
    tmp=OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    tmp.replace(OUT)
    print(json.dumps({"summary":str(OUT),"configs":{k:{"seed_mean":v["five_seed_run_macro_f1_mean"],"seed_sd":v["five_seed_run_macro_f1_sample_sd"],"family_vote":v["five_seed_family_vote"]["macro_f1"],"fixed_three_family":v["fixed_v19_v6_power_v7_three_family_vote"]["macro_f1"],"shuffle":v["shuffle_seed17"]["macro_f1"]} for k,v in report["configs"].items()}},indent=2))


if __name__=="__main__": main()
