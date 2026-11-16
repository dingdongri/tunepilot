"""전략 선택 검증 결과 정리 → reports/strategy_rank_report.md

    python scripts/strategy_rank_report.py [--log data/strategy_runs_kaggle.jsonl]

시나리오(GPU 2장)마다: 전략별 최고 설정의 예측 vs 실측(메모리·속도), 예측 순위 vs 실측 순위,
추천 1위(예측 최고속)를 골랐을 때 실제 최고속 대비 손해, GPU 1장 대비 확장 배수.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--log", default=str(REPO / "data" / "strategy_runs_kaggle.jsonl"))
ap.add_argument("--out", default=str(REPO / "reports" / "strategy_rank_report.md"))
a = ap.parse_args()

NAMES = {"single": "GPU 1장", "ddp": "DDP", "zero1": "ZeRO-1", "zero2": "ZeRO-2", "zero3": "ZeRO-3",
         "zero-offload": "ZeRO-2 + optimizer offload", "zero3-offload": "ZeRO-3 + 파라미터·optimizer offload",
         "naive": "HF 기본값"}
SIMPLE = {"single": 0, "ddp": 0, "zero1": 1, "zero2": 1, "zero3": 2, "zero-offload": 3, "zero3-offload": 4}
TIE = 0.03                                            # 예측 차이 3% 미만인 쌍은 순위 평가에서 제외

rows = [json.loads(l) for l in open(a.log, encoding="utf-8") if l.strip() and not l.startswith("#")]
by = defaultdict(dict)
for r in rows:                                        # 같은 번들을 여러 번 돌렸으면 마지막 결과
    sc2, _, st2 = r["bundle"].rpartition("_")
    for s in sorted(NAMES, key=len, reverse=True):    # 'zero3-offload'처럼 '-'가 든 전략 이름 처리
        if r["bundle"].endswith("_" + s):
            sc2, st2 = r["bundle"][: -len(s) - 1], s
            break
    by[sc2][st2] = r


def cfg(r):
    return (f"micro {r['micro_batch']} × 누적 {r['grad_accum']} · ckpt {'on' if r['gradient_checkpointing'] else 'off'}"
            + (f" · {r['optim_label']}" if r.get("optim_label") and r["optim_label"] != "adamw" else ""))


gpu = next((r.get("gpu_name") for d in by.values() for r in d.values() if r.get("gpu_name")), "") or "GPU"
L = [f"# GPU 2장에서 전략 선택 검증 ({gpu} ×2)\n",
     "전략(DDP · ZeRO-1/2/3 · offload)마다 도구가 고른 가장 빠른 설정을 실제로 학습 (도구가 만든 번들 그대로 · alpaca · "
     "앞 3스텝은 속도 측정 제외). 원자료 `data/strategy_runs_kaggle.jsonl`, 실행 `scripts/strategy_rank_kaggle.sh`\n"]
tot = dict(sc=0, ok=0, n=0, top1=0, top1_n=0, top1_ran=0, pair_ok=0, pair_n=0, naive_oom=0, naive_n=0)
peak_err, tps_err, regrets, body = [], [], [], []
for sc, d in sorted(by.items()):
    multi = {s: r for s, r in d.items() if s not in ("naive", "single")}
    if not multi:
        continue
    first = next(iter(multi.values()))
    tot["sc"] += 1
    body.append(f"\n## {sc} — {first['model_key']} · {first['method']} · seq {first['seq_len']} · GPU {first['gpus']}장\n")
    body.append("| 전략 | 설정 | 결과 | peak 예측 | peak 실측 | 오차 | 속도 예측 | 속도 실측 | 오차 | 예측 순위 | 실측 순위 |")
    body.append("|---|---|---|---|---|---|---|---|---|---|---|")
    # 도구의 정렬 기준과 같게: 예측 속도 2% 이내는 동률 → 단순한 전략 먼저
    pred_rank = sorted(multi, key=lambda s: (-round(math.log(multi[s]["predicted_tokens_per_s"]) / math.log(1.02)),
                                             SIMPLE.get(s, 9)))
    okm = {s: r for s, r in multi.items() if r.get("ok") == 1 and r.get("tokens_per_s")}
    meas_rank = sorted(okm, key=lambda s: -okm[s]["tokens_per_s"])
    if "naive" in d:
        tot["naive_n"] += 1
        tot["naive_oom"] += d["naive"].get("note") == "OOM"
    for s in (["naive"] if "naive" in d else []) + pred_rank:
        r = d[s]
        ok = r.get("ok") == 1
        pk, pt, mk, mt = r["predicted_peak_gb"], r["predicted_tokens_per_s"], r.get("device_peak_gb"), r.get("tokens_per_s")
        ek = f"{(pk - mk) / mk * 100:+.1f}%" if ok and mk else ""
        et = f"{(pt - mt) / mt * 100:+.1f}%" if ok and mt else ""
        if s != "naive":
            tot["n"] += 1
            tot["ok"] += ok
            if ok and mk:
                peak_err.append((pk - mk) / mk * 100)
            if ok and mt:
                tps_err.append((pt - mt) / mt * 100)
        res = "✅ 성공" if ok else f"❌ {(r.get('note') or '')[:40]}"
        pr = "" if s == "naive" else str(pred_rank.index(s) + 1)
        mr = str(meas_rank.index(s) + 1) if s in meas_rank else ""
        body.append(f"| {NAMES.get(s, s)} | {cfg(r)} | {res} | {pk:.1f} | {f'{mk:.2f}' if mk else '-'} | {ek} | "
                    f"{pt:,.0f} | {f'{mt:,.0f}' if mt else '-'} | {et} | {pr} | {mr} |")
    if len(okm) >= 2:
        pick, fastest = pred_rank[0], meas_rank[0]                 # 도구가 고르는 전략 = 예측 1위
        tot["top1_n"] += 1
        tot["top1_ran"] += pick in okm
        if pick in okm:
            regret = (okm[fastest]["tokens_per_s"] - okm[pick]["tokens_per_s"]) / okm[fastest]["tokens_per_s"] * 100
            regrets.append(regret)
            tot["top1"] += regret <= 3.0
        ss = list(okm)
        for i in range(len(ss)):
            for j in range(i + 1, len(ss)):
                pi, pj = okm[ss[i]]["predicted_tokens_per_s"], okm[ss[j]]["predicted_tokens_per_s"]
                if abs(pi - pj) / max(pi, pj) < TIE:
                    continue
                tot["pair_n"] += 1
                tot["pair_ok"] += (pi > pj) == (okm[ss[i]]["tokens_per_s"] > okm[ss[j]]["tokens_per_s"])
        if pick in okm:
            body.append(f"\n- 도구가 고른 전략: **{NAMES[pick]}** · 실제 가장 빠른 전략: **{NAMES[fastest]}** "
                        f"→ 손해 {regret:.1f}%" + (" ✅" if regret <= 3.0 else " ⚠️"))
        else:
            body.append(f"\n- 도구가 고른 전략: **{NAMES[pick]}** → ❌ 학습 실패({(multi[pick].get('note') or '')[:20]}) · "
                        f"실제 가장 빠른 전략: **{NAMES[fastest]}**")
    one = by.get(sc.replace("_2g", "_1g"), {}).get("single")
    if one and one.get("ok") == 1 and one.get("tokens_per_s") and okm:
        b = max(okm.values(), key=lambda r: r["tokens_per_s"])
        pb = max(multi.values(), key=lambda r: r["predicted_tokens_per_s"])
        body.append(f"- GPU 1장({cfg(one)}) {one['tokens_per_s']:,.0f} tok/s (예측 {one['predicted_tokens_per_s']:,.0f}) → "
                    f"GPU 2장 최고 {b['tokens_per_s']:,.0f} tok/s = **{b['tokens_per_s'] / one['tokens_per_s']:.2f}배** "
                    f"(예측 {pb['predicted_tokens_per_s'] / one['predicted_tokens_per_s']:.2f}배)")

mean = lambda v: sum(map(abs, v)) / len(v) if v else float("nan")  # noqa: E731
rng = lambda v: f"{min(v):+.1f} ~ {max(v):+.1f}%" if v else "-"    # noqa: E731
S = ["\n## 요약\n", "| 항목 | 결과 |", "|---|---|",
     f"| 시나리오 (GPU 2장 · Full) | {tot['sc']}개 |",
     f"| HF 기본값 → OOM | {tot['naive_oom']}/{tot['naive_n']} |",
     f"| 전략별 최고 설정 학습 성공 | **{tot['ok']}/{tot['n']}** |",
     f"| 도구가 고른 전략(예측 1위)이 학습 성공 | **{tot['top1_ran']}/{tot['top1_n']}** |",
     f"| 도구가 고른 전략이 실제 최고속(3% 이내) | **{tot['top1']}/{tot['top1_n']}**"
     + (f" (손해 최대 {max(regrets):.1f}%)" if regrets else "") + " |",
     f"| 두 전략의 빠르기 순서가 맞음 (예측 차이 3% 이상인 쌍) | **{tot['pair_ok']}/{tot['pair_n']}** |",
     f"| peak 메모리 예측 오차 | 평균 {mean(peak_err):.1f}% (범위 {rng(peak_err)}) |",
     f"| 속도 예측 오차 | 평균 {mean(tps_err):.1f}% (범위 {rng(tps_err)}) |"]
Path(a.out).write_text("\n".join(L + S + body) + "\n", encoding="utf-8")
print("\n".join(S))
print(f"-> {a.out}")
