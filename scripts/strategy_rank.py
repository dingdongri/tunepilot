"""GPU 여러 장에서 '전략 선택'이 맞는지 확인 — 전략(DDP · ZeRO-1/2/3 · offload)마다 가장 빠른 설정을 실제로 돌린다.

    python scripts/strategy_rank.py --name q05_2g --model Qwen/Qwen2.5-0.5B --method full --seq 1024 --gpu t4 --gpus 2
    python scripts/strategy_rank.py ... --dry-run          # 예측 표만 (GPU 불필요)

추천 1~3위는 같은 전략에서 micro-batch만 다른 경우가 많아, 전략 사이의 순위는 따로 확인해야 한다.
여기서는 메모리에 들어가는 후보 중 전략별 최고 예측 처리량 설정을 골라 번들로 만들고 run_bundle.py로 실행한다
→ data/strategy_runs_kaggle.jsonl (bundle 이름 = <name>_<전략>) → scripts/strategy_rank_report.py
"""
from __future__ import annotations
import argparse
import math
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from fitcfg import PlanRequest, get_gpu, load_calibration, plan  # noqa: E402
from fitcfg.generate import build_bundle, write_bundle  # noqa: E402
from fitcfg.models import get_model  # noqa: E402
from fitcfg.recommend import _ram_ok  # noqa: E402

ORDER = ["single", "ddp", "zero1", "zero2", "zero3", "zero-offload", "zero3-offload"]


def best_per_strategy(res, req) -> dict:
    """메모리(와 CPU RAM)에 들어가는 후보 중 전략별 1위. 정렬 기준은 plan()과 같다(2% 이내는 동률 → 단순한 쪽)."""
    spec = get_model(req.model)
    best = {}
    for r in res.candidates:
        if not (r.fits and _ram_ok(req, spec, r)):
            continue
        key = (-round(math.log(r.throughput.tokens_per_s) / math.log(1.02)), r.config.optim != "adamw", r.pred_mem_gb)
        if r.strategy not in best or key < best[r.strategy][0]:
            best[r.strategy] = (key, r)
    return {s: best[s][1] for s in ORDER if s in best}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="시나리오 이름 (번들 폴더 runs/<name>_<전략>)")
    p.add_argument("--model", required=True)
    p.add_argument("--method", default="full")
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--gpu", default="t4")
    p.add_argument("--gpus", type=int, default=2)
    p.add_argument("--global-batch", type=int, default=16)
    p.add_argument("--cpu-ram", type=float, default=28)
    p.add_argument("--strategies", nargs="*", default=None, help="돌릴 전략 (기본: 들어가는 전략 전부)")
    p.add_argument("--naive", action="store_true", help="HF 기본값도 실행")
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--log", default=str(REPO / "data" / "strategy_runs_kaggle.jsonl"))
    p.add_argument("--dataset", default="tatsu-lab/alpaca")
    p.add_argument("--text-field", default="text")
    p.add_argument("--dataset-split", default="train[:10000]")
    p.add_argument("--calib", default=str(REPO / "fitcfg" / "calibrated.json"))
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    load_calibration(a.calib)
    req = PlanRequest(model=a.model, gpu=get_gpu(a.gpu), gpus=a.gpus, method=a.method, seq_len=a.seq,
                      global_batch=a.global_batch, cpu_ram_gb=a.cpu_ram, check_3d=False)
    res = plan(req, top_k=3)
    best = best_per_strategy(res, req)
    picks = [s for s in (a.strategies or list(best)) if s in best]
    top = res.best.strategy if res.best else None
    print(f"## {a.name}: {a.model} · {a.method} · seq {a.seq} · {req.gpu.label} ×{a.gpus} "
          f"(들어가는 후보 {res.n_feasible}/{res.n_candidates})", flush=True)
    for s in best:
        r = best[s]
        mark = "→ 추천 1위" if s == top else ("" if s in picks else "(건너뜀)")
        print(f"  {s:14} {r.reason:68} 예측 {r.pred_mem_gb:5.1f}GB · {r.throughput.tokens_per_s:6.0f} tok/s {mark}")
    for s in (a.strategies or []):
        if s not in best:
            print(f"  {s:14} 메모리에 들어가는 설정 없음")
    if a.dry_run:
        return

    jobs = [(f"{a.name}_{s}", best[s], f"strategy:{s}") for s in picks]
    if a.naive:
        jobs.insert(0, (f"{a.name}_naive", res.naive, "naive"))
    done = Path(a.log).read_text(encoding="utf-8") if Path(a.log).exists() else ""
    for bundle, rec, label in jobs:
        if f'"bundle": "{bundle}"' in done:
            print(f"  ({bundle} 이미 완료)")
            continue
        d = write_bundle(build_bundle(rec, req, dataset=a.dataset, text_field=a.text_field,
                                      dataset_split=a.dataset_split), REPO / "runs" / bundle)
        out = subprocess.run([sys.executable, str(REPO / "scripts" / "run_bundle.py"), str(d), "--label", label,
                              "--max-steps", str(a.steps), "--log", a.log, "--timeout", str(a.timeout)],
                             text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
        print("\n".join(out.strip().splitlines()[-3:]), flush=True)


if __name__ == "__main__":
    main()
