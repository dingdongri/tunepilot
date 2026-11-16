#!/usr/bin/env bash
# GPU 2장에서 전략 선택 검증 (Kaggle 무료 T4 ×2): Full 학습에서 DDP · ZeRO-1 · ZeRO-2 · ZeRO-3 · offload 를
#   전략마다 가장 빠른 설정으로 실제 학습 → 예측 순위 = 실측 순위인가, 추천 1위가 실제로 가장 빠른가
#   1B Full은 T4 ×2에 들어가는 전략이 거의 없어, 여러 전략이 모두 들어가는 0.5B·0.36B 모델(처음 보는 모델)로 확인
#   사용법 (Kaggle 노트북): !bash scripts/strategy_rank_kaggle.sh   → data/strategy_runs_kaggle.jsonl
set -u
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
STEPS=${STEPS:-12}

# 이름  모델  seq  GPU수  전략(all = 들어가는 전략 전부)
SCENARIOS=(
  "q05_2g      Qwen/Qwen2.5-0.5B           1024 2  all"
  "q05_1g      Qwen/Qwen2.5-0.5B           1024 1  single"
  "s360_2g     HuggingFaceTB/SmolLM2-360M  1024 2  ddp zero1 zero2 zero3 zero-offload"
  "s360_1g     HuggingFaceTB/SmolLM2-360M  1024 1  single"
  # fp16 혼합정밀 Full(DeepSpeed 없음: DDP · GPU 1장) 경로 추가 확인 — seq 512
  "q05s512_2g  Qwen/Qwen2.5-0.5B           512  2  ddp zero1"
  "q05s512_1g  Qwen/Qwen2.5-0.5B           512  1  single"
  "s360s512_2g HuggingFaceTB/SmolLM2-360M  512  2  ddp zero1"
  "s360s512_1g HuggingFaceTB/SmolLM2-360M  512  1  single"
)
for sc in "${SCENARIOS[@]}"; do
  read -r name model seq ngpu strategies <<< "$sc"
  echo; echo "################ $(date '+%H:%M:%S') $name"
  extra=(); [ "$strategies" != all ] && extra=(--strategies $strategies)
  [ "$ngpu" = 2 ] && extra+=(--naive)
  python scripts/strategy_rank.py --name "$name" --model "$model" --method full --seq "$seq" --gpu t4 \
      --gpus "$ngpu" --steps "$STEPS" "${extra[@]}" 2>&1 | grep -vE 'Warning|warn'
done
python scripts/strategy_rank_report.py || true
echo "=== 완료 $(date '+%H:%M:%S') → data/strategy_runs_kaggle.jsonl ==="
