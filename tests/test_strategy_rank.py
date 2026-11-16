"""전략별 최고 설정 고르기 (scripts/strategy_rank.py) — GPU 불필요.  실행: python -m pytest -q tests/test_strategy_rank.py"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from fitcfg import PlanRequest, get_gpu, plan, reset_calibration
from fitcfg.autospec import register
import strategy_rank as SR

SMALL = {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": 960, "intermediate_size": 2560,
         "num_attention_heads": 15, "num_hidden_layers": 32, "num_key_value_heads": 5, "vocab_size": 49152,
         "tie_word_embeddings": True, "max_position_embeddings": 8192, "hidden_act": "silu"}


def setup_function(_):
    reset_calibration()


def test_best_per_strategy_matches_plan_top():
    register("test/small-360m", SMALL)
    req = PlanRequest(model="test/small-360m", gpu=get_gpu("t4"), gpus=2, method="full", seq_len=1024,
                      cpu_ram_gb=28, check_3d=False)
    res = plan(req, top_k=3)
    best = SR.best_per_strategy(res, req)
    assert {"ddp", "zero1", "zero2"} <= set(best) and "single" not in best
    assert all(r.fits and r.strategy == s for s, r in best.items())
    # 추천 1위는 그 전략의 최고 설정과 같아야 한다
    top = res.best
    assert best[top.strategy].config == top.config
    # 전략별 최고는 같은 전략의 다른 들어가는 후보보다 느리지 않다 (2% 동률 허용)
    for s, r in best.items():
        others = [c.throughput.tokens_per_s for c in res.candidates if c.strategy == s and c.fits]
        assert r.throughput.tokens_per_s >= max(others) / 1.021
