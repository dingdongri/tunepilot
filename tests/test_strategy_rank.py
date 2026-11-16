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


# ---- fp16 혼합정밀 Full (bf16 없는 GPU, DeepSpeed 없음) — Kaggle T4 실측(2026-11-16)에서 추천 설정이 OOM 났던 경로 ----
QWEN05 = {"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2", "hidden_size": 896, "intermediate_size": 4864,
          "num_attention_heads": 14, "num_hidden_layers": 24, "num_key_value_heads": 2, "vocab_size": 151936,
          "tie_word_embeddings": True, "max_position_embeddings": 32768, "hidden_act": "silu"}
T4_USABLE_GB = 14.56


def _calibrated():
    from fitcfg import load_calibration
    load_calibration(os.path.join(ROOT, "fitcfg", "calibrated.json"))


def _est(model, **kw):
    from fitcfg.memory import GB, TrainConfig, estimate
    from fitcfg.models import get_model
    cfg = TrainConfig(model=model, method="full", seq_len=1024, dtype="fp16", flash_attn=False, **kw)
    return estimate(cfg, get_model(model)).total / GB


def test_amp_full_configs_that_oomed_on_t4_no_longer_fit():
    _calibrated()
    register("test/small-360m", SMALL)
    register("test/qwen-05b", QWEN05)
    oomed = [  # (모델, GPU 수, micro-batch, 누적, optimizer) — 모두 실제 T4에서 OOM
        ("test/qwen-05b", 2, 1, 8, "adamw"), ("test/qwen-05b", 1, 2, 8, "adamw8bit"),
        ("test/small-360m", 2, 2, 4, "adamw"), ("test/small-360m", 1, 2, 8, "adamw"),
    ]
    for model, gpus, mb, ga, optim in oomed:
        pred = _est(model, gpus=gpus, batch_size=mb, grad_accum=ga, grad_checkpointing=False, optim=optim)
        assert pred > T4_USABLE_GB, (model, gpus, mb, pred)


def test_zero_full_on_t4_matches_kaggle_measurements():
    _calibrated()
    register("test/small-360m", SMALL)
    register("test/qwen-05b", QWEN05)
    measured = [  # (모델, micro-batch, 누적, ZeRO 단계, 실측 GB) — Kaggle T4 ×2, 2026-11-16
        ("test/qwen-05b", 1, 8, 1, 10.282), ("test/qwen-05b", 1, 8, 2, 9.788),
        ("test/small-360m", 2, 4, 1, 12.356), ("test/small-360m", 2, 4, 2, 12.001),
    ]
    for model, mb, ga, zs, gb in measured:
        pred = _est(model, gpus=2, batch_size=mb, grad_accum=ga, grad_checkpointing=False, zero_stage=zs)
        assert -0.03 < (pred - gb) / gb < 0.15, (model, zs, pred, gb)      # 예전: 최대 −16.6%


def test_amp_terms_do_not_touch_bf16_or_zero_paths():
    from fitcfg.memory import TrainConfig, features
    from fitcfg.models import get_model
    m = get_model("llama-3.2-1b")
    for kw in (dict(dtype="bf16"), dict(dtype="fp16", zero_stage=2, gpus=2), dict(dtype="fp16", method="lora")):
        f = features(TrainConfig(**{"model": "llama-3.2-1b", "method": "full", "grad_accum": 8, **kw}), m)
        assert f.amp_h_basis == f.amp_i_basis == f.amp_known == f.accum_grads == 0.0
    f = features(TrainConfig(model="llama-3.2-1b", method="full", dtype="fp16", grad_accum=8), m)
    assert f.amp_h_basis > 0 and f.accum_grads == f.known_gradients > 0
    assert features(TrainConfig(model="llama-3.2-1b", method="full", dtype="fp16"), m).accum_grads == 0.0


def test_recommended_t4_full_strategy_is_not_plain_ddp_at_the_edge():
    """수정 후 T4 ×2 · 0.5B Full의 추천 1위는 실측 최고속이었던 ZeRO-1."""
    _calibrated()
    register("test/qwen-05b", QWEN05)
    req = PlanRequest(model="test/qwen-05b", gpu=get_gpu("t4"), gpus=2, method="full", seq_len=1024,
                      cpu_ram_gb=28, check_3d=False)
    assert plan(req).best.strategy == "zero1"
