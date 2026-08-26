from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.confidence_training.config import (
    GPT_OSS_LORA_TARGET_MODULES,
    GPT_OSS_LORA_TARGET_PARAMETERS,
    RunConfig,
    resolve_run_config,
)
from src.confidence_training.losses import confidence_loss, ranknet_loss
from src.confidence_training.models import (
    LLMScoreModel,
    LinearScoreModel,
    Mxfp4ExpertLora,
    build_optimizer,
    load_score_model,
    matched_module_suffixes,
    matched_parameter_suffixes,
    parameter_report,
    save_score_model,
)


def config(strategy: str) -> RunConfig:
    value = RunConfig(
        run_name="run",
        dataset_name="dataset",
        model_name="model",
        model_id="model-id",
        adaptation_strategy=strategy,
        train_dir="train",
        validation_dir="validation",
        output_dir="output",
        learning_rate=2e-4,
    )
    value.validate()
    return value


def test_losses_use_scores_and_soft_targets() -> None:
    scores = torch.tensor([0.0, 1.0], requires_grad=True)
    targets = torch.tensor([0.25, 0.75])
    mse = confidence_loss(scores, targets, "mse")
    expected = torch.mean((torch.sigmoid(scores) - targets) ** 2)
    assert torch.allclose(mse, expected)
    bce = confidence_loss(scores, targets, "bce")
    assert bce.item() > 0
    bce.backward()
    assert scores.grad is not None


def test_ranknet_direction_ties_and_pair_count() -> None:
    correct = ranknet_loss(torch.tensor([0.0, 2.0]), torch.tensor([0.0, 1.0]))
    reversed_loss = ranknet_loss(torch.tensor([2.0, 0.0]), torch.tensor([0.0, 1.0]))
    assert correct < reversed_loss
    tied_scores = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    tied = ranknet_loss(tied_scores, torch.ones(3))
    assert tied.item() == 0.0
    tied.backward()
    assert torch.equal(tied_scores.grad, torch.zeros_like(tied_scores))
    with pytest.raises(ValueError):
        ranknet_loss(torch.ones(2), torch.ones(2), temperature=0)


def test_standardization_uses_population_std_and_clamp() -> None:
    model = LinearScoreModel(2)
    values = torch.tensor([[1.0, 2.0], [3.0, 2.0]])
    model.set_standardization(values, epsilon=1e-6)
    assert torch.allclose(model.input_mean, torch.tensor([2.0, 2.0]))
    assert torch.allclose(model.input_std, torch.tensor([1.0, 1e-6]))


def test_standardized_linear_checkpoint_round_trip(tmp_path) -> None:
    original = LinearScoreModel(2)
    original.set_standardization(
        torch.tensor([[1.0, 2.0], [3.0, 6.0]]),
        epsilon=1e-6,
    )
    with torch.no_grad():
        original.regression_head.weight.copy_(torch.tensor([[0.25, -0.5]]))
        original.regression_head.bias.copy_(torch.tensor([0.75]))
    checkpoint = tmp_path / "model"
    save_score_model(original, checkpoint, config("linear_probe"))

    restored, tokenizer = load_score_model(
        checkpoint,
        config("linear_probe"),
        device=torch.device("cpu"),
    )

    assert tokenizer is None
    assert torch.equal(restored.input_mean, original.input_mean)
    assert torch.equal(restored.input_std, original.input_std)
    features = torch.tensor([[2.0, 4.0], [5.0, 8.0]])
    assert torch.equal(
        restored.forward_scores(features), original.forward_scores(features)
    )


def test_optimizer_protocol_and_parameter_membership() -> None:
    linear = LinearScoreModel(3)
    sgd = build_optimizer(linear, config("linear_probe"))
    assert isinstance(sgd, torch.optim.SGD)
    assert sgd.defaults["momentum"] == 0.9
    assert sgd.defaults["weight_decay"] == 0.01
    full = LinearScoreModel(3)
    adamw = build_optimizer(full, config("full_weight"))
    assert isinstance(adamw, torch.optim.AdamW)
    assert adamw.defaults["betas"] == (0.9, 0.999)
    assert adamw.defaults["eps"] == 1e-8
    report = parameter_report(full, adamw)
    assert report["optimizer_matches_trainable"]
    assert report["trainable_by_component"]["head"] == 4


class FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=2)
        self.proj = nn.Linear(2, 2, dtype=torch.bfloat16)

    def forward(self, input_ids, attention_mask, return_dict=True, **kwargs):
        del return_dict, kwargs
        values = torch.nn.functional.one_hot(input_ids, num_classes=2).to(torch.bfloat16)
        return SimpleNamespace(last_hidden_state=self.proj(values))


def test_llm_head_and_scores_are_fp32_with_bf16_backbone() -> None:
    model = LLMScoreModel(FakeBackbone())
    scores = model.forward_scores(
        input_ids=torch.tensor([[0, 1], [1, 0]]),
        attention_mask=torch.tensor([[1, 1], [0, 1]]),
    )
    assert next(model.backbone.parameters()).dtype == torch.bfloat16
    assert model.regression_head.weight.dtype == torch.float32
    assert scores.dtype == torch.float32


class ModuleTree(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Module()
        self.block.q_proj = nn.Linear(1, 1)


def test_lora_suffix_validation_fails_on_missing() -> None:
    assert matched_module_suffixes(ModuleTree(), ["q_proj"])["q_proj"] == ["block.q_proj"]
    with pytest.raises(ValueError, match="matched no modules"):
        matched_module_suffixes(ModuleTree(), ["k_proj"])


class FusedExperts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(2, 3, 6))
        self.down_proj = nn.Parameter(torch.zeros(2, 3, 3))


class GptOssLikeTree(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Module()
        self.layer.q_proj = nn.Linear(3, 3)
        self.layer.k_proj = nn.Linear(3, 3)
        self.layer.v_proj = nn.Linear(3, 3)
        self.layer.o_proj = nn.Linear(3, 3)
        self.layer.mlp = nn.Module()
        self.layer.mlp.experts = FusedExperts()


def test_gpt_oss_profile_uses_fused_expert_parameters() -> None:
    resolved = resolve_run_config(
        None,
        {
            "run_name": "gpt",
            "dataset_name": "dataset",
            "model_name": "gpt-oss-20b",
            "model_id": "openai/gpt-oss-20b",
            "adaptation_strategy": "lora",
            "train_dir": "train",
            "validation_dir": "validation",
            "output_dir": "output",
        },
    )
    assert resolved.lora_target_modules == GPT_OSS_LORA_TARGET_MODULES
    assert resolved.lora_target_parameters == GPT_OSS_LORA_TARGET_PARAMETERS
    tree = GptOssLikeTree()
    matches = matched_parameter_suffixes(tree, resolved.lora_target_parameters)
    assert matches["mlp.experts.gate_up_proj"] == [
        "layer.mlp.experts.gate_up_proj"
    ]
    with pytest.raises(ValueError, match="matched no parameters"):
        matched_parameter_suffixes(tree, ["mlp.experts.missing"])


def test_mxfp4_expert_lora_routing_matches_manual_matmuls() -> None:
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    lora_a = torch.tensor(
        [[[1.0], [0.0]], [[0.0], [1.0]]], requires_grad=True
    )
    lora_b = torch.tensor(
        [[[2.0, 3.0]], [[4.0, 5.0]]], requires_grad=True
    )
    routing = SimpleNamespace(expt_hist=torch.tensor([2, 1]), n_expts_act=1)
    gather = SimpleNamespace(src_indx=torch.tensor([2, 0, 1]))
    scatter = SimpleNamespace(dst_indx=torch.tensor([2, 0, 1]))
    gammas = torch.tensor([0.5, 0.25, 1.0])
    result = Mxfp4ExpertLora._routed_delta(
        inputs,
        lora_a,
        lora_b,
        routing,
        gather_idx=gather,
        scatter_idx=scatter,
        gammas=gammas,
    )
    expected_sorted = torch.stack(
        [
            (inputs[2] @ lora_a[0]) @ lora_b[0] * 0.5,
            (inputs[0] @ lora_a[0]) @ lora_b[0] * 0.25,
            (inputs[1] @ lora_a[1]) @ lora_b[1],
        ]
    )
    expected = torch.stack([expected_sorted[1], expected_sorted[2], expected_sorted[0]])
    torch.testing.assert_close(result, expected)
    result.sum().backward()
    assert lora_a.grad is not None
    assert lora_b.grad is not None
