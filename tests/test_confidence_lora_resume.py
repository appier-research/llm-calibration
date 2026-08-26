from pathlib import Path

import torch
from torch import nn
from transformers import PretrainedConfig

from src.confidence_training.config import RunConfig
from src.confidence_training.models import LLMScoreModel, apply_lora, build_optimizer
from src.confidence_training.training import _load_resume_state, _save_resume_state


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = PretrainedConfig()
        self.config.hidden_size = 4
        for name in (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ):
            setattr(self, name, nn.Linear(4, 4, bias=False, dtype=torch.bfloat16))

    def forward(self, input_ids, attention_mask, return_dict=True, **kwargs):
        del attention_mask, return_dict, kwargs
        hidden = torch.nn.functional.one_hot(input_ids, num_classes=4).to(torch.bfloat16)
        return type("Output", (), {"last_hidden_state": self.q_proj(hidden)})()


def make_model_and_config(tmp_path: Path):
    config = RunConfig(
        run_name="lora-resume",
        dataset_name="dataset",
        model_name="standard-model",
        model_id="model-id",
        adaptation_strategy="lora",
        train_dir="train",
        validation_dir="validation",
        output_dir=str(tmp_path),
        learning_rate=3e-5,
        batch_size=16,
    )
    config.validate()
    backbone, _ = apply_lora(TinyBackbone(), config)
    model = LLMScoreModel(backbone)
    return model, config, build_optimizer(model, config)


def test_lora_resume_state_contains_only_adapter_head_and_restores_them(tmp_path: Path) -> None:
    torch.manual_seed(7)
    model, config, optimizer = make_model_and_config(tmp_path)
    expected_head = model.regression_head.weight.detach().clone()
    expected_adapters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if "lora_" in name
    }
    path = tmp_path / "last.pt"
    _save_resume_state(
        path,
        epoch=2,
        optimizer_steps=11,
        model=model,
        optimizer=optimizer,
        config=config,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["resume_state_version"] == 3
    assert payload["model"]["kind"] == "lora_adapter_and_head"
    assert not any("base_layer" in name for name in payload["model"]["adapter"])

    with torch.no_grad():
        model.regression_head.weight.zero_()
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.add_(1)
    assert _load_resume_state(
        path, model=model, optimizer=optimizer, config=config
    ) == (2, 11)
    torch.testing.assert_close(model.regression_head.weight, expected_head)
    for name, parameter in model.named_parameters():
        if name in expected_adapters:
            torch.testing.assert_close(parameter, expected_adapters[name])
