"""Versioned configuration for confidence-model experiments."""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping


CONFIG_SCHEMA_VERSION = 1
ADAPTATION_STRATEGIES = ("linear_probe", "lora", "full_weight")
LOSS_TYPES = ("mse", "bce", "ranknet")
FULL_WEIGHT_MODEL_SPECS = {
    "Olmo-3-7B-Instruct": {
        "model_id": "allenai/Olmo-3-7B-Instruct",
        "revision": "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc",
        "backbone_parameters": 6_887_272_448,
    },
    "Qwen3-8B-non-thinking": {
        "model_id": "Qwen/Qwen3-8B",
        "revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "backbone_parameters": 7_568_405_504,
    },
}
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
GPT_OSS_LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
GPT_OSS_LORA_TARGET_PARAMETERS = (
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
)


def lora_target_profile(model_name: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the fixed architecture-equivalent LoRA target profile."""
    if "gpt-oss" in model_name.casefold():
        return GPT_OSS_LORA_TARGET_MODULES, GPT_OSS_LORA_TARGET_PARAMETERS
    return LORA_TARGET_MODULES, ()


@dataclass(slots=True)
class RunConfig:
    """Fully resolved configuration for exactly one training run."""

    schema_version: int = CONFIG_SCHEMA_VERSION
    run_name: str = ""
    dataset_name: str = ""
    model_name: str = ""
    model_id: str = ""
    model_revision: str | None = None
    adaptation_strategy: Literal["linear_probe", "lora", "full_weight"] = "linear_probe"
    loss: Literal["mse", "bce", "ranknet"] = "mse"
    train_dir: str = ""
    validation_dir: str = ""
    output_dir: str = ""
    probing_dataset_name: str = "probing_dataset_cc"
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 1
    seed: int = 42
    standardize: bool = False
    standardization_epsilon: float = 1e-6
    weight_decay: float = 0.01
    sgd_momentum: float = 0.9
    adamw_betas: tuple[float, float] = (0.9, 0.999)
    adamw_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    ranknet_temperature: float = 1.0
    ranknet_tie_loss_weight: float = 0.0
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = LORA_TARGET_MODULES
    lora_target_parameters: tuple[str, ...] = ()
    device: str = "cuda"
    max_sequence_length: int | None = None
    gradient_checkpointing: bool = False
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    train_chat_template_kwargs: dict[str, Any] | None = None
    validation_chat_template_kwargs: dict[str, Any] | None = None
    num_workers: int = 0
    log_every_steps: int | None = None
    save_resume_state: bool = False
    resume: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_mode: Literal["online", "offline", "disabled"] = "disabled"
    validate_every_k_steps: int = 10

    @property
    def train_path(self) -> Path:
        return Path(self.train_dir)

    @property
    def validation_path(self) -> Path:
        return Path(self.validation_dir)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["adamw_betas"] = list(self.adamw_betas)
        value["lora_target_modules"] = list(self.lora_target_modules)
        value["lora_target_parameters"] = list(self.lora_target_parameters)
        return value

    def scientific_dict(self) -> dict[str, Any]:
        """Return fields that must match for an exact resume."""
        value = self.to_dict()
        for key in (
            "output_dir",
            "resume",
            "save_resume_state",
            "wandb_project",
            "wandb_entity",
            "wandb_mode",
            "log_every_steps",
        ):
            value.pop(key, None)
        return value

    def scientific_hash(self) -> str:
        payload = json.dumps(
            self.scientific_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def validate(self, *, require_paths: bool = True) -> None:
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported config schema_version={self.schema_version}; "
                f"expected {CONFIG_SCHEMA_VERSION}."
            )
        if self.adaptation_strategy not in ADAPTATION_STRATEGIES:
            raise ValueError(f"Unknown adaptation strategy: {self.adaptation_strategy}")
        if self.loss not in LOSS_TYPES:
            raise ValueError(f"Unknown loss: {self.loss}")
        if require_paths:
            for name in (
                "run_name",
                "dataset_name",
                "model_name",
                "model_id",
                "train_dir",
                "validation_dir",
                "output_dir",
            ):
                if not getattr(self, name):
                    raise ValueError(f"{name} is required")
        for name, value in (
            ("learning_rate", self.learning_rate),
            ("weight_decay", self.weight_decay),
            ("adamw_epsilon", self.adamw_epsilon),
            ("max_grad_norm", self.max_grad_norm),
            ("ranknet_temperature", self.ranknet_temperature),
            ("standardization_epsilon", self.standardization_epsilon),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("batch_size and epochs must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.log_every_steps is not None:
            if self.log_every_steps <= 0:
                raise ValueError("log_every_steps must be positive")
            if self.adaptation_strategy == "linear_probe":
                raise ValueError(
                    "--log-every-steps cannot be used with adaptation_strategy=linear_probe"
                )
        if self.validate_every_k_steps <= 0:
            raise ValueError("validate_every_k_steps must be positive")
        if not 0.0 <= self.sgd_momentum < 1.0:
            raise ValueError("sgd_momentum must be in [0, 1)")
        if len(self.adamw_betas) != 2 or not all(
            0.0 <= beta < 1.0 for beta in self.adamw_betas
        ):
            raise ValueError("adamw_betas must contain two values in [0, 1)")
        if self.ranknet_tie_loss_weight != 0.0:
            raise ValueError("This experiment fixes ranknet_tie_loss_weight to 0.0")
        if self.ranknet_temperature != 1.0:
            raise ValueError("This experiment fixes ranknet_temperature to 1.0")
        if self.weight_decay != 0.01:
            raise ValueError("This experiment fixes weight_decay to 0.01")
        if self.max_grad_norm != 1.0:
            raise ValueError("This experiment fixes max_grad_norm to 1.0")
        if self.adaptation_strategy != "linear_probe" and self.standardize:
            raise ValueError("Feature standardization applies only to linear probes")
        if self.adaptation_strategy == "full_weight":
            if "gpt-oss" in self.model_name.casefold() or "gpt-oss" in self.model_id.casefold():
                raise ValueError("GPT-OSS/MXFP4 is excluded from full-weight training")
            if self.lora_target_parameters:
                raise ValueError("LoRA-only target parameters are invalid for full_weight")
        if self.adaptation_strategy == "lora":
            expected_modules, expected_parameters = lora_target_profile(self.model_name)
            if (
                self.lora_rank != 16
                or self.lora_alpha != 32
                or self.lora_dropout != 0.0
                or tuple(self.lora_target_modules) != expected_modules
                or tuple(self.lora_target_parameters) != expected_parameters
            ):
                raise ValueError("LoRA settings differ from the fixed experiment protocol")
        for split, kwargs in (
            ("shared", self.chat_template_kwargs),
            ("train", self.train_chat_template_kwargs),
            ("validation", self.validation_chat_template_kwargs),
        ):
            if kwargs is None:
                continue
            template_date = kwargs.get("template_date")
            if template_date is not None:
                try:
                    dt.date.fromisoformat(str(template_date))
                except ValueError as exc:
                    raise ValueError(
                        f"{split} chat template_date must be YYYY-MM-DD"
                    ) from exc
        if self.wandb_mode not in ("online", "offline", "disabled"):
            raise ValueError(f"Unknown wandb_mode: {self.wandb_mode}")


def _coerce_field(name: str, value: Any) -> Any:
    if name in {
        "adamw_betas",
        "lora_target_modules",
        "lora_target_parameters",
    } and value is not None:
        return tuple(value)
    return value


def run_config_from_mapping(mapping: Mapping[str, Any]) -> RunConfig:
    values = dict(mapping)
    values.pop("_provenance", None)
    values.pop("_runtime", None)
    known = {item.name for item in dataclasses.fields(RunConfig)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"Unknown run config fields: {unknown}")
    if values.get("adaptation_strategy") == "lora" and values.get("model_name"):
        modules, parameters = lora_target_profile(str(values["model_name"]))
        values.setdefault("lora_target_modules", modules)
        values.setdefault("lora_target_parameters", parameters)
    values = {key: _coerce_field(key, value) for key, value in values.items()}
    config = RunConfig(**values)
    config.validate()
    return config


def resolve_run_config(
    config_path: str | Path | None,
    overrides: Mapping[str, Any],
    *,
    require_paths: bool = True,
) -> RunConfig:
    values: dict[str, Any] = {}
    if config_path is not None:
        loaded = json.loads(Path(config_path).read_text())
        if not isinstance(loaded, dict):
            raise ValueError("Run configuration must be a JSON object")
        loaded.pop("_provenance", None)
        loaded.pop("_runtime", None)
        values.update(loaded)
    values.update({key: value for key, value in overrides.items() if value is not None})
    known = {item.name for item in dataclasses.fields(RunConfig)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"Unknown run config fields: {unknown}")
    if values.get("adaptation_strategy") == "lora" and values.get("model_name"):
        modules, parameters = lora_target_profile(str(values["model_name"]))
        values.setdefault("lora_target_modules", modules)
        values.setdefault("lora_target_parameters", parameters)
    values = {key: _coerce_field(key, value) for key, value in values.items()}
    config = RunConfig(**values)
    config.validate(require_paths=require_paths)
    return config
