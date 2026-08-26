"""Unified scalar-score models, optimizer construction, and parameter reports."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn

from .config import FULL_WEIGHT_MODEL_SPECS, LORA_TARGET_MODULES, RunConfig
from .data import last_non_padding_indices


class LinearScoreModel(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.regression_head = nn.Linear(input_dim, 1, dtype=torch.float32)
        nn.init.xavier_uniform_(self.regression_head.weight)
        nn.init.zeros_(self.regression_head.bias)
        self.register_buffer("input_mean", torch.empty(0, dtype=torch.float32))
        self.register_buffer("input_std", torch.empty(0, dtype=torch.float32))

    def set_standardization(
        self,
        features: torch.Tensor,
        *,
        epsilon: float,
    ) -> None:
        values = features.float()
        self.input_mean = values.mean(dim=0)
        self.input_std = values.std(dim=0, correction=0).clamp_min(epsilon)

    def forward_scores(self, features: torch.Tensor) -> torch.Tensor:
        values = features.float()
        if self.input_mean.numel():
            values = (values - self.input_mean) / self.input_std
        return self.regression_head(values).squeeze(-1).float()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_scores(features))


class LLMScoreModel(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        config = getattr(backbone, "config", None)
        if config is None and hasattr(backbone, "get_base_model"):
            config = backbone.get_base_model().config
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Backbone configuration must expose hidden_size")
        self.regression_head = nn.Linear(int(hidden_size), 1, dtype=torch.float32)

    def forward_scores(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs,
        )
        hidden = outputs.last_hidden_state
        indices = last_non_padding_indices(attention_mask)
        batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
        pooled = hidden[batch_indices, indices].float()
        return self.regression_head(pooled).squeeze(-1).float()

    def forward(self, **inputs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_scores(**inputs))


class Mxfp4ExpertLora(nn.Module):
    """Exact low-rank deltas around a frozen native-MXFP4 GPT-OSS expert."""

    def __init__(self, base_layer: nn.Module, *, rank: int, alpha: int) -> None:
        super().__init__()
        self.base_layer = base_layer
        experts = int(base_layer.num_experts)
        hidden = int(base_layer.hidden_size)
        intermediate = int(base_layer.intermediate_size)
        self.scaling = alpha / rank
        self.lora_gate_A = nn.Parameter(
            torch.empty(experts, hidden, rank, dtype=torch.bfloat16)
        )
        self.lora_gate_B = nn.Parameter(
            torch.zeros(experts, rank, 2 * intermediate, dtype=torch.bfloat16)
        )
        self.lora_down_A = nn.Parameter(
            torch.empty(experts, intermediate, rank, dtype=torch.bfloat16)
        )
        self.lora_down_B = nn.Parameter(
            torch.zeros(experts, rank, hidden, dtype=torch.bfloat16)
        )
        for expert in range(experts):
            nn.init.kaiming_uniform_(self.lora_gate_A[expert].transpose(0, 1))
            nn.init.kaiming_uniform_(self.lora_down_A[expert].transpose(0, 1))

    @staticmethod
    def _scatter_routes(
        values: torch.Tensor,
        scatter_idx: Any,
        n_experts_active: int,
    ) -> torch.Tensor:
        route_positions = scatter_idx.dst_indx[: len(values)]
        valid = route_positions >= 0
        ordered = torch.zeros(
            scatter_idx.dst_indx.numel(),
            values.shape[-1],
            device=values.device,
            dtype=values.dtype,
        )
        ordered = ordered.index_copy(
            0, route_positions[valid].long(), values[valid]
        )
        return ordered.reshape(-1, n_experts_active, values.shape[-1]).sum(dim=1)

    @staticmethod
    def _routed_delta(
        inputs: torch.Tensor,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        routing_data: Any,
        *,
        gather_idx: Any | None = None,
        scatter_idx: Any | None = None,
        gammas: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reference routed BF16 matmuls with gradients for low-rank weights."""
        sizes = [int(value) for value in routing_data.expt_hist.detach().cpu()]
        chunks: list[torch.Tensor] = []
        offset = 0
        for expert, size in enumerate(sizes):
            if gather_idx is None:
                current = inputs[offset : offset + size]
            else:
                source = (
                    gather_idx.src_indx[offset : offset + size]
                    // routing_data.n_expts_act
                )
                current = inputs[source]
            chunks.append((current @ lora_a[expert]) @ lora_b[expert])
            offset += size
        delta = torch.cat(chunks, dim=0)
        if gammas is not None:
            delta = delta * gammas[: len(delta), None]
        if scatter_idx is None:
            return delta
        return Mxfp4ExpertLora._scatter_routes(
            delta, scatter_idx, routing_data.n_expts_act
        )

    def forward(self, hidden_states, routing_data, gather_idx, scatter_idx):
        from transformers.integrations import mxfp4

        matmul = mxfp4.triton_kernels_hub.matmul_ogs.matmul_ogs
        base = self.base_layer
        gate_up = matmul(
            hidden_states,
            base.gate_up_proj,
            base.gate_up_proj_bias.to(torch.float32),
            routing_data,
            gather_indx=gather_idx,
            precision_config=base.gate_up_proj_precision_config,
        )
        gate_delta = self._routed_delta(
            hidden_states,
            self.lora_gate_A,
            self.lora_gate_B,
            routing_data,
            gather_idx=gather_idx,
        )
        gate_up = gate_up + gate_delta * self.scaling
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(max=base.limit)
        up = up.clamp(min=-base.limit, max=base.limit)
        intermediate = (up + 1) * gate * torch.sigmoid(gate * base.alpha)

        output_routes = matmul(
            intermediate,
            base.down_proj,
            base.down_proj_bias.to(torch.float32),
            routing_data,
            precision_config=base.down_proj_precision_config,
            gammas=routing_data.gate_scal,
        )
        output = self._scatter_routes(
            output_routes, scatter_idx, routing_data.n_expts_act
        )
        down_delta = self._routed_delta(
            intermediate,
            self.lora_down_A,
            self.lora_down_B,
            routing_data,
            scatter_idx=scatter_idx,
            gammas=routing_data.gate_scal,
        )
        return output + down_delta * self.scaling


def scores_to_probabilities(scores: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(scores.float())


def matched_module_suffixes(
    model: nn.Module,
    suffixes: Iterable[str] = LORA_TARGET_MODULES,
) -> dict[str, list[str]]:
    names = [name for name, _ in model.named_modules()]
    result = {
        suffix: sorted(name for name in names if name == suffix or name.endswith(f".{suffix}"))
        for suffix in suffixes
    }
    missing = [suffix for suffix, matches in result.items() if not matches]
    if missing:
        raise ValueError(f"LoRA target suffixes matched no modules: {missing}")
    return result


def matched_parameter_suffixes(
    model: nn.Module,
    suffixes: Iterable[str],
) -> dict[str, list[str]]:
    names = [name for name, _ in model.named_parameters()]
    result = {
        suffix: sorted(name for name in names if name == suffix or name.endswith(f".{suffix}"))
        for suffix in suffixes
    }
    missing = [suffix for suffix, matches in result.items() if not matches]
    if missing:
        raise ValueError(f"LoRA target suffixes matched no parameters: {missing}")
    return result


def _native_mxfp4_expert_modules(model: nn.Module) -> dict[str, nn.Module]:
    return {
        name: module
        for name, module in model.named_modules()
        if module.__class__.__name__ == "Mxfp4GptOssExperts"
    }


def _quantization_evidence(model: nn.Module) -> list[str]:
    evidence: list[str] = []
    configuration = getattr(model, "config", None)
    quantization = getattr(configuration, "quantization_config", None)
    if quantization is not None:
        evidence.append(f"config.quantization_config={quantization!r}")
    if bool(getattr(model, "is_quantized", False)):
        evidence.append("model.is_quantized=True")
    quantized_markers = ("mxfp4", "quantized", "quantlinear", "bitsandbytes")
    for name, module in model.named_modules():
        class_name = module.__class__.__name__.casefold()
        if any(marker in class_name for marker in quantized_markers):
            evidence.append(f"module {name or '<root>'} has class {module.__class__.__name__}")
    return evidence


def prepare_full_weight_backbone(
    backbone: nn.Module,
    *,
    expected_parameters: int | None = None,
    enforce_bf16: bool = True,
) -> int:
    """Make every logical backbone parameter trainable or fail loudly."""

    evidence = _quantization_evidence(backbone)
    if evidence:
        raise ValueError("Quantized full-weight backbones are unsupported: " + "; ".join(evidence))
    initially_frozen = [
        name for name, parameter in backbone.named_parameters() if not parameter.requires_grad
    ]
    logical = 0
    for name, parameter in backbone.named_parameters():
        if not parameter.is_floating_point():
            raise ValueError(
                f"Full-weight backbone parameter {name} has non-floating dtype {parameter.dtype}"
            )
        if enforce_bf16 and parameter.dtype != torch.bfloat16:
            parameter.data = parameter.data.to(dtype=torch.bfloat16)
        parameter.requires_grad_(True)
        logical += parameter.numel()
    if initially_frozen:
        raise ValueError(
            "Full-weight backbone arrived with frozen logical parameters: "
            f"{initially_frozen[:10]}"
        )
    if logical == 0:
        raise ValueError("Full-weight backbone has no logical parameters")
    frozen = [name for name, parameter in backbone.named_parameters() if not parameter.requires_grad]
    wrong_dtype = [
        f"{name}:{parameter.dtype}"
        for name, parameter in backbone.named_parameters()
        if enforce_bf16 and parameter.dtype != torch.bfloat16
    ]
    if frozen:
        raise ValueError(f"Full-weight backbone contains frozen parameters: {frozen[:10]}")
    if wrong_dtype:
        raise ValueError(f"Full-weight backbone contains non-BF16 parameters: {wrong_dtype[:10]}")
    if expected_parameters is not None and logical != expected_parameters:
        raise ValueError(
            f"Pinned backbone parameter count drift: {logical} != {expected_parameters}"
        )
    return logical


def _install_native_mxfp4_expert_lora(model: nn.Module, config: RunConfig) -> list[str]:
    matches = _native_mxfp4_expert_modules(model)
    for name, module in list(matches.items()):
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(
            parent,
            child_name,
            Mxfp4ExpertLora(module, rank=config.lora_rank, alpha=config.lora_alpha),
        )
    return sorted(matches)


def apply_lora(backbone: nn.Module, config: RunConfig) -> tuple[nn.Module, dict[str, Any]]:
    native_mxfp4 = _native_mxfp4_expert_modules(backbone)
    if config.lora_target_parameters and native_mxfp4:
        parameter_matches = {
            suffix: [f"{name}.{suffix.rsplit('.', 1)[-1]}" for name in sorted(native_mxfp4)]
            for suffix in config.lora_target_parameters
        }
        peft_target_parameters = None
    else:
        parameter_matches = (
            matched_parameter_suffixes(backbone, config.lora_target_parameters)
            if config.lora_target_parameters
            else {}
        )
        peft_target_parameters = list(config.lora_target_parameters) or None
    matches = {
        "modules": matched_module_suffixes(backbone, config.lora_target_modules),
        "parameters": parameter_matches,
    }
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("LoRA training requires the 'peft' package") from error
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
        target_parameters=peft_target_parameters,
        bias="none",
    )
    adapted = get_peft_model(backbone, peft_config, autocast_adapter_dtype=False)
    if native_mxfp4:
        installed = _install_native_mxfp4_expert_lora(adapted, config)
        if len(installed) != len(native_mxfp4):
            raise RuntimeError("Failed to install LoRA on every native MXFP4 expert")
    for name, parameter in adapted.named_parameters():
        if "lora_" in name and parameter.dtype != torch.bfloat16:
            parameter.data = parameter.data.to(torch.bfloat16)
    return adapted, matches


def native_expert_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if any(
            marker in name
            for marker in ("lora_gate_A", "lora_gate_B", "lora_down_A", "lora_down_B")
        )
    }


def load_native_expert_lora_state_dict(
    model: nn.Module, state: dict[str, torch.Tensor]
) -> None:
    if not state:
        return
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = [name for name in incompatible.unexpected_keys if name in state]
    if unexpected:
        raise ValueError(f"Unexpected native expert LoRA checkpoint keys: {unexpected}")


def load_llm_score_model(config: RunConfig) -> tuple[LLMScoreModel, Any, dict[str, Any] | None]:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs a pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModel.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    if hasattr(backbone.config, "use_cache"):
        backbone.config.use_cache = False
    if config.gradient_checkpointing:
        if not hasattr(backbone, "gradient_checkpointing_enable"):
            raise ValueError("Backbone does not support gradient checkpointing")
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(backbone, "enable_input_require_grads"):
            backbone.enable_input_require_grads()
    matches = None
    if config.adaptation_strategy == "lora":
        backbone, matches = apply_lora(backbone, config)
    elif config.adaptation_strategy == "full_weight":
        spec = FULL_WEIGHT_MODEL_SPECS.get(config.model_name)
        expected = int(spec["backbone_parameters"]) if spec is not None else None
        prepare_full_weight_backbone(backbone, expected_parameters=expected)
    else:
        raise ValueError(f"Cannot load LLM for {config.adaptation_strategy}")
    return LLMScoreModel(backbone), tokenizer, matches


def build_optimizer(model: nn.Module, config: RunConfig) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Model has no trainable parameters")
    if config.adaptation_strategy == "linear_probe":
        return torch.optim.SGD(
            parameters,
            lr=config.learning_rate,
            momentum=config.sgd_momentum,
            weight_decay=config.weight_decay,
        )
    return torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        betas=config.adamw_betas,
        eps=config.adamw_epsilon,
        weight_decay=config.weight_decay,
    )


def parameter_report(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    adaptation_strategy: str | None = None,
) -> dict[str, Any]:
    total = 0
    trainable = 0
    groups: dict[str, int] = defaultdict(int)
    totals_by_component: dict[str, int] = defaultdict(int)
    dtypes: dict[str, int] = defaultdict(int)
    trainable_ids: set[int] = set()
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        dtypes[str(parameter.dtype)] += count
        component = "head" if "regression_head" in name else "backbone"
        if "lora_" in name:
            component = "adapter"
        totals_by_component[component] += count
        if parameter.requires_grad:
            trainable += count
            trainable_ids.add(id(parameter))
            if "regression_head" in name:
                groups["head"] += count
            elif "lora_" in name:
                groups["adapter"] += count
            else:
                groups["backbone"] += count
    logical_mxfp4_parameters = 0
    for module in model.modules():
        if isinstance(module, Mxfp4ExpertLora):
            base = module.base_layer
            logical_mxfp4_parameters += int(base.num_experts) * (
                int(base.hidden_size) * 2 * int(base.intermediate_size)
                + int(base.intermediate_size) * int(base.hidden_size)
            )
    if logical_mxfp4_parameters:
        total += logical_mxfp4_parameters
        dtypes["mxfp4_logical"] += logical_mxfp4_parameters
    named_by_id = {id(parameter): (name, parameter) for name, parameter in model.named_parameters()}
    optimizer_occurrences: list[nn.Parameter] = []
    if optimizer is not None:
        for group in optimizer.param_groups:
            optimizer_occurrences.extend(group["params"])
    optimizer_ids = {id(parameter) for parameter in optimizer_occurrences}
    duplicate_optimizer_ids = sorted(
        parameter_id
        for parameter_id in optimizer_ids
        if sum(id(parameter) == parameter_id for parameter in optimizer_occurrences) != 1
    )
    frozen_optimizer_ids = {
        id(parameter) for parameter in optimizer_occurrences if not parameter.requires_grad
    }
    missing_trainable_ids = trainable_ids - optimizer_ids
    unexpected_optimizer_ids = optimizer_ids - set(named_by_id)
    logical_backbone = totals_by_component.get("backbone", 0)
    trainable_backbone = groups.get("backbone", 0)
    optimizer_matches = (
        optimizer is None
        or (
            optimizer_ids == trainable_ids
            and not duplicate_optimizer_ids
            and not frozen_optimizer_ids
            and not unexpected_optimizer_ids
        )
    )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percentage": 0.0 if total == 0 else 100.0 * trainable / total,
        "trainable_by_component": dict(sorted(groups.items())),
        "total_by_component": dict(sorted(totals_by_component.items())),
        "parameters_by_dtype": dict(sorted(dtypes.items())),
        "optimizer_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if id(parameter) in optimizer_ids
        ),
        "optimizer_matches_trainable": optimizer_matches,
        "optimizer_parameter_occurrences": len(optimizer_occurrences),
        "optimizer_unique_parameter_tensors": len(optimizer_ids),
        "optimizer_duplicate_parameter_tensors": len(duplicate_optimizer_ids),
        "optimizer_frozen_parameter_count": sum(
            named_by_id[parameter_id][1].numel()
            for parameter_id in frozen_optimizer_ids
            if parameter_id in named_by_id
        ),
        "optimizer_missing_trainable_parameter_count": sum(
            named_by_id[parameter_id][1].numel() for parameter_id in missing_trainable_ids
        ),
        "optimizer_unexpected_parameter_tensors": len(unexpected_optimizer_ids),
        "logical_backbone_parameters": logical_backbone,
        "trainable_backbone_parameters": trainable_backbone,
        "full_weight_trainable_backbone_matches_logical": (
            adaptation_strategy != "full_weight" or logical_backbone == trainable_backbone
        ),
        "logical_mxfp4_parameters": logical_mxfp4_parameters,
    }


def save_score_model(model: nn.Module, path: str | Path, config: RunConfig) -> None:
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=False)
    if isinstance(model, LinearScoreModel):
        torch.save(
            {
                "class_name": "LinearScoreModel",
                "input_dim": model.regression_head.in_features,
                "state_dict": model.state_dict(),
            },
            destination / "model.pt",
        )
    elif isinstance(model, LLMScoreModel):
        native_state = native_expert_lora_state_dict(model.backbone)
        save_kwargs: dict[str, Any] = {}
        if native_state:
            markers = ("lora_gate_A", "lora_gate_B", "lora_down_A", "lora_down_B")
            save_kwargs["state_dict"] = {
                name: value
                for name, value in model.backbone.state_dict().items()
                if not any(marker in name for marker in markers)
            }
        model.backbone.save_pretrained(
            destination / "backbone", safe_serialization=True, **save_kwargs
        )
        if native_state:
            torch.save(native_state, destination / "native_expert_lora.pt")
        torch.save(model.regression_head.state_dict(), destination / "regression_head.pt")
        (destination / "model.json").write_text(
            json.dumps(
                {
                    "class_name": "LLMScoreModel",
                    "adaptation_strategy": config.adaptation_strategy,
                    "model_id": config.model_id,
                    "model_revision": config.model_revision,
                },
                indent=2,
            )
            + "\n"
        )
    else:
        raise TypeError(f"Unsupported score model: {type(model).__name__}")


def load_score_model(
    checkpoint: str | Path,
    config: RunConfig,
    *,
    device: torch.device,
) -> tuple[nn.Module, Any | None]:
    source = Path(checkpoint)
    if config.adaptation_strategy == "linear_probe":
        payload = torch.load(source / "model.pt", map_location="cpu", weights_only=False)
        model = LinearScoreModel(int(payload["input_dim"]))
        state_dict = payload["state_dict"]
        # Standardized probes save nonempty preprocessing buffers, while a new
        # probe intentionally initializes them empty. Restore their shapes
        # before strict state-dict loading.
        for name in ("input_mean", "input_std"):
            saved = state_dict.get(name)
            if saved is not None:
                setattr(model, name, torch.empty_like(saved, dtype=torch.float32))
        model.load_state_dict(state_dict)
        return model.to(device), None
    from transformers import AutoModel, AutoTokenizer

    tokenizer_root = source.parents[2] / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_root if tokenizer_root.exists() else config.model_id,
        revision=None if tokenizer_root.exists() else config.model_revision,
        trust_remote_code=True,
    )
    if config.adaptation_strategy == "lora":
        from peft import PeftModel

        base = AutoModel.from_pretrained(
            config.model_id,
            revision=config.model_revision,
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
        backbone = PeftModel.from_pretrained(
            base,
            source / "backbone",
            autocast_adapter_dtype=False,
        )
        native_path = source / "native_expert_lora.pt"
        if config.lora_target_parameters and native_path.exists():
            installed = _install_native_mxfp4_expert_lora(backbone, config)
            if not installed:
                raise ValueError("Checkpoint requires native MXFP4 expert LoRA modules")
            load_native_expert_lora_state_dict(
                backbone,
                torch.load(native_path, map_location="cpu", weights_only=True),
            )
    else:
        metadata = json.loads((source / "model.json").read_text())
        if metadata.get("adaptation_strategy") != "full_weight":
            raise ValueError("Selected checkpoint is not a full-weight checkpoint")
        if not (source / "backbone").is_dir():
            raise FileNotFoundError(f"Full-weight checkpoint has no backbone: {source}")
        backbone = AutoModel.from_pretrained(source / "backbone", dtype=torch.bfloat16)
    if hasattr(backbone.config, "use_cache"):
        backbone.config.use_cache = False
    model = LLMScoreModel(backbone)
    model.regression_head.load_state_dict(
        torch.load(source / "regression_head.pt", map_location="cpu", weights_only=True)
    )
    return model.to(device), tokenizer
