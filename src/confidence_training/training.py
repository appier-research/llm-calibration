"""Epoch-based trainer with dual checkpointing and exact rung continuation."""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.metrics.custom import c_star_metrics

from .artifacts import RunArtifacts
from .config import RunConfig
from .control import TrainingControl
from .losses import confidence_loss
from .models import (
    LLMScoreModel,
    load_native_expert_lora_state_dict,
    native_expert_lora_state_dict,
    save_score_model,
    scores_to_probabilities,
)


@dataclass(slots=True)
class ValidationOutput:
    example_ids: list[str]
    scores: torch.Tensor
    probabilities: torch.Tensor
    targets: torch.Tensor
    metrics: dict[str, float]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


@torch.inference_mode()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> ValidationOutput:
    model.eval()
    identifiers: list[str] = []
    all_scores: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    for batch in loader:
        inputs = _to_device(batch["inputs"], device)
        targets = batch["targets"].to(device).float()
        scores = model.forward_scores(**inputs).float()
        identifiers.extend(batch["example_ids"])
        all_scores.append(scores.cpu())
        all_targets.append(targets.cpu())
    scores_tensor = torch.cat(all_scores)
    targets_tensor = torch.cat(all_targets)
    probabilities = scores_to_probabilities(scores_tensor)
    result = c_star_metrics(probabilities.tolist(), targets_tensor.tolist())
    metrics = {
        "mae": result.mae,
        "mse": result.mse,
        "rmse": result.rmse,
        "pearson_r": result.pearson_r,
        "spearman_r": result.spearman_r,
        "ece": result.ece,
    }
    return ValidationOutput(identifiers, scores_tensor, probabilities, targets_tensor, metrics)


def _save_resume_state(
    path: Path,
    *,
    epoch: int,
    optimizer_steps: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: RunConfig,
) -> None:
    model_state: dict[str, Any]
    if config.adaptation_strategy == "lora":
        if not isinstance(model, LLMScoreModel):
            raise TypeError("LoRA resume requires an LLMScoreModel")
        from peft import get_peft_model_state_dict

        adapter = {
            name: value.detach().cpu().clone()
            for name, value in get_peft_model_state_dict(model.backbone).items()
            if not any(
                marker in name
                for marker in (
                    "lora_gate_A",
                    "lora_gate_B",
                    "lora_down_A",
                    "lora_down_B",
                )
            )
        }
        head = {
            name: value.detach().cpu().clone()
            for name, value in model.regression_head.state_dict().items()
        }
        model_state = {
            "kind": "lora_adapter_and_head",
            "adapter": adapter,
            "native_expert_lora": native_expert_lora_state_dict(model.backbone),
            "head": head,
        }
    else:
        model_state = {"kind": "full_state_dict", "state_dict": model.state_dict()}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "resume_state_version": 3,
            "epoch": epoch,
            "optimizer_steps": optimizer_steps,
            "scientific_config_hash": config.scientific_hash(),
            "model": model_state,
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        temporary,
    )
    os.replace(temporary, path)


def _load_resume_state(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: RunConfig,
) -> tuple[int, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["scientific_config_hash"] != config.scientific_hash():
        raise ValueError("Resume state scientific configuration does not match")
    if payload.get("resume_state_version", 2) == 2:
        model.load_state_dict(payload["model"])
    elif payload["model"]["kind"] == "lora_adapter_and_head":
        if not isinstance(model, LLMScoreModel):
            raise TypeError("LoRA resume requires an LLMScoreModel")
        from peft import set_peft_model_state_dict

        set_peft_model_state_dict(model.backbone, payload["model"]["adapter"])
        load_native_expert_lora_state_dict(
            model.backbone, payload["model"].get("native_expert_lora", {})
        )
        model.regression_head.load_state_dict(payload["model"]["head"])
    else:
        model.load_state_dict(payload["model"]["state_dict"])
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng"])
    np.random.set_state(payload["numpy_rng"])
    random.setstate(payload["python_rng"])
    if payload.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return int(payload["epoch"]), int(payload.get("optimizer_steps", 0))


def train_run(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    train_collate: Callable,
    validation_collate: Callable,
    config: RunConfig,
    artifacts: RunArtifacts,
    log_callback: Callable[[dict[str, Any]], None] | None = None,
    control: TrainingControl | None = None,
) -> dict[str, Any]:
    run_started = time.perf_counter()
    control = control or TrainingControl()
    target_epoch = control.target_epoch(config)
    device = torch.device(
        config.device
        if torch.cuda.is_available() or not config.device.startswith("cuda")
        else "cpu"
    )
    model.to(device)
    start_epoch = 0
    optimizer_steps = 0
    resume_path = artifacts.root / "last.pt"
    if config.resume:
        if not resume_path.exists():
            raise FileNotFoundError(f"Missing resume state: {resume_path}")
        start_epoch, optimizer_steps = _load_resume_state(
            resume_path, model=model, optimizer=optimizer, config=config
        )
    if target_epoch <= start_epoch:
        raise ValueError(
            f"stop_after_epoch={target_epoch} must exceed resumed epoch={start_epoch}"
        )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers,
        collate_fn=validation_collate,
    )

    if start_epoch == 0:
        initial = validate(model, validation_loader, device)
        row = {
            "kind": "validation",
            "epoch": 0,
            "optimizer_steps": 0,
            "rung": control.rung,
            **initial.metrics,
        }
        artifacts.append_history(row)
        if log_callback:
            log_callback(row)

    keep_resume_state = config.save_resume_state or target_epoch < config.epochs
    for epoch in range(start_epoch + 1, target_epoch + 1):
        generator = torch.Generator().manual_seed(config.seed + epoch)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            generator=generator,
            drop_last=False,
            num_workers=config.num_workers,
            collate_fn=train_collate,
        )
        model.train()
        loss_sum = 0.0
        count = 0
        gradient_norms: list[float] = []
        started = time.perf_counter()
        interval_started = started
        interval_loss_sum = 0.0
        interval_count = 0
        interval_gradient_norms: list[float] = []
        for epoch_step, batch in enumerate(train_loader, start=1):
            inputs = _to_device(batch["inputs"], device)
            targets = batch["targets"].to(device).float()
            optimizer.zero_grad(set_to_none=True)
            scores = model.forward_scores(**inputs).float()
            loss = confidence_loss(
                scores,
                targets,
                config.loss,
                ranknet_temperature=config.ranknet_temperature,
                ranknet_tie_loss_weight=config.ranknet_tie_loss_weight,
            )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                config.max_grad_norm,
            )
            optimizer.step()
            optimizer_steps += 1
            detached_loss = float(loss.detach())
            loss_sum += detached_loss * len(targets)
            count += len(targets)
            detached_gradient_norm = float(gradient_norm.detach())
            gradient_norms.append(detached_gradient_norm)
            interval_loss_sum += detached_loss * len(targets)
            interval_count += len(targets)
            interval_gradient_norms.append(detached_gradient_norm)
            should_log_step = config.log_every_steps is not None and (
                epoch_step % config.log_every_steps == 0
                or epoch_step == len(train_loader)
            )
            if should_log_step:
                step_row = {
                    "kind": "train_step",
                    "epoch": epoch,
                    "epoch_step": epoch_step,
                    "optimizer_steps": optimizer_steps,
                    "examples_seen_in_epoch": count,
                    "interval_examples": interval_count,
                    "rung": control.rung,
                    "interval_train_loss": interval_loss_sum / interval_count,
                    "interval_gradient_norm_mean": float(
                        np.mean(interval_gradient_norms)
                    ),
                    "interval_seconds": time.perf_counter() - interval_started,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
                artifacts.append_history(step_row)
                if log_callback:
                    log_callback(step_row)
                interval_started = time.perf_counter()
                interval_loss_sum = 0.0
                interval_count = 0
                interval_gradient_norms = []
        validation = validate(model, validation_loader, device)
        epoch_row = {
            "kind": "epoch",
            "epoch": epoch,
            "optimizer_steps": optimizer_steps,
            "rung": control.rung,
            "train_loss": loss_sum / count,
            "gradient_norm_mean": float(np.mean(gradient_norms)),
            "elapsed_seconds": time.perf_counter() - started,
            **{f"validation_{key}": value for key, value in validation.metrics.items()},
        }
        artifacts.append_history(epoch_row)
        if log_callback:
            log_callback(epoch_row)
        improved = artifacts.improvements(epoch, validation.metrics)
        artifacts.save_improved_checkpoint(
            epoch=epoch,
            metrics=validation.metrics,
            example_ids=validation.example_ids,
            scores=validation.scores,
            probabilities=validation.probabilities,
            targets=validation.targets,
            improved=improved,
            model_saver=lambda path: save_score_model(model, path, config),
        )
        if keep_resume_state:
            _save_resume_state(
                resume_path,
                epoch=epoch,
                optimizer_steps=optimizer_steps,
                model=model,
                optimizer=optimizer,
                config=config,
            )

    runtime_summary = {
        "wall_seconds": time.perf_counter() - run_started,
        "optimizer_steps": optimizer_steps,
        "peak_vram_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "peak_vram_reserved_bytes": (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
        ),
    }
    artifacts.update_runtime_summary(runtime_summary)
    if target_epoch < config.epochs:
        artifacts.mark_paused(trained_through_epoch=target_epoch, rung=control.rung)
    else:
        artifacts.mark_complete(trained_through_epoch=target_epoch)
    return json.loads(artifacts.manifest_path.read_text())
