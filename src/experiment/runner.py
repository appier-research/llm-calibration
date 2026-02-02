"""Experiment runner with git tracking and resumption."""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig

from src.data.base import BaseDataset, DataExample
from src.estimators.base import BaseEstimator
from src.models.base import BaseModel, UsageStats
from src.prompts.prediction import PredictionPromptBuilder
from src.verifiers.base import BaseVerifier
from .checkpointing import Checkpoint, ExampleResult
from .tracking import ExperimentTracker

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    output_dir: Path
    max_tokens: int = 8192
    temperature: float = 0.0
    top_p: float = 1.0
    stop_sequences: Optional[list[str]] = None
    checkpoint_enabled: bool = True
    use_async: bool = True
    max_concurrent: int = 100


@dataclass
class ExperimentResult:
    """Complete result of an experiment run."""
    results: list[ExampleResult]
    metrics: dict[str, float]
    usage: UsageStats
    num_completed: int
    num_total: int
    output_dir: Path
    resumed_from: int = 0  # Number of examples loaded from checkpoint


class ExperimentRunner:
    """
    Runs calibration experiments with git tracking and resumption.
    
    Workflow:
    1. Initialize with model, dataset, verifier, estimator
    2. Load checkpoint if resuming
    3. For each example:
       - Build prompt
       - Generate response (with caching)
       - Verify correctness
       - Estimate confidence
       - Save checkpoint
    4. Compute metrics
    5. Save results, metrics, usage
    """

    def __init__(
        self,
        model: BaseModel,
        dataset: BaseDataset,
        verifier: BaseVerifier,
        estimator: Optional[BaseEstimator],
        prompt_builder: PredictionPromptBuilder,
        config: ExperimentConfig,
    ):
        self.model = model
        self.dataset = dataset
        self.verifier = verifier
        self.estimator = estimator
        self.prompt_builder = prompt_builder
        self.config = config
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking
        self.tracker = ExperimentTracker(self.output_dir)
        
        # Initialize checkpoint
        checkpoint_path = self.output_dir / "checkpoint.jsonl"
        self.checkpoint = Checkpoint(
            checkpoint_path,
            enabled=config.checkpoint_enabled,
        )
        
        # Usage tracking
        self._total_usage = UsageStats()

    def run(self, hydra_config: Optional[DictConfig] = None) -> ExperimentResult:
        """
        Run the experiment.
        
        Args:
            hydra_config: Optional Hydra config for logging.
        
        Returns:
            ExperimentResult with all results and metrics.
        """
        # Log experiment metadata
        if hydra_config is not None:
            self.tracker.log_experiment(hydra_config)
        
        # Load checkpoint
        self.checkpoint.load()
        num_resumed = self.checkpoint.num_completed()
        if num_resumed > 0:
            logger.info(f"Resuming from checkpoint: {num_resumed} examples already completed")
        
        # Load dataset
        self.dataset.load()
        examples = list(self.dataset)
        
        # Run experiment
        if self.config.use_async:
            results = asyncio.run(self._run_async(examples))
        else:
            results = self._run_sync(examples)
        
        # Compute metrics
        metrics = self._compute_metrics(results)
        
        # Save outputs
        self._save_outputs(results, metrics)
        
        return ExperimentResult(
            results=results,
            metrics=metrics,
            usage=self._total_usage,
            num_completed=len([r for r in results if r.error is None]),
            num_total=len(examples),
            output_dir=self.output_dir,
            resumed_from=num_resumed,
        )

    def _run_sync(self, examples: list[DataExample]) -> list[ExampleResult]:
        """Run experiment synchronously."""
        results = []
        
        for i, example in enumerate(examples):
            # Skip if already completed
            if self.checkpoint.is_completed(example.id):
                cached = self.checkpoint.get_result(example.id)
                if cached:
                    results.append(cached)
                continue
            
            result = self._process_example(example)
            results.append(result)
            
            # Save checkpoint
            self.checkpoint.save_result(result)
            
            # Log progress
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(examples)} examples")
        
        return results

    async def _run_async(self, examples: list[DataExample]) -> list[ExampleResult]:
        """Run experiment asynchronously with concurrency control."""
        from src.inference.semaphore import AsyncSemaphore
        
        semaphore = AsyncSemaphore(max_concurrent=self.config.max_concurrent)
        results: list[Optional[ExampleResult]] = [None] * len(examples)
        
        async def process_one(idx: int, example: DataExample) -> None:
            # Skip if already completed
            if self.checkpoint.is_completed(example.id):
                cached = self.checkpoint.get_result(example.id)
                results[idx] = cached
                return
            
            async with semaphore.acquire():
                result = await self._process_example_async(example)
            
            results[idx] = result
            self.checkpoint.save_result(result)
            
            # Log progress periodically
            completed = sum(1 for r in results if r is not None)
            if completed % 10 == 0:
                logger.info(f"Processed {completed}/{len(examples)} examples")
        
        tasks = [process_one(i, ex) for i, ex in enumerate(examples)]
        await asyncio.gather(*tasks)
        
        return [r for r in results if r is not None]

    def _process_example(self, example: DataExample) -> ExampleResult:
        """Process a single example synchronously."""
        try:
            # Build chat messages
            messages = self.prompt_builder.build_chat_messages(
                question=example.question,
                dataset_name=self.dataset.name,
            )
            
            # Generate response using chat template
            output = self.model.generate_from_messages(
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                stop=self.config.stop_sequences,
                return_logprobs=self.estimator.requires_logprobs if self.estimator else False,
            )
            
            self._total_usage = self._total_usage + output.usage
            
            # Extract answer
            extracted = self.dataset.extract_answer(output.text)
            
            # Verify correctness
            is_correct = self.verifier.verify(example, extracted)
            
            # Estimate confidence
            confidence = None
            if self.estimator:
                confidence = self.estimator.estimate_v(
                    model=self.model,
                    question=example.question,
                    answer=extracted,
                    messages=messages,
                )
            
            return ExampleResult(
                example_id=example.id,
                question=example.question,
                ground_truth=example.answer,
                prediction=output.text,
                extracted_answer=extracted,
                is_correct=is_correct,
                confidence=confidence,
                raw_response=output.text,
            )
        
        except Exception as e:
            logger.error(f"Error processing example {example.id}: {e}")
            return ExampleResult(
                example_id=example.id,
                question=example.question,
                ground_truth=example.answer,
                error=str(e),
            )

    async def _process_example_async(self, example: DataExample) -> ExampleResult:
        """Process a single example asynchronously."""
        try:
            # Build chat messages
            messages = self.prompt_builder.build_chat_messages(
                question=example.question,
                dataset_name=self.dataset.name,
            )
            
            # Generate response using chat template
            output = await self.model.generate_from_messages_async(
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                stop=self.config.stop_sequences,
                return_logprobs=self.estimator.requires_logprobs if self.estimator else False,
            )
            
            self._total_usage = self._total_usage + output.usage
            
            # Extract answer
            extracted = self.dataset.extract_answer(output.text)
            
            # Verify correctness
            is_correct = self.verifier.verify(example, extracted)
            
            # Estimate confidence
            confidence = None
            if self.estimator:
                confidence = await self.estimator.estimate_v_async(
                    model=self.model,
                    question=example.question,
                    answer=extracted,
                    messages=messages,
                )
            
            return ExampleResult(
                example_id=example.id,
                question=example.question,
                ground_truth=example.answer,
                prediction=output.text,
                extracted_answer=extracted,
                is_correct=is_correct,
                confidence=confidence,
                raw_response=output.text,
            )
        
        except Exception as e:
            logger.error(f"Error processing example {example.id}: {e}")
            return ExampleResult(
                example_id=example.id,
                question=example.question,
                ground_truth=example.answer,
                error=str(e),
            )

    def _compute_metrics(self, results: list[ExampleResult]) -> dict[str, float]:
        """Compute calibration and accuracy metrics."""
        # Filter out errors
        valid = [r for r in results if r.error is None and r.is_correct is not None]
        
        if not valid:
            return {"accuracy": 0.0, "num_valid": 0}
        
        # Accuracy
        correct = sum(1 for r in valid if r.is_correct)
        accuracy = correct / len(valid)
        
        metrics = {
            "accuracy": accuracy,
            "num_correct": correct,
            "num_valid": len(valid),
            "num_errors": len(results) - len(valid),
        }
        
        # Calibration metrics (only if we have confidence scores)
        confidences = [r.confidence for r in valid if r.confidence is not None]
        correctness = [r.is_correct for r in valid if r.confidence is not None]
        
        if confidences:
            metrics["mean_confidence"] = sum(confidences) / len(confidences)
            
            # Expected Calibration Error (ECE)
            ece = self._compute_ece(confidences, correctness)
            metrics["ece"] = ece
            
            # Brier score
            brier = sum(
                (c - int(is_correct)) ** 2 
                for c, is_correct in zip(confidences, correctness)
            ) / len(confidences)
            metrics["brier_score"] = brier
        
        return metrics

    def _compute_ece(
        self,
        confidences: list[float],
        correctness: list[bool],
        n_bins: int = 10,
    ) -> float:
        """Compute Expected Calibration Error."""
        if not confidences:
            return 0.0
        
        bin_boundaries = [i / n_bins for i in range(n_bins + 1)]
        bin_counts = [0] * n_bins
        bin_correct = [0] * n_bins
        bin_confidence = [0.0] * n_bins
        
        for conf, corr in zip(confidences, correctness):
            # Find bin
            bin_idx = min(int(conf * n_bins), n_bins - 1)
            bin_counts[bin_idx] += 1
            bin_correct[bin_idx] += int(corr)
            bin_confidence[bin_idx] += conf
        
        ece = 0.0
        for i in range(n_bins):
            if bin_counts[i] > 0:
                avg_confidence = bin_confidence[i] / bin_counts[i]
                avg_accuracy = bin_correct[i] / bin_counts[i]
                ece += (bin_counts[i] / len(confidences)) * abs(avg_accuracy - avg_confidence)
        
        return ece

    def _save_outputs(
        self,
        results: list[ExampleResult],
        metrics: dict[str, float],
    ) -> None:
        """Save all experiment outputs."""
        # Save results
        self.tracker.log_results([r.to_dict() for r in results])
        
        # Save metrics
        self.tracker.log_metrics(metrics)
        
        # Save usage
        self.tracker.log_usage({
            "prompt_tokens": self._total_usage.prompt_tokens,
            "completion_tokens": self._total_usage.completion_tokens,
            "total_tokens": self._total_usage.total_tokens,
        })
        
        logger.info(f"Results saved to {self.output_dir}")

