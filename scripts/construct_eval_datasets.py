#!/usr/bin/env python
"""
Compute ground truth c*(x) by sampling k responses per question.

Usage:
    uv run python scripts/construct_eval_datasets.py \
        experiment=default \
        experiment.sampling.k=100 \
        experiment.async_.max_concurrent=100 \
        dataset=gsm8k \  # or other datasets
        dataset.split=test \  # or other splits
        model=openai_compat \
        model.model_id="openai/gpt-oss-20b" \
        model.base_url="" \  # the OpenAI-compatible API endpoint (e.g., served by vLLM, SGLang, etc.)
        hydra.job_logging.root.level=INFO \
        inference.max_tokens=32700 \
        inference.temperature=1.0 \
        inference.top_p=1.0 \
        hydra.run.dir=outputs/gsm8k-test__gpt-oss-20b
"""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import hydra
from omegaconf import DictConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SampledResult:
    """Single sampled response for one example."""
    example_id: str
    sampled_id: int  # 0 to k-1
    prompt_tokens: int
    completion_tokens: int
    prompt: list[dict[str, str]]
    response: str
    parsed_response: str
    ground_truth_answer: Any
    correctness: int  # 0 or 1
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GroundTruthResult:
    """Aggregated result per example (compact, no responses)."""
    example_id: str
    total_prompt_tokens: int
    total_completion_tokens: int
    ground_truth_answer: Any
    num_samples: int
    num_correct: int
    expected_accuracy: float  # c*(x) - the ground truth calibration target
    question: str
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for ground truth computation."""
    # NOTE: override cfg with debug config
    from omegaconf import OmegaConf
    cfg = OmegaConf.merge(cfg, cfg.experiment)
    d = OmegaConf.to_container(cfg, resolve=True)

    from collections import defaultdict

    from hydra.utils import instantiate
    
    from src.data.base import BaseDataset, DataExample
    from src.experiment.tracking import ExperimentTracker
    from src.inference.cache import ResponseCache
    from src.inference.semaphore import AsyncSemaphore
    from src.models.base import BaseModel, UsageStats
    from src.prompts.prediction import PredictionPromptBuilder
    from src.verifiers.base import BaseVerifier
    
    # Get output directory from Hydra
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # File paths
    sampled_path = output_dir / "sampled.jsonl"
    prev_sampled_path = output_dir / "prev_sampled.jsonl"
    results_path = output_dir / "ground_truth.jsonl"
    
    # Load existing responses FIRST (before dataset loading)
    # Rename sampled.jsonl -> prev_sampled.jsonl to preserve responses
    existing_responses: dict[tuple[str, int], dict] = {}
    existing_example_ids: set[str] = set()
    if sampled_path.exists():
        sampled_path.rename(prev_sampled_path)
        logger.info(f"Renamed {sampled_path} -> {prev_sampled_path}")
    
    if prev_sampled_path.exists():
        with open(prev_sampled_path) as f:
            for line in f:
                data = json.loads(line)
                key = (data["example_id"], data["sampled_id"])
                existing_responses[key] = data
                existing_example_ids.add(data["example_id"])
        logger.info(f"Loaded {len(existing_responses)} existing responses from {len(existing_example_ids)} examples (will re-verify)")
    
    # Log experiment
    tracker = ExperimentTracker(output_dir)
    tracker.log_experiment(cfg)
    
    logger.info("Initializing components...")
    
    # Instantiate components from config
    model: BaseModel = instantiate(cfg.model)
    verifier: BaseVerifier = instantiate(cfg.verifier)
    
    # Build prompt builder from config
    prompt_builder = PredictionPromptBuilder.from_config(cfg.prompts.prediction)
    
    # Setup cache
    cache = ResponseCache(
        cache_path=cfg.cache.path,
        enabled=cfg.cache.enabled,
    )
    
    # Setup semaphore for async operations
    semaphore = AsyncSemaphore(max_concurrent=cfg.async_.max_concurrent)
    
    # Load dataset - we need to handle existing examples specially
    max_examples = cfg.dataset.get("max_examples", None)
    
    # Temporarily disable max_examples to load full dataset
    dataset: BaseDataset = instantiate(cfg.dataset, max_examples=None)
    dataset.load()
    all_examples = list(dataset)
    all_examples_by_id = {ex.id: ex for ex in all_examples}
    logger.info(f"Loaded {len(all_examples)} total examples from {dataset.name}")
    
    # Separate existing and new examples
    existing_examples = [all_examples_by_id[eid] for eid in existing_example_ids if eid in all_examples_by_id]
    new_candidate_examples = [ex for ex in all_examples if ex.id not in existing_example_ids]
    
    # Calculate how many new examples we need
    num_existing = len(existing_examples)
    num_new_needed = max(0, (max_examples or len(all_examples)) - num_existing)
    
    # Sample new examples if needed
    if num_new_needed > 0 and new_candidate_examples:
        import random
        if cfg.dataset.get("seed") is not None:
            random.seed(cfg.dataset.seed)
        new_examples = random.sample(new_candidate_examples, min(num_new_needed, len(new_candidate_examples)))
    else:
        new_examples = []
    
    # Combine: existing examples + newly sampled examples
    examples = existing_examples + new_examples
    examples_by_id = {ex.id: ex for ex in examples}
    logger.info(f"Using {len(examples)} examples: {num_existing} existing + {len(new_examples)} new")
    
    # Sampling parameters
    k = cfg.sampling.k
    temperature = cfg.inference.temperature
    max_tokens = cfg.inference.max_tokens
    top_p = cfg.inference.top_p
    
    logger.info(f"Computing ground truth with k={k} samples, temperature={temperature}, top_p={top_p}, max_tokens={max_tokens}")
    
    # Async lock for writing to sampled.jsonl
    write_lock = asyncio.Lock()
    total_usage = UsageStats()
    usage_lock = asyncio.Lock()
    sampled_file = None  # Will hold open file handle
    
    async def sample_one(
        example: DataExample,
        sampled_id: int,
        messages: list[dict[str, str]],
    ) -> SampledResult:
        """Process a single (example, sampled_id) pair."""
        nonlocal total_usage
        
        key = (example.id, sampled_id)
        existing = existing_responses.get(key)
        
        if existing is not None:
            # Use existing response, just re-verify
            response_text = existing["response"]
            prompt_tokens = existing["prompt_tokens"]
            completion_tokens = existing["completion_tokens"]
        else:
            # Generate new response
            cache_key = json.dumps(messages, sort_keys=True)
            sample_params = {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stop": None,
                "return_logprobs": False,
                "sample_idx": sampled_id,
            }
            
            # Check cache first
            cached = cache.get(cache_key, model.model_id, sample_params) if cache else None
            
            if cached is not None:
                output = cached
            else:
                # Generate with semaphore rate limiting
                async with semaphore.acquire():
                    output = await model.generate_from_messages_async(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                # Cache the result
                if cache:
                    cache.put(cache_key, model.model_id, sample_params, output)
            
            # Track usage (only for new generations)
            async with usage_lock:
                total_usage = total_usage + output.usage
            
            response_text = output.text
            prompt_tokens = output.usage.prompt_tokens
            completion_tokens = output.usage.completion_tokens
        
        # Always re-extract and re-verify
        extracted = dataset.extract_answer(response_text)
        dataset_result = await dataset.verify_answer(extracted, example)
        is_correct = dataset_result if dataset_result is not None else verifier.verify(example, extracted)
        
        result = SampledResult(
            example_id=example.id,
            sampled_id=sampled_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt=messages,
            response=response_text,
            parsed_response=extracted,
            ground_truth_answer=example.answer,
            correctness=1 if is_correct else 0,
        )
        
        # Write to sampled.jsonl atomically
        async with write_lock:
            sampled_file.write(json.dumps(result.to_dict()) + "\n")
            sampled_file.flush()  # Ensure data is written, especially important on NFS
        
        return result
    
    async def run_all_samples():
        """Run sampling with sliding window - only max_concurrent tasks alive at once."""
        from tqdm import tqdm
        
        max_concurrent = cfg.async_.max_concurrent
        
        # Generator - lazily yields coroutines without creating all upfront
        # Interleaved iteration: (ex1,s0), (ex2,s0), ..., (ex1,s1), (ex2,s1), ...
        # This distributes slow examples across time to avoid head-of-line blocking
        def task_generator():
            # Pre-build messages for all examples to avoid rebuilding k times
            example_messages = [
                (example, prompt_builder.build_chat_messages(
                    question=example.question,
                    dataset_name=dataset.name,
                ))
                for example in examples
            ]
            # Iterate sampled_id first, then examples
            for sampled_id in range(k):
                for example, messages in example_messages:
                    yield sample_one(example, sampled_id, messages)
        
        # Count total tasks for progress bar
        total_tasks = len(examples) * k
        num_existing = len(existing_responses)
        num_new = total_tasks - num_existing
        logger.info(f"Processing {total_tasks} samples ({num_existing} existing responses, {num_new} new generations)")
        
        logger.info(f"Running {total_tasks} sampling tasks with sliding window (max_concurrent={max_concurrent})...")
        
        # Sliding window: only keep max_concurrent tasks alive at a time
        pending: set[asyncio.Task] = set()
        task_iter = iter(task_generator())
        completed = 0
        
        with tqdm(total=total_tasks, desc="Sampling & Verifying", dynamic_ncols=True) as pbar:
            # Fill initial window
            for coro in task_iter:
                pending.add(asyncio.create_task(coro))
                if len(pending) >= max_concurrent:
                    break
            
            # Process with sliding window
            while pending:
                # Wait for at least one to complete
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                completed += len(done)
                pbar.update(len(done))
                
                # Refill window up to max_concurrent
                while len(pending) < max_concurrent:
                    try:
                        coro = next(task_iter)
                        pending.add(asyncio.create_task(coro))
                    except StopIteration:
                        break
    
    # Open file for writing (fresh file, we renamed the old one)
    sampled_file = open(sampled_path, "w")
    
    try:
        # Run the async sampling
        asyncio.run(run_all_samples())
        
        # # Success - delete prev_sampled.jsonl
        # if prev_sampled_path.exists():
        #     prev_sampled_path.unlink()
        #     logger.info(f"Deleted {prev_sampled_path}")
    finally:
        # Always close the file, even if an error occurs
        sampled_file.close()
    
    # === Aggregation Phase ===
    # Read sampled.jsonl and aggregate into GroundTruthResult
    logger.info("Aggregating results...")
    
    samples_by_example: dict[str, list[SampledResult]] = defaultdict(list)
    with open(sampled_path) as f:
        for line in f:
            data = json.loads(line)
            samples_by_example[data["example_id"]].append(
                SampledResult(**data)
            )
    
    # Build GroundTruthResult for each example
    results: list[GroundTruthResult] = []
    for example_id, samples in samples_by_example.items():
        num_correct = sum(s.correctness for s in samples)
        num_samples = len(samples)
        gt_result = GroundTruthResult(
            example_id=example_id,
            total_prompt_tokens=sum(s.prompt_tokens for s in samples),
            total_completion_tokens=sum(s.completion_tokens for s in samples),
            ground_truth_answer=examples_by_id[example_id].answer,
            num_samples=num_samples,
            num_correct=num_correct,
            expected_accuracy=num_correct / num_samples if num_samples > 0 else 0.0,
            question=examples_by_id[example_id].question,
        )
        results.append(gt_result)
    
    # Write ground_truth.jsonl
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if results_path.exists():
        results_path.rename(f"{results_path.with_suffix('')}_{timestamp}.jsonl")
    with open(results_path, "w") as f:
        for gt_result in results:
            f.write(json.dumps(gt_result.to_dict()) + "\n")
    
    # Compute summary statistics
    accuracies = [r.expected_accuracy for r in results]
    summary = {
        "num_examples": len(results),
        "k_samples": k,
        "mean_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0,
        "std_accuracy": (
            (sum((a - sum(accuracies)/len(accuracies))**2 for a in accuracies) / len(accuracies))**0.5
            if len(accuracies) > 1 else 0
        ),
        "min_accuracy": min(accuracies) if accuracies else 0,
        "max_accuracy": max(accuracies) if accuracies else 0,
    }
    
    # Save summary
    summary_path = output_dir / "ground_truth_summary.json"
    if summary_path.exists():
        summary_path.rename(f"{summary_path.with_suffix('')}_{timestamp}.json")
    summary_path.write_text(json.dumps(summary, indent=4))
    
    # Save usage
    tracker.log_usage({
        "prompt_tokens": total_usage.prompt_tokens,
        "completion_tokens": total_usage.completion_tokens,
        "total_tokens": total_usage.total_tokens,
    })
    
    logger.info(f"Ground truth computation complete!")
    logger.info(f"Sampled results saved to {sampled_path}")
    logger.info(f"Aggregated results saved to {results_path}")
    logger.info(f"Mean accuracy: {summary['mean_accuracy']:.4f}")
    logger.info(f"Total tokens: {total_usage.total_tokens:,}")


if __name__ == "__main__":
    main()

