#!/usr/bin/env python
"""
Few-shot verbalized confidence: prepend k training demos with \\boxed{expected_accuracy}, then elicit confidence on test questions.

Usage:
    uv run python scripts/few_shot_verbalized_confidence.py \
        --model "openai/gpt-oss-20b" \
        --base_url "<your_base_url>" \
        --temperature 1.0 \
        --top_p 1.0 \
        --max_concurrent 500 \
        --ground_truth_jsonl "outputs/math-500__gpt-oss-20b/ground_truth.jsonl" \
        --train_ground_truth_jsonl "outputs/gsm8k-train__gpt-oss-20b/ground_truth.jsonl" \
        --few_shot_k 5 \
        --few_shot_seed 42 \
        --few_shot_sampling_strategy stratified \
        --output_dir "estimator_results/few_shot_verbalized_confidence/math-500__gpt-oss-20b" \
        --max_completion_tokens 5000

Train/test ground_truth.jsonl fields:
    {example_id, question, expected_accuracy, num_samples, num_correct}

Output files:
    * few_shot_meta.json: k, seed, train path, chosen demo example_ids
    * confidence_predictions.jsonl, evaluation_metrics.json, reliability_diagram.png
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from openai import AsyncOpenAI
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm


@dataclass
class VerbalizedConfidencePrediction:
    example_id: str
    confidence: float
    expected_accuracy: float
    num_samples: int
    num_correct: int
    prompt_tokens: int
    completion_tokens: int
    prompt: list[dict[str, str]]
    response: str
    parsed_response: str
    parse_success: bool


def strip_all_lines(text: str) -> str:
    return "\n".join(line.strip() for line in text.split("\n"))


def test_question_prompt(question: str) -> str:
    prompt = f"""
    Question: {question}

    How likely are you to answer the question correctly? You may refer to the following probabilities P:
    - 0.0-0.1: "Almost no chance"
    - 0.1-0.2: "Highly unlikely"
    - 0.2-0.3: "Chances are slight"
    - 0.3-0.4: "Unlikely"
    - 0.4-0.5: "Less than even"
    - 0.5-0.6: "Better than even"
    - 0.6-0.7: "Likely"
    - 0.7-0.8: "Very good chance"
    - 0.8-0.9: "Highly likely"
    - 0.9-1.0: "Almost certain"
    Reason about your uncertainty and confidence, and then provide a probability P between 0.0 and 1.0 in the format of \\boxed{{P}}.
    """.strip()
    return strip_all_lines(prompt)


def format_boxed_accuracy(expected_accuracy: float) -> str:
    p = float(expected_accuracy)
    return f"\\boxed{{{p:.2f}}}"


def get_demo_block(question: str, expected_accuracy: float) -> str:
    """Same rubric as test_question_prompt, but ends with the given \\boxed{expected_accuracy} (no model generation)."""
    boxed = format_boxed_accuracy(expected_accuracy)
    block = f"""
    Question: {question}

    How likely are you to answer the question correctly? You may refer to the following probabilities P:
    - 0.0-0.1: "Almost no chance"
    - 0.1-0.2: "Highly unlikely"
    - 0.2-0.3: "Chances are slight"
    - 0.3-0.4: "Unlikely"
    - 0.4-0.5: "Less than even"
    - 0.5-0.6: "Better than even"
    - 0.6-0.7: "Likely"
    - 0.7-0.8: "Very good chance"
    - 0.8-0.9: "Highly likely"
    - 0.9-1.0: "Almost certain"
    {boxed}
    """.strip()
    return strip_all_lines(block)


def build_few_shot_user_prompt(demo_examples: list[dict], test_question: str) -> str:
    parts: list[str] = [
        "Below are example questions with reference calibrated probabilities",
        "",
    ]
    for i, ex in enumerate(demo_examples, 1):
        parts.append(f"--- Example {i} ---")
        parts.append(get_demo_block(ex["question"], ex["expected_accuracy"]))
        parts.append("")
    parts.append("--- Your task ---")
    parts.append(test_question_prompt(test_question))
    return strip_all_lines("\n".join(parts))


def sample_few_shot_demos(train_rows: list[dict], k: int, seed: int, strategy: str = "stratified") -> list[dict]:
    """
    Sample k demonstrations from training pool.
    
    Args:
        train_rows: Training examples with 'expected_accuracy' field
        k: Number of demos to sample
        seed: Random seed for reproducibility
        strategy: 'stratified' (diversity across confidence bins) or 'random' (uniform random)
    
    Returns:
        List of k sampled examples
    """
    if len(train_rows) < k:
        raise ValueError(
            f"Train pool has {len(train_rows)} examples but few_shot_k={k}. "
            "Reduce k or use a larger train ground_truth.jsonl."
        )
    
    rng = random.Random(seed)
    
    if strategy == "random":
        return rng.sample(train_rows, k)
    
    elif strategy == "stratified":
        # Stratified sampling: ensure diversity across confidence bins
        # Divide [0,1] into k bins and sample one example from each bin
        bins = [(i / k, (i + 1) / k) for i in range(k)]
        
        # Group examples by bin
        binned_examples: list[list[dict]] = [[] for _ in range(k)]
        for ex in train_rows:
            acc = ex["expected_accuracy"]
            # Find which bin this example belongs to
            for bin_idx, (low, high) in enumerate(bins):
                if low <= acc < high or (bin_idx == k - 1 and acc == 1.0):  # Last bin includes 1.0
                    binned_examples[bin_idx].append(ex)
                    break
        
        # Sample one from each bin (or skip if bin is empty)
        demos = []
        empty_bins = []
        for bin_idx, bin_examples in enumerate(binned_examples):
            if bin_examples:
                demos.append(rng.choice(bin_examples))
            else:
                empty_bins.append(bin_idx)
        
        # If we have empty bins, fill remaining slots with random samples from available examples
        if len(demos) < k:
            remaining_needed = k - len(demos)
            # Pool of examples not yet selected
            selected_ids = {ex["example_id"] for ex in demos}
            remaining_pool = [ex for ex in train_rows if ex["example_id"] not in selected_ids]
            
            if len(remaining_pool) >= remaining_needed:
                additional = rng.sample(remaining_pool, remaining_needed)
                demos.extend(additional)
                print(f"Warning: {len(empty_bins)} bins empty (bins {empty_bins}), filled with {remaining_needed} random examples")
            else:
                raise ValueError(
                    f"Stratified sampling failed: need {k} demos but only {len(demos)} bins have examples. "
                    f"Empty bins: {empty_bins}. Consider using --few_shot_sampling_strategy random"
                )
        
        return demos
    
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}. Use 'stratified' or 'random'.")


def parse_confidence(response: str) -> float | None:
    """Extract probability from \\boxed{P} format."""
    if "<think>" in response:
        response = response.split("</think>")[1]
    matches = re.findall(r"\\boxed\{([0-9.]+)\}", response)
    if matches:
        last_match = matches[-1]
        try:
            val = float(last_match)
            if not 0 <= val <= 1:
                print(f"Warning: confidence value {val} is out of range [0, 1]")
                return None
            return val
        except ValueError:
            return None
    return None


def compute_ece(confidences: np.ndarray, expected_accuracies: np.ndarray, n_bins: int = 10) -> float:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_conf = confidences[mask].mean()
            bin_acc = expected_accuracies[mask].mean()
            ece += mask.sum() * abs(bin_conf - bin_acc)
    return ece / len(confidences)


def compute_metrics(predictions: list[VerbalizedConfidencePrediction]) -> dict:
    confs = np.array([p.confidence for p in predictions])
    accs = np.array([p.expected_accuracy for p in predictions])

    mse = float(np.mean((confs - accs) ** 2))
    ece = float(compute_ece(confs, accs, n_bins=10))
    pearson_r_val, _ = pearsonr(confs, accs)
    spearman_r_val, _ = spearmanr(confs, accs)

    return {
        "mse": mse,
        "ece": ece,
        "pearson_r": float(pearson_r_val),
        "spearman_r": float(spearman_r_val),
    }


def plot_reliability_diagram(
    predictions: list[VerbalizedConfidencePrediction],
    output_path: Path,
    n_bins: int = 10,
) -> None:
    confs = np.array([p.confidence for p in predictions])
    accs = np.array([p.expected_accuracy for p in predictions])

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        mask = (confs >= bin_boundaries[i]) & (confs < bin_boundaries[i + 1])
        count = mask.sum()
        bin_counts.append(count)
        if count > 0:
            bin_accs.append(accs[mask].mean())
            bin_confs.append(confs[mask].mean())
        else:
            bin_accs.append(np.nan)
            bin_confs.append(np.nan)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    bar_width = 0.08
    ax.bar(bin_centers, bin_accs, width=bar_width, alpha=0.7, label="Actual accuracy", color="steelblue")

    valid_mask = ~np.isnan(bin_accs)
    ax.scatter(
        np.array(bin_confs)[valid_mask],
        np.array(bin_accs)[valid_mask],
        color="red",
        s=50,
        zorder=5,
        label="Bin mean",
    )

    ax.set_xlabel("Predicted Confidence", fontsize=12)
    ax.set_ylabel("Expected Accuracy", fontsize=12)
    ax.set_title("Reliability Diagram", fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    for i, (center, count) in enumerate(zip(bin_centers, bin_counts)):
        if count > 0:
            ax.text(center, 0.02, f"n={count}", ha="center", fontsize=8, alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


async def predict_confidence(
    client: AsyncOpenAI,
    example: dict,
    demo_examples: list[dict],
    model: str,
    timeout: int,
    max_retries: int = 3,
    max_completion_tokens: int = 8192,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> VerbalizedConfidencePrediction:
    prompt_text = build_few_shot_user_prompt(demo_examples, example["question"])
    messages = [{"role": "user", "content": prompt_text}]

    response_text = ""
    prompt_tokens = 0
    completion_tokens = 0
    parsed_confidence = None

    for attempt in range(max_retries):
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_completion_tokens=max_completion_tokens,
                ),
                timeout=timeout,
            )
            response_text = response.choices[0].message.content or ""
            if response.choices[0].message.reasoning is not None:
                response_text = response.choices[0].message.reasoning + "</think>" + response_text
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0

            parsed_confidence = parse_confidence(response_text)
            if parsed_confidence is not None:
                break
        except asyncio.TimeoutError:
            response_text = f"[TIMEOUT after {timeout}s on attempt {attempt + 1}]"
        except Exception as e:
            response_text = f"[ERROR on attempt {attempt + 1}: {str(e)}]"

    parse_success = parsed_confidence is not None
    if not parse_success:
        parsed_confidence = 0.5

    return VerbalizedConfidencePrediction(
        example_id=example["example_id"],
        confidence=parsed_confidence,
        expected_accuracy=example["expected_accuracy"],
        num_samples=example["num_samples"],
        num_correct=example["num_correct"],
        prompt=messages,
        response=response_text,
        parsed_response=str(parsed_confidence),
        parse_success=parse_success,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


async def run_all_predictions(
    examples: list[dict],
    demo_examples: list[dict],
    client: AsyncOpenAI,
    model: str,
    max_concurrent: int,
    timeout: int,
    output_path: Path,
    existing_ids: set[str],
    max_completion_tokens: int = 8192,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> list[VerbalizedConfidencePrediction]:
    examples_to_process = [ex for ex in examples if ex["example_id"] not in existing_ids]

    if not examples_to_process:
        print("All examples already processed. Loading from file...")
        return []

    print(f"Processing {len(examples_to_process)} examples ({len(existing_ids)} already done)")

    results: list[VerbalizedConfidencePrediction] = []
    write_lock = asyncio.Lock()

    async def process_and_save(example: dict) -> VerbalizedConfidencePrediction:
        result = await predict_confidence(
            client,
            example,
            demo_examples,
            model,
            timeout,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        async with write_lock:
            with open(output_path, "a") as f:
                f.write(json.dumps(asdict(result)) + "\n")
        return result

    def task_generator():
        for example in examples_to_process:
            yield process_and_save(example)

    pending: set[asyncio.Task] = set()
    task_iter = iter(task_generator())

    with tqdm(total=len(examples_to_process), desc="Predicting confidence") as pbar:
        for coro in task_iter:
            pending.add(asyncio.create_task(coro))
            if len(pending) >= max_concurrent:
                break

        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                results.append(task.result())
            pbar.update(len(done))

            while len(pending) < max_concurrent:
                try:
                    coro = next(task_iter)
                    pending.add(asyncio.create_task(coro))
                except StopIteration:
                    break

    return results


def load_existing_predictions(output_path: Path) -> tuple[list[VerbalizedConfidencePrediction], set[str]]:
    existing: list[VerbalizedConfidencePrediction] = []
    existing_ids: set[str] = set()

    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                data = json.loads(line)
                existing.append(VerbalizedConfidencePrediction(**data))
                existing_ids.add(data["example_id"])
        print(f"Loaded {len(existing)} existing predictions")

    return existing, existing_ids


def reparse_and_archive_existing(
    output_dir: Path,
    predictions_path: Path,
    metrics_path: Path,
    diagram_path: Path,
) -> list[VerbalizedConfidencePrediction]:
    if not predictions_path.exists():
        return []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    old_predictions_path = output_dir / f"old_confidence_predictions_{timestamp}.jsonl"
    predictions_path.rename(old_predictions_path)
    print(f"Renamed {predictions_path.name} -> {old_predictions_path.name}")

    if metrics_path.exists():
        old_metrics_path = output_dir / f"prev_evaluation_metrics_{timestamp}.json"
        metrics_path.rename(old_metrics_path)
        print(f"Renamed {metrics_path.name} -> {old_metrics_path.name}")

    if diagram_path.exists():
        old_diagram_path = output_dir / f"prev_reliability_diagram_{timestamp}.png"
        diagram_path.rename(old_diagram_path)
        print(f"Renamed {diagram_path.name} -> {old_diagram_path.name}")

    reparsed: list[VerbalizedConfidencePrediction] = []
    with open(old_predictions_path) as f:
        for line in f:
            data = json.loads(line)
            new_confidence = parse_confidence(data["response"])
            parse_success = new_confidence is not None
            if not parse_success:
                new_confidence = 0.5

            reparsed.append(
                VerbalizedConfidencePrediction(
                    example_id=data["example_id"],
                    confidence=new_confidence,
                    expected_accuracy=data["expected_accuracy"],
                    num_samples=data["num_samples"],
                    num_correct=data["num_correct"],
                    prompt=data["prompt"],
                    response=data["response"],
                    parsed_response=str(new_confidence),
                    parse_success=parse_success,
                    prompt_tokens=data["prompt_tokens"],
                    completion_tokens=data["completion_tokens"],
                )
            )

    with open(predictions_path, "w") as f:
        for pred in reparsed:
            f.write(json.dumps(asdict(pred)) + "\n")

    print(f"Re-parsed {len(reparsed)} predictions and wrote to {predictions_path.name}")
    return reparsed


def write_few_shot_meta(
    path: Path,
    *,
    few_shot_k: int,
    few_shot_seed: int,
    few_shot_sampling_strategy: str,
    train_ground_truth_jsonl: str,
    demo_example_ids: list[str],
    demo_confidences: list[float],
) -> None:
    meta = {
        "few_shot_k": few_shot_k,
        "few_shot_seed": few_shot_seed,
        "few_shot_sampling_strategy": few_shot_sampling_strategy,
        "train_ground_truth_jsonl": train_ground_truth_jsonl,
        "demo_example_ids": demo_example_ids,
        "demo_confidences": demo_confidences,
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {path.name} with {len(demo_example_ids)} demo IDs and confidences")


def main() -> None:
    parser = argparse.ArgumentParser(description="Few-shot verbalized confidence estimation")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--base_url", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_concurrent", type=int, default=500)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--ground_truth_jsonl",
        type=str,
        required=True,
        help="Test set ground_truth.jsonl",
    )
    parser.add_argument(
        "--train_ground_truth_jsonl",
        type=str,
        required=True,
        help="Train pool for few-shot demos (same schema as test)",
    )
    parser.add_argument("--few_shot_k", type=int, required=True)
    parser.add_argument("--few_shot_seed", type=int, required=True)
    parser.add_argument(
        "--few_shot_sampling_strategy",
        type=str,
        default="stratified",
        choices=["stratified", "random"],
        help="Sampling strategy: 'stratified' for diversity across confidence bins, 'random' for uniform sampling (default: stratified)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )
    parser.add_argument("--max_completion_tokens", type=int, default=8000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "confidence_predictions.jsonl"
    metrics_path = output_dir / "evaluation_metrics.json"
    diagram_path = output_dir / "reliability_diagram.png"
    meta_path = output_dir / "few_shot_meta.json"

    train_rows: list[dict] = []
    with open(args.train_ground_truth_jsonl) as f:
        for line in f:
            train_rows.append(json.loads(line))
    print(f"Loaded {len(train_rows)} train pool examples from {args.train_ground_truth_jsonl}")

    demo_examples = sample_few_shot_demos(
        train_rows, args.few_shot_k, args.few_shot_seed, strategy=args.few_shot_sampling_strategy
    )
    demo_ids = [ex["example_id"] for ex in demo_examples]
    demo_confidences = [ex["expected_accuracy"] for ex in demo_examples]
    print(f"Sampled {len(demo_examples)} few-shot demos (strategy={args.few_shot_sampling_strategy}, seed={args.few_shot_seed})")
    print(f"  Demo IDs: {demo_ids}")
    print(f"  Demo confidences: {[f'{c:.3f}' for c in demo_confidences]} (min={min(demo_confidences):.3f}, max={max(demo_confidences):.3f})")

    write_few_shot_meta(
        meta_path,
        few_shot_k=args.few_shot_k,
        few_shot_seed=args.few_shot_seed,
        few_shot_sampling_strategy=args.few_shot_sampling_strategy,
        train_ground_truth_jsonl=args.train_ground_truth_jsonl,
        demo_example_ids=demo_ids,
        demo_confidences=demo_confidences,
    )

    examples: list[dict] = []
    with open(args.ground_truth_jsonl) as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"Loaded {len(examples)} test examples from {args.ground_truth_jsonl}")

    reparse_and_archive_existing(output_dir, predictions_path, metrics_path, diagram_path)

    existing_predictions, existing_ids = load_existing_predictions(predictions_path)

    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key="EMPTY",
    )

    new_predictions = asyncio.run(
        run_all_predictions(
            examples=examples,
            demo_examples=demo_examples,
            client=client,
            model=args.model,
            max_concurrent=args.max_concurrent,
            timeout=args.timeout,
            output_path=predictions_path,
            existing_ids=existing_ids,
            max_completion_tokens=args.max_completion_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    )

    all_predictions = existing_predictions + new_predictions
    print(f"Total predictions: {len(all_predictions)}")

    parse_failures = sum(1 for p in all_predictions if not p.parse_success)
    if parse_failures > 0:
        print(f"Warning: {parse_failures} predictions had parse failures (using fallback 0.5)")

    total_prompt_tokens = sum(p.prompt_tokens for p in all_predictions)
    total_completion_tokens = sum(p.completion_tokens for p in all_predictions)
    print(f"Total tokens: {total_prompt_tokens:,} prompt + {total_completion_tokens:,} completion")

    metrics = compute_metrics(all_predictions)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    print(f"  MSE: {metrics['mse']:.4f}")
    print(f"  ECE: {metrics['ece']:.4f}")
    print(f"  Pearson r: {metrics['pearson_r']:.4f}")
    print(f"  Spearman r: {metrics['spearman_r']:.4f}")

    plot_reliability_diagram(all_predictions, diagram_path)
    print(f"Reliability diagram saved to {diagram_path}")


if __name__ == "__main__":
    main()
