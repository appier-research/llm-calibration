#!/usr/bin/env python
"""
Verbalized confidence estimation for calibration.

Usage:
    uv run python scripts/verbalize_confidence.py \
        --model "openai/gpt-oss-20b" \
        --base_url "<your_base_url>" \
        --temperature 1.0 \
        --top_p 1.0 \
        --max_concurrent 500 \
        --ground_truth_jsonl "outputs/triviaqa-test__gpt-oss-20b/ground_truth.jsonl" \
        --output_dir "estimator_results/verbalized_confidence/triviaqa-test__gpt-oss-20b" \
        --max_completion_tokens 5000

Input ground_truth.jsonl fields:
    {example_id: str, question: str, expected_accuracy: float, num_samples: int, num_correct: int}

Output files:
    * confidence_predictions.jsonl: per-example predictions
    * evaluation_metrics.json: {mse, ece, pearson_r, spearman_r}
    * reliability_diagram.png: reliability diagram plot
"""

import argparse
import asyncio
import json
import re
from dataclasses import dataclass, asdict
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


def get_prompt(question: str) -> str:
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


def parse_confidence(response: str) -> float | None:
    """Extract probability from \\boxed{P} format."""
    if "<think>" in response:
        response = response.split("</think>")[1]
    matches = re.findall(r'\\boxed\{([0-9.]+)\}', response)
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
    """Compute Expected Calibration Error with continuous expected_accuracy."""
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
    """Compute calibration metrics."""
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
    """Plot reliability diagram with bins [0, 0.1), [0.1, 0.2), ..."""
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

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    # Bar plot for actual accuracy per bin
    bar_width = 0.08
    ax.bar(bin_centers, bin_accs, width=bar_width, alpha=0.7, label="Actual accuracy", color="steelblue")

    # Scatter for bin centers
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

    # Add bin counts as text
    for i, (center, count) in enumerate(zip(bin_centers, bin_counts)):
        if count > 0:
            ax.text(center, 0.02, f"n={count}", ha="center", fontsize=8, alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


async def predict_confidence(
    client: AsyncOpenAI,
    example: dict,
    model: str,
    timeout: int,
    max_retries: int = 3,
    max_completion_tokens: int = 8192,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> VerbalizedConfidencePrediction:
    """Get verbalized confidence for a single example with retries."""
    prompt_text = get_prompt(example["question"])
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
                break  # Successfully parsed
        except asyncio.TimeoutError:
            response_text = f"[TIMEOUT after {timeout}s on attempt {attempt + 1}]"
        except Exception as e:
            response_text = f"[ERROR on attempt {attempt + 1}: {str(e)}]"

    # Fallback if all retries failed
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
    """Run predictions with sliding window concurrency and streaming output."""
    # Filter out already processed examples
    examples_to_process = [ex for ex in examples if ex["example_id"] not in existing_ids]

    if not examples_to_process:
        print("All examples already processed. Loading from file...")
        return []

    print(f"Processing {len(examples_to_process)} examples ({len(existing_ids)} already done)")

    results: list[VerbalizedConfidencePrediction] = []
    write_lock = asyncio.Lock()

    async def process_and_save(example: dict) -> VerbalizedConfidencePrediction:
        result = await predict_confidence(client, example, model, timeout, max_completion_tokens=max_completion_tokens, temperature=temperature, top_p=top_p)
        async with write_lock:
            with open(output_path, "a") as f:
                f.write(json.dumps(asdict(result)) + "\n")
        return result

    # Task generator
    def task_generator():
        for example in examples_to_process:
            yield process_and_save(example)

    pending: set[asyncio.Task] = set()
    task_iter = iter(task_generator())

    with tqdm(total=len(examples_to_process), desc="Predicting confidence") as pbar:
        # Fill initial window
        for coro in task_iter:
            pending.add(asyncio.create_task(coro))
            if len(pending) >= max_concurrent:
                break

        # Sliding window processing
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                results.append(task.result())
            pbar.update(len(done))

            # Refill window
            while len(pending) < max_concurrent:
                try:
                    coro = next(task_iter)
                    pending.add(asyncio.create_task(coro))
                except StopIteration:
                    break

    return results


def load_existing_predictions(output_path: Path) -> tuple[list[VerbalizedConfidencePrediction], set[str]]:
    """Load existing predictions from file for resume capability."""
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
    """Rename existing files with timestamp and re-parse responses."""
    if not predictions_path.exists():
        return []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Rename existing files
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

    # Re-parse responses from the archived file
    reparsed: list[VerbalizedConfidencePrediction] = []
    with open(old_predictions_path) as f:
        for line in f:
            data = json.loads(line)
            # Re-parse the response with the current parse_confidence function
            new_confidence = parse_confidence(data["response"])
            parse_success = new_confidence is not None
            if not parse_success:
                new_confidence = 0.5

            reparsed.append(VerbalizedConfidencePrediction(
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
            ))

    # Write re-parsed predictions to new file
    with open(predictions_path, "w") as f:
        for pred in reparsed:
            f.write(json.dumps(asdict(pred)) + "\n")

    print(f"Re-parsed {len(reparsed)} predictions and wrote to {predictions_path.name}")
    return reparsed


def main():
    parser = argparse.ArgumentParser(description="Verbalized confidence estimation")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--base_url", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_concurrent", type=int, default=500)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--ground_truth_jsonl",
        type=str,
        default="outputs/aime25-test__gpt-oss-20b__A40/ground_truth.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="estimator_results/verbalized_confidence/aime25-test__gpt-oss-20b__A40",
    )
    parser.add_argument(
        "--max_completion_tokens",
        type=int,
        default=8000,
    )
    args = parser.parse_args()

    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "confidence_predictions.jsonl"
    metrics_path = output_dir / "evaluation_metrics.json"
    diagram_path = output_dir / "reliability_diagram.png"

    # Load ground truth examples
    examples = []
    with open(args.ground_truth_jsonl) as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"Loaded {len(examples)} examples from {args.ground_truth_jsonl}")

    # Archive existing files and re-parse responses if they exist
    reparse_and_archive_existing(
        output_dir, predictions_path, metrics_path, diagram_path
    )

    # Load existing predictions for resume (now contains re-parsed data)
    existing_predictions, existing_ids = load_existing_predictions(predictions_path)

    # Setup client
    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key="EMPTY",  # Often not needed for local endpoints
    )

    # Run predictions
    new_predictions = asyncio.run(
        run_all_predictions(
            examples=examples,
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

    # Combine existing + new predictions
    all_predictions = existing_predictions + new_predictions
    print(f"Total predictions: {len(all_predictions)}")

    # Report parse failures
    parse_failures = sum(1 for p in all_predictions if not p.parse_success)
    if parse_failures > 0:
        print(f"Warning: {parse_failures} predictions had parse failures (using fallback 0.5)")

    # Report token usage
    total_prompt_tokens = sum(p.prompt_tokens for p in all_predictions)
    total_completion_tokens = sum(p.completion_tokens for p in all_predictions)
    print(f"Total tokens: {total_prompt_tokens:,} prompt + {total_completion_tokens:,} completion")

    # Compute and save metrics
    metrics = compute_metrics(all_predictions)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    print(f"  MSE: {metrics['mse']:.4f}")
    print(f"  ECE: {metrics['ece']:.4f}")
    print(f"  Pearson r: {metrics['pearson_r']:.4f}")
    print(f"  Spearman r: {metrics['spearman_r']:.4f}")

    # Plot reliability diagram
    plot_reliability_diagram(all_predictions, diagram_path)
    print(f"Reliability diagram saved to {diagram_path}")


if __name__ == "__main__":
    main()
