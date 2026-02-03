#!/usr/bin/env python
"""
P(True) confidence estimation using token logprobs.

Usage:
    base_url="<your_base_url>"
    uv run python scripts/ptrue.py \
        --model "Qwen/Qwen3-8B" \
        --base_url ${base_url} \
        --temperature 0.7 \
        --top_p 0.8 \
        --max_concurrent 1000 \
        --ground_truth_jsonl "outputs/triviaqa-validation__Qwen3-8B-non-thinking/ground_truth.jsonl" \
        --output_dir "estimator_results/ptrue/triviaqa-validation__Qwen3-8B-non-thinking" \
        --max_completion_tokens 1  # should change to ~8192 for reasoning LMs (e.g., gpt-oss-20b)

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
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from openai import AsyncOpenAI
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm


@dataclass
class PTrueConfidencePrediction:
    example_id: str
    confidence: float
    expected_accuracy: float
    num_samples: int
    num_correct: int
    prompt_tokens: int
    completion_tokens: int
    prompt: list[dict[str, str]]
    response: str
    parsed_response: dict[str, float]  # {"Yes": prob, "No": prob}
    parse_success: bool  # True if at least one of Yes/No in top-k


def strip_all_lines(text: str) -> str:
    return "\n".join(line.strip() for line in text.split("\n"))


def get_prompt(question: str) -> str:
    prompt = f"""
    Question: {question}

    Are you able to answer the question correctly?
    Answer with only a single word: Yes or No.
    """.strip()
    return strip_all_lines(prompt)


def find_answer_token_index(tokens: list[str]) -> int:
    """
    Find the index of the first answer token.
    
    Supports multiple reasoning model formats:
    - </think> format: return index after </think>
    - <|channel|>final<|message|> format: return index after <|message|>
    - Non-reasoning models: first token (index 0)
    """
    # Check for </think> format (e.g., Qwen, DeepSeek reasoning models)
    for i, token in enumerate(tokens):
        if "</think>" in token:
            return i + 1
    
    # Check for <|channel|>final<|message|> format (e.g., gpt-oss reasoning models)
    # Pattern: ... <|channel|> final <|message|> ANSWER ...
    for i, token in enumerate(tokens):
        if token == "final" and i > 0 and tokens[i - 1] == "<|channel|>":
            # Look for <|message|> after "final"
            for j in range(i + 1, len(tokens)):
                if tokens[j] == "<|message|>":
                    return j + 1
    
    # Non-reasoning model: use first token
    return 0


def compute_ptrue_confidence(top_logprobs: list) -> tuple[dict[str, float], bool]:
    """
    Extract Yes/No logprobs from top_logprobs and apply softmax.
    
    Args:
        top_logprobs: List of top logprob objects at a token position.
                      Each has .token and .logprob attributes.
    
    Returns:
        (softmax_probs_dict, parse_success)
        - softmax_probs_dict: {"Yes": prob, "No": prob}
        - parse_success: True if at least one of Yes/No was found
    """
    yes_logprob = None
    no_logprob = None
    
    for item in top_logprobs:
        if item.token == "Yes":
            yes_logprob = item.logprob
        elif item.token == "No":
            no_logprob = item.logprob
    
    # Handle missing tokens with floor logprob
    FLOOR_LOGPROB = -100.0
    
    if yes_logprob is None and no_logprob is None:
        # Neither token found - fallback to 0.5
        return {"Yes": 0.5, "No": 0.5}, False
    
    # At least one token found - use floor for missing
    yes_logprob = yes_logprob if yes_logprob is not None else FLOOR_LOGPROB
    no_logprob = no_logprob if no_logprob is not None else FLOOR_LOGPROB
    
    # Softmax with numerical stability
    max_lp = max(yes_logprob, no_logprob)
    yes_prob = math.exp(yes_logprob - max_lp)
    no_prob = math.exp(no_logprob - max_lp)
    total = yes_prob + no_prob
    
    return {"Yes": yes_prob / total, "No": no_prob / total}, True


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


def compute_metrics(predictions: list[PTrueConfidencePrediction]) -> dict:
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
    predictions: list[PTrueConfidencePrediction],
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
    ax.set_title("Reliability Diagram (P(True))", fontsize=14)
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
    max_completion_tokens: int = 1,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_logprobs: int = 20,
) -> PTrueConfidencePrediction:
    """Get P(True) confidence for a single example with retries."""
    prompt_text = get_prompt(example["question"])
    messages = [{"role": "user", "content": prompt_text}]

    response_text = ""
    prompt_tokens = 0
    completion_tokens = 0
    parsed_probs = {"Yes": 0.5, "No": 0.5}
    parse_success = False

    for attempt in range(max_retries):
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_completion_tokens=max_completion_tokens,
                    logprobs=True,
                    top_logprobs=top_logprobs,
                ),
                timeout=timeout,
            )
            
            choice = response.choices[0]
            response_text = choice.message.content or ""
            
            # Handle reasoning models that return reasoning separately
            if choice.message.reasoning is not None:
                response_text = choice.message.reasoning + "</think>" + response_text
            
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0

            # Extract logprobs
            if choice.logprobs and choice.logprobs.content:
                # Get all tokens for finding answer position
                tokens = [t.token for t in choice.logprobs.content]
                
                # Find the answer token index
                answer_idx = find_answer_token_index(tokens)
                
                if answer_idx < len(choice.logprobs.content):
                    # Get top_logprobs at the answer position
                    answer_token_info = choice.logprobs.content[answer_idx]
                    if answer_token_info.top_logprobs:
                        parsed_probs, parse_success = compute_ptrue_confidence(
                            answer_token_info.top_logprobs
                        )
                        break  # Successfully parsed
                    else:
                        response_text += " [No top_logprobs at answer position]"
                else:
                    response_text += f" [Answer index {answer_idx} out of bounds, only {len(choice.logprobs.content)} tokens]"
            else:
                response_text += " [No logprobs returned]"
                
        except asyncio.TimeoutError:
            response_text = f"[TIMEOUT after {timeout}s on attempt {attempt + 1}]"
        except Exception as e:
            response_text = f"[ERROR on attempt {attempt + 1}: {str(e)}]"

    # Confidence is P(Yes)
    confidence = parsed_probs["Yes"]

    return PTrueConfidencePrediction(
        example_id=example["example_id"],
        confidence=confidence,
        expected_accuracy=example["expected_accuracy"],
        num_samples=example["num_samples"],
        num_correct=example["num_correct"],
        prompt=messages,
        response=response_text,
        parsed_response=parsed_probs,
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
    max_completion_tokens: int = 1,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_logprobs: int = 20,
) -> list[PTrueConfidencePrediction]:
    """Run predictions with sliding window concurrency and streaming output."""
    # Filter out already processed examples
    examples_to_process = [ex for ex in examples if ex["example_id"] not in existing_ids]

    if not examples_to_process:
        print("All examples already processed. Loading from file...")
        return []

    print(f"Processing {len(examples_to_process)} examples ({len(existing_ids)} already done)")

    results: list[PTrueConfidencePrediction] = []
    write_lock = asyncio.Lock()

    async def process_and_save(example: dict) -> PTrueConfidencePrediction:
        result = await predict_confidence(
            client, example, model, timeout,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            top_p=top_p,
            top_logprobs=top_logprobs,
        )
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

    with tqdm(total=len(examples_to_process), desc="Predicting P(True)") as pbar:
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


def load_existing_predictions(output_path: Path) -> tuple[list[PTrueConfidencePrediction], set[str]]:
    """Load existing predictions from file for resume capability."""
    existing: list[PTrueConfidencePrediction] = []
    existing_ids: set[str] = set()

    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                data = json.loads(line)
                existing.append(PTrueConfidencePrediction(**data))
                existing_ids.add(data["example_id"])
        print(f"Loaded {len(existing)} existing predictions")

    return existing, existing_ids


def reparse_and_archive_existing(
    output_dir: Path,
    predictions_path: Path,
    metrics_path: Path,
    diagram_path: Path,
) -> list[PTrueConfidencePrediction]:
    """Rename existing files with timestamp. Re-parsing not applicable for P(True)."""
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

    # Load archived predictions (no re-parsing needed for P(True) since logprobs are stored)
    reparsed: list[PTrueConfidencePrediction] = []
    with open(old_predictions_path) as f:
        for line in f:
            data = json.loads(line)
            reparsed.append(PTrueConfidencePrediction(**data))

    # Write to new file
    with open(predictions_path, "w") as f:
        for pred in reparsed:
            f.write(json.dumps(asdict(pred)) + "\n")

    print(f"Copied {len(reparsed)} predictions to {predictions_path.name}")
    return reparsed


def main():
    parser = argparse.ArgumentParser(description="P(True) confidence estimation")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--base_url", type=str, required=True)
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
        default="estimator_results/ptrue/aime25-test__gpt-oss-20b__A40",
    )
    parser.add_argument(
        "--max_completion_tokens",
        type=int,
        default=1,
        help="Max tokens to generate. Use 1 for non-reasoning models, higher for reasoning models.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top p for sampling.",
    )
    parser.add_argument(
        "--top_logprobs",
        type=int,
        default=20,
        help="Number of top logprobs to retrieve (max 20).",
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

    # Archive existing files if they exist
    reparse_and_archive_existing(
        output_dir, predictions_path, metrics_path, diagram_path
    )

    # Load existing predictions for resume
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
            top_logprobs=args.top_logprobs,
        )
    )

    # Combine existing + new predictions
    all_predictions = existing_predictions + new_predictions
    print(f"Total predictions: {len(all_predictions)}")

    # Report parse failures
    parse_failures = sum(1 for p in all_predictions if not p.parse_success)
    if parse_failures > 0:
        print(f"Warning: {parse_failures} predictions had parse failures (neither Yes nor No in top-{args.top_logprobs})")

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
