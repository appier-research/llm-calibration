#!/usr/bin/env python
"""
Evaluate consistency-based confidence estimation from pre-sampled outputs.

Usage:
    python scripts/consistency.py \
        --sampled_path outputs/gsm8k-test__Qwen3-8B-non-thinking/sampled.jsonl \
        --ground_truth_path outputs/gsm8k-test__Qwen3-8B-non-thinking/ground_truth.jsonl \
        --output_dir estimator_results/consistency/gsm8k-test__Qwen3-8B-non-thinking \
        --k_values 5 10 20 \
        --seed 42
"""

import argparse
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

from src.metrics.custom import c_star_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate consistency-based confidence from sampled outputs"
    )
    parser.add_argument(
        "--sampled_path",
        type=str,
        required=True,
        help="Path to sampled.jsonl with model outputs",
    )
    parser.add_argument(
        "--ground_truth_path",
        type=str,
        required=True,
        help="Path to ground_truth.jsonl with expected_accuracy",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save consistency results (NOT in ./outputs)",
    )
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help="Values of k to evaluate (default: 5 10 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    return parser.parse_args()


def load_ground_truth(path: str) -> dict[str, float]:
    """Load ground truth expected accuracies."""
    ground_truth = {}
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            ground_truth[item["example_id"]] = item["expected_accuracy"]
    return ground_truth


def load_samples(path: str) -> dict[str, list[dict]]:
    """Load all samples grouped by example_id."""
    samples_by_id = defaultdict(list)
    with open(path) as f:
        for line in f:
            sample = json.loads(line)
            samples_by_id[sample["example_id"]].append(sample)
    return dict(samples_by_id)


def compute_majority_vote_confidence(samples: list[dict], k: int, seed: int) -> dict:
    """
    Compute confidence via majority vote on k randomly sampled outputs.

    Confidence = frequency of most common answer / k
    """
    # Set seed for reproducibility
    random.seed(seed)

    # Randomly sample k samples without replacement
    if len(samples) < k:
        raise ValueError(f"Not enough samples: {len(samples)} < {k}")

    k_samples = random.sample(samples, k=k)

    # Count answer frequencies
    answer_counts = Counter(s["parsed_response"] for s in k_samples)
    most_common_answer, count = answer_counts.most_common(1)[0]

    return {
        "confidence": count / k,
        "most_common_answer": most_common_answer,
        "answer_frequency": count,
        "num_unique_answers": len(answer_counts),
    }


def evaluate_consistency_for_k(
    samples_by_id: dict[str, list[dict]],
    ground_truth: dict[str, float],
    k: int,
    seed: int,
) -> list[dict]:
    """Evaluate consistency-based confidence for a given k value."""
    results = []

    for example_id in tqdm(
        sorted(ground_truth.keys()),
        desc=f"Evaluating k={k}",
        unit="examples",
    ):
        samples = samples_by_id.get(example_id)
        if samples is None:
            logger.warning(f"No samples found for {example_id}")
            continue

        # Compute confidence
        result = compute_majority_vote_confidence(samples, k, seed)

        # Add metadata
        result["example_id"] = example_id
        result["expected_accuracy"] = ground_truth[example_id]
        result["k"] = k

        results.append(result)

    return results


def save_results(results: list[dict], output_path: Path) -> None:
    """Save results to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


def save_metrics(metrics: dict, output_path: Path) -> None:
    """Save metrics to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)


def main():
    args = parse_args()

    # Load data
    logger.info(f"Loading ground truth from {args.ground_truth_path}")
    ground_truth = load_ground_truth(args.ground_truth_path)

    logger.info(f"Loading samples from {args.sampled_path}")
    samples_by_id = load_samples(args.sampled_path)

    logger.info(f"Loaded {len(ground_truth)} examples with {sum(len(s) for s in samples_by_id.values())} total samples")

    output_dir = Path(args.output_dir)

    # Evaluate for each k
    all_metrics = []
    for k in args.k_values:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating k={k}")
        logger.info(f"{'='*60}")

        # Compute results
        results = evaluate_consistency_for_k(
            samples_by_id=samples_by_id,
            ground_truth=ground_truth,
            k=k,
            seed=args.seed,
        )

        # Save confidence file
        confidence_path = output_dir / f"consistency_k{k}.jsonl"
        save_results(results, confidence_path)
        logger.info(f"Saved confidence to: {confidence_path}")

        # Compute metrics
        confidences = [r["confidence"] for r in results]
        c_star = [r["expected_accuracy"] for r in results]

        metrics_obj = c_star_metrics(confidences, c_star)

        metrics = {
            "k": k,
            "mse": metrics_obj.mse,
            "ece": metrics_obj.ece,
            "pearson_r": metrics_obj.pearson_r,
            "spearman_r": metrics_obj.spearman_r,
            "num_examples": metrics_obj.num_examples,
            "seed": args.seed,
        }

        # Save metrics
        metrics_path = output_dir / f"consistency_k{k}_metrics.json"
        save_metrics(metrics, metrics_path)
        logger.info(f"Saved metrics to: {metrics_path}")

        # Print summary
        logger.info(f"\nMetrics for k={k}:")
        logger.info(f"  MSE:        {metrics['mse']:.4f}")
        logger.info(f"  ECE:        {metrics['ece']:.4f}")
        logger.info(f"  Pearson r:  {metrics['pearson_r']:.4f}")
        logger.info(f"  Spearman r: {metrics['spearman_r']:.4f}")
        logger.info(f"  Examples:   {metrics['num_examples']}")

        all_metrics.append(metrics)

    # Save summary
    summary_path = output_dir / "summary.json"
    save_metrics({
        "k_values": args.k_values,
        "seed": args.seed,
        "sampled_path": args.sampled_path,
        "ground_truth_path": args.ground_truth_path,
        "metrics_by_k": all_metrics,
    }, summary_path)
    logger.info(f"\nSaved summary to: {summary_path}")

    logger.info(f"\n{'='*60}")
    logger.info("Evaluation complete!")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
