#!/usr/bin/env python
"""
Construct probing dataset by extracting last-token hidden states from all layers.

Three modes are supported:
1. expected_accuracy (default): Extract hidden states from prompts only, target is expected_accuracy.
2. sample_correctness: Extract hidden states from prompt+response, target is per-sample correctness.
3. verbalized_confidence: Extract hidden states from prompt+response (from confidence_predictions.jsonl),
   target is expected_accuracy.

Usage (expected_accuracy mode):
    uv run python scripts/construct_probing_dataset.py \
        --model_id allenai/Olmo-3-7B-Instruct \
        --ground_truth_path outputs/triviaqa-train__Olmo-3-7B-Instruct__RTXPro6000/ground_truth.jsonl \
        --sampled_path outputs/triviaqa-train__Olmo-3-7B-Instruct__RTXPro6000/sampled.jsonl \
        --output_dir outputs/triviaqa-train__Olmo-3-7B-Instruct__RTXPro6000/probing_dataset \
        --batch_size 8

Usage (sample_correctness mode):
    uv run python scripts/construct_probing_dataset.py \
        --model_id allenai/Olmo-3-7B-Instruct \
        --sampled_path outputs/triviaqa-train__Olmo-3-7B-Instruct__RTXPro6000/sampled.jsonl \
        --output_dir outputs/triviaqa-train__Olmo-3-7B-Instruct__RTXPro6000/probing_dataset_sample \
        --mode sample_correctness \
        --batch_size 8

Usage (verbalized_confidence mode):
    uv run python scripts/construct_probing_dataset.py \
        --model_id Qwen/Qwen3-8B \
        --confidence_predictions_path estimator_results/verbalized_confidence/math-valid__Qwen3-8B-non-thinking__RTXPro6000/confidence_predictions.jsonl \
        --output_dir outputs/math-valid__Qwen3-8B__probing_verbalized \
        --mode verbalized_confidence \
        --batch_size 8

Output format (safetensors):
    - hidden_states.safetensors: shape (n_examples, n_layers, hidden_dim)
    - targets.safetensors: shape (n_examples,)
    - metadata.json: example_ids and model info
"""

import argparse
import json
import logging
import random
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct probing dataset from hidden states"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g., allenai/OLMo-3-7B-Instruct)",
    )
    parser.add_argument(
        "--ground_truth_path",
        type=str,
        default=None,
        help="Path to ground_truth.jsonl with example_id and expected_accuracy (required for expected_accuracy mode)",
    )
    parser.add_argument(
        "--sampled_path",
        type=str,
        default=None,
        help="Path to sampled.jsonl with example_id and prompt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the probing dataset",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["expected_accuracy", "sample_correctness", "verbalized_confidence"],
        default="expected_accuracy",
        help="Mode: 'expected_accuracy' (probe on aggregated accuracy), 'sample_correctness' (probe on per-sample correctness), or 'verbalized_confidence' (probe on verbalized confidence predictions)",
    )
    parser.add_argument(
        "--confidence_predictions_path",
        type=str,
        default=None,
        help="Path to confidence_predictions.jsonl (required for verbalized_confidence mode)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference (default: 8)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for inference (default: cuda:0)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="hf_models",
        help="HuggingFace cache directory (default: hf_models)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for selecting which sampled_id to use per example_id (sample_correctness mode), and for --n subsampling",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="If set, randomly subsample N instances with unique example_ids from all instances (uses --seed for determinism)",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == "expected_accuracy" and args.ground_truth_path is None:
        parser.error("--ground_truth_path is required for expected_accuracy mode")
    if args.mode == "verbalized_confidence" and args.confidence_predictions_path is None:
        parser.error("--confidence_predictions_path is required for verbalized_confidence mode")
    
    return args


def load_ground_truth(path: str) -> list[dict]:
    """Load ground truth data with example_id and expected_accuracy."""
    data = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            data.append({
                "example_id": item["example_id"],
                "expected_accuracy": item["expected_accuracy"],
            })
    return data


def build_prompt_lookup(sampled_path: str, example_ids: set[str]) -> dict[str, list]:
    """
    Build a lookup from example_id to prompt.
    Only stores the first sample for each example_id.
    """
    logger.info(f"Building prompt lookup from {sampled_path}...")
    prompt_lookup = {}
    
    with open(sampled_path) as f:
        for line in tqdm(f, desc="Scanning sampled.jsonl"):
            item = json.loads(line)
            example_id = item["example_id"]
            
            # Only store if we need this example_id and haven't seen it yet
            if example_id in example_ids and example_id not in prompt_lookup:
                prompt_lookup[example_id] = item["prompt"]
            
            # Early exit if we have all needed prompts
            if len(prompt_lookup) == len(example_ids):
                break
    
    return prompt_lookup


def load_samples_for_correctness(sampled_path: str, seed: int = 0) -> list[dict]:
    """
    Load one sample per example_id from sampled.jsonl for sample_correctness mode.
    
    The seed determines which sampled_id is selected for each example_id.
    
    Each sample becomes a separate training example with:
    - sample_id: "{example_id}_{sampled_id}"
    - prompt: the conversation prompt
    - response: the model's response
    - target: correctness (0 or 1)
    """
    logger.info(f"Loading samples from {sampled_path} (one per example_id, seed={seed})...")
    
    # First pass: group all samples by example_id
    samples_by_example = {}
    with open(sampled_path) as f:
        for line in tqdm(f, desc="Loading samples"):
            item = json.loads(line)
            example_id = item["example_id"]
            
            if example_id not in samples_by_example:
                samples_by_example[example_id] = []
            
            samples_by_example[example_id].append({
                "example_id": example_id,
                "sample_id": f"{example_id}_{item['sampled_id']}",
                "prompt": item["prompt"],
                "response": item["response"],
                "target": float(item["correctness"]),  # 0 or 1
            })
    
    # Second pass: randomly select one sample per example_id using seed
    rng = random.Random(seed)
    samples = []
    for example_id in sorted(samples_by_example.keys()):  # Sort for determinism
        candidates = samples_by_example[example_id]
        selected = rng.choice(candidates)
        samples.append(selected)
    
    logger.info(f"Loaded {len(samples)} samples (one per example_id)")
    return samples


def load_verbalized_confidence_data(confidence_predictions_path: str) -> list[dict]:
    """
    Load verbalized confidence predictions for the verbalized_confidence mode.
    
    Each entry contains:
    - example_id: unique identifier
    - prompt: list of message dicts (conversation format)
    - response: assistant's response string
    - expected_accuracy: target value (float 0-1)
    """
    logger.info(f"Loading verbalized confidence data from {confidence_predictions_path}...")
    data = []
    
    with open(confidence_predictions_path) as f:
        for line in tqdm(f, desc="Loading confidence predictions"):
            item = json.loads(line)
            data.append({
                "example_id": item["example_id"],
                "prompt": item["prompt"],  # Already a list of message dicts
                "response": item["response"],
                "target": item["expected_accuracy"],
            })
    
    logger.info(f"Loaded {len(data)} examples")
    return data


def subsample_unique(
    items: list[dict], n: int, seed: int, id_key: str = "example_id"
) -> list[dict]:
    """
    Randomly subsample `n` items such that all `id_key` values are unique.
    
    If items contain duplicate ids, the first occurrence (in input order) is kept.
    Sampling is deterministic given `seed`. The returned items preserve the
    original relative order.
    """
    seen = set()
    unique_items = []
    for item in items:
        key = item[id_key]
        if key in seen:
            continue
        seen.add(key)
        unique_items.append(item)
    
    if n >= len(unique_items):
        logger.info(
            f"Requested n={n} >= available unique {id_key}s ({len(unique_items)}); "
            f"using all {len(unique_items)} items."
        )
        return unique_items
    
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(unique_items)), n))
    sampled = [unique_items[i] for i in indices]
    logger.info(f"Subsampled {n} items with unique {id_key} (seed={seed})")
    return sampled


def extract_last_token_hidden_states(
    model,
    tokenizer,
    prompts: list[list[dict]],
    device: str,
) -> torch.Tensor:
    """
    Extract last-token hidden states from all layers for a batch of prompts.
    
    Args:
        model: The loaded HuggingFace model
        tokenizer: The tokenizer
        prompts: List of conversation prompts (each is a list of message dicts)
        device: Device to run inference on
    
    Returns:
        Tensor of shape (batch_size, n_layers, hidden_dim)
    """
    # Apply chat template to get input strings
    input_strs = tokenizer.apply_chat_template(
        prompts, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Tokenize with padding
    inputs = tokenizer(
        input_strs, 
        return_tensors="pt", 
        padding=True
    ).to(device)
    
    # Run inference with hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Get last token positions (accounting for padding)
    # attention_mask: 1 for real tokens, 0 for padding
    last_token_positions = inputs.attention_mask.sum(dim=1) - 1
    
    # Stack all hidden states: (n_layers, batch_size, seq_len, hidden_dim)
    all_layers = torch.stack(outputs.hidden_states, dim=0)
    
    # Extract last token for each example: (n_layers, batch_size, hidden_dim)
    batch_size = len(prompts)
    last_token_activations = all_layers[:, range(batch_size), last_token_positions, :]
    
    # Transpose to (batch_size, n_layers, hidden_dim)
    last_token_activations = last_token_activations.permute(1, 0, 2)
    
    return last_token_activations


def extract_last_token_hidden_states_with_response(
    model,
    tokenizer,
    conversations: list[list[dict]],
    device: str,
) -> torch.Tensor:
    """
    Extract last-token hidden states from all layers for a batch of full conversations
    (prompt + assistant response).
    
    Args:
        model: The loaded HuggingFace model
        tokenizer: The tokenizer
        conversations: List of conversations, each is a list of message dicts including
                       the assistant's response
        device: Device to run inference on
    
    Returns:
        Tensor of shape (batch_size, n_layers, hidden_dim)
    """
    # Apply chat template to get input strings (no generation prompt since response is included)
    input_strs = tokenizer.apply_chat_template(
        conversations, 
        tokenize=False, 
        add_generation_prompt=False
    )
    
    # Tokenize with padding
    inputs = tokenizer(
        input_strs, 
        return_tensors="pt", 
        padding=True,
        truncation=True,
        max_length=4500,
    ).to(device)
    
    # Run inference with hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Get last token positions (accounting for padding)
    last_token_positions = inputs.attention_mask.sum(dim=1) - 1
    
    # Stack all hidden states: (n_layers, batch_size, seq_len, hidden_dim)
    all_layers = torch.stack(outputs.hidden_states, dim=0)
    
    # Extract last token for each example: (n_layers, batch_size, hidden_dim)
    batch_size = len(conversations)
    last_token_activations = all_layers[:, range(batch_size), last_token_positions, :]
    
    # Transpose to (batch_size, n_layers, hidden_dim)
    last_token_activations = last_token_activations.permute(1, 0, 2)
    
    return last_token_activations


def run_expected_accuracy_mode(args, model, tokenizer, device, output_dir):
    """Run the expected_accuracy mode: probe on aggregated accuracy from ground_truth.jsonl."""
    # Load ground truth
    logger.info(f"Loading ground truth from {args.ground_truth_path}")
    ground_truth = load_ground_truth(args.ground_truth_path)
    logger.info(f"Loaded {len(ground_truth)} examples")
    
    if args.n is not None:
        ground_truth = subsample_unique(ground_truth, args.n, args.seed)
    
    # Build prompt lookup
    example_ids = {item["example_id"] for item in ground_truth}
    prompt_lookup = build_prompt_lookup(args.sampled_path, example_ids)
    
    # Verify we have all prompts
    missing = example_ids - set(prompt_lookup.keys())
    if missing:
        raise ValueError(f"Missing prompts for {len(missing)} examples: {list(missing)[:5]}...")
    
    logger.info(f"Found prompts for all {len(prompt_lookup)} examples")
    
    all_targets = []
    all_example_ids = []
    
    # Prepare data in order
    examples = []
    for item in ground_truth:
        examples.append({
            "example_id": item["example_id"],
            "prompt": prompt_lookup[item["example_id"]],
            "target": item["expected_accuracy"],
        })
    
    # Process batches — save hidden states to disk per batch to avoid OOM
    batch_idx = 0
    for i in tqdm(range(0, len(examples), args.batch_size), desc="Extracting hidden states", dynamic_ncols=True):
        batch = examples[i : i + args.batch_size]
        
        prompts = [ex["prompt"] for ex in batch]
        targets = [ex["target"] for ex in batch]
        example_ids_batch = [ex["example_id"] for ex in batch]
        
        hidden_states = extract_last_token_hidden_states(
            model, tokenizer, prompts, device
        )
        
        save_file(
            {"hidden_states": hidden_states.cpu().contiguous()},
            output_dir / f"_batch_{batch_idx:05d}.safetensors",
        )
        all_targets.extend(targets)
        all_example_ids.extend(example_ids_batch)
        batch_idx += 1
    
    return all_targets, all_example_ids


def run_sample_correctness_mode(args, model, tokenizer, device, output_dir):
    """Run the sample_correctness mode: probe on per-sample correctness from sampled.jsonl."""
    # Load one sample per example_id (selected by seed)
    samples = load_samples_for_correctness(args.sampled_path, args.seed)
    
    if args.n is not None:
        samples = subsample_unique(samples, args.n, args.seed)
    
    all_targets = []
    all_sample_ids = []
    
    # Process batches — save hidden states to disk per batch to avoid OOM
    batch_idx = 0
    for i in tqdm(range(0, len(samples), args.batch_size), desc="Extracting hidden states", dynamic_ncols=True):
        batch = samples[i : i + args.batch_size]
        
        conversations = []
        for sample in batch:
            conv = sample["prompt"] + [{"role": "assistant", "content": sample["response"]}]
            conversations.append(conv)
        
        targets = [sample["target"] for sample in batch]
        sample_ids_batch = [sample["sample_id"] for sample in batch]
        
        hidden_states = extract_last_token_hidden_states_with_response(
            model, tokenizer, conversations, device
        )
        
        save_file(
            {"hidden_states": hidden_states.cpu().contiguous()},
            output_dir / f"_batch_{batch_idx:05d}.safetensors",
        )
        all_targets.extend(targets)
        all_sample_ids.extend(sample_ids_batch)
        batch_idx += 1
    
    return all_targets, all_sample_ids


def run_verbalized_confidence_mode(args, model, tokenizer, device, output_dir):
    """Run the verbalized_confidence mode: probe on expected_accuracy from confidence_predictions.jsonl."""
    # Load verbalized confidence data
    data = load_verbalized_confidence_data(args.confidence_predictions_path)
    
    if args.n is not None:
        data = subsample_unique(data, args.n, args.seed)
    
    all_targets = []
    all_example_ids = []
    
    # Process batches — save hidden states to disk per batch to avoid OOM
    batch_idx = 0
    for i in tqdm(range(0, len(data), args.batch_size), desc="Extracting hidden states", dynamic_ncols=True):
        batch = data[i : i + args.batch_size]
        
        conversations = []
        for item in batch:
            conv = item["prompt"] + [{"role": "assistant", "content": item["response"]}]
            conversations.append(conv)
        
        targets = [item["target"] for item in batch]
        example_ids_batch = [item["example_id"] for item in batch]
        
        hidden_states = extract_last_token_hidden_states_with_response(
            model, tokenizer, conversations, device
        )
        
        save_file(
            {"hidden_states": hidden_states.cpu().contiguous()},
            output_dir / f"_batch_{batch_idx:05d}.safetensors",
        )
        all_targets.extend(targets)
        all_example_ids.extend(example_ids_batch)
        batch_idx += 1
    
    return all_targets, all_example_ids


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Mode: {args.mode}")
    
    # Load model and tokenizer
    logger.info(f"Loading model {args.model_id}...")
    device = args.device if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        device_map=device,
        torch_dtype="auto",
        cache_dir=args.cache_dir,
    )
    model.eval()
    
    logger.info(f"Model loaded on {device}")
    
    # Run the appropriate mode (hidden states saved to per-batch files on disk)
    if args.mode == "expected_accuracy":
        all_targets, all_ids = run_expected_accuracy_mode(
            args, model, tokenizer, device, output_dir
        )
    elif args.mode == "sample_correctness":
        all_targets, all_ids = run_sample_correctness_mode(
            args, model, tokenizer, device, output_dir
        )
    else:  # verbalized_confidence
        all_targets, all_ids = run_verbalized_confidence_mode(
            args, model, tokenizer, device, output_dir
        )
    
    # Load per-batch hidden states from disk, concatenate, and save final file
    batch_files = sorted(output_dir.glob("_batch_*.safetensors"))
    logger.info(f"Merging {len(batch_files)} batch files...")
    
    chunks = [load_file(p)["hidden_states"] for p in batch_files]
    hidden_states_tensor = torch.cat(chunks, dim=0)
    del chunks
    
    targets_tensor = torch.tensor(all_targets, dtype=torch.float32)
    
    logger.info(f"Hidden states shape: {hidden_states_tensor.shape}")
    logger.info(f"Targets shape: {targets_tensor.shape}")
    
    # Save final safetensors
    logger.info(f"Saving to {output_dir}")
    
    save_file(
        {"hidden_states": hidden_states_tensor},
        output_dir / "hidden_states.safetensors",
    )
    save_file(
        {"targets": targets_tensor},
        output_dir / "targets.safetensors",
    )
    
    # Clean up batch files
    for p in batch_files:
        p.unlink()
    logger.info(f"Deleted {len(batch_files)} temporary batch files")
    
    # Save metadata
    metadata = {
        "model_id": args.model_id,
        "mode": args.mode,
        "n_examples": len(all_ids),
        "n_layers": hidden_states_tensor.shape[1],
        "hidden_dim": hidden_states_tensor.shape[2],
        "example_ids": all_ids,  # For sample_correctness, these are sample_ids
    }
    if args.mode == "sample_correctness":
        metadata["seed"] = args.seed
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Done!")
    logger.info(f"  - hidden_states.safetensors: {hidden_states_tensor.shape}")
    logger.info(f"  - targets.safetensors: {targets_tensor.shape}")
    logger.info(f"  - metadata.json: {len(all_ids)} example IDs")


if __name__ == "__main__":
    main()
