# General configs for the language model
model_id="Qwen/Qwen3-8B"
model_name="Qwen3-8B-non-thinking"
base_url="http://TBD/v1"  # TODO: set your own base URL here
llm_judge_model="openai/gpt-oss-20b"
llm_judge_base_url="http://TBD/v1"  # TODO: set your own base URL here
max_tokens=8192
temperature=0.7
top_p=0.8
max_concurrent=500

# Dataset-specific configs
k=100
# For evaluation
# TriviaQA (validation)
# first, run inference with string matching
uv run python scripts/construct_eval_datasets.py \
    experiment=default \
    experiment.sampling.k=${k} \
    experiment.dataset.max_examples=1000 \
    experiment.async_.max_concurrent=${max_concurrent} \
    dataset=triviaqa \
    dataset.split=validation \
    dataset.verification_mode="string_match" \
    model=openai_compat \
    model.model_id=${model_id} \
    model.base_url=${base_url} \
    hydra.job_logging.root.level=INFO \
    inference.max_tokens=${max_tokens} \
    inference.temperature=${temperature} \
    inference.top_p=${top_p} \
    hydra.run.dir="outputs/triviaqa-validation__${model_name}"
# then, run evaluation with LLM-as-a-judge
uv run python scripts/construct_eval_datasets.py \
    experiment=default \
    experiment.sampling.k=${k} \
    experiment.dataset.max_examples=1000 \
    experiment.async_.max_concurrent=1000 \
    dataset=triviaqa \
    dataset.split=validation \
    dataset.verification_mode="llm_judge" \
    dataset.llm_judge_model=${llm_judge_model} \
    dataset.llm_judge_base_url=${llm_judge_base_url} \
    model=openai_compat \
    model.model_id=${model_id} \
    model.base_url=${base_url} \
    hydra.job_logging.root.level=INFO \
    inference.max_tokens=${max_tokens} \
    inference.temperature=${temperature} \
    inference.top_p=${top_p} \
    hydra.run.dir="outputs/triviaqa-validation__${model_name}"

# SimpleQA verified
# first, run inference with string matching
uv run python scripts/construct_eval_datasets.py \
    experiment=default \
    experiment.sampling.k=${k} \
    experiment.async_.max_concurrent=${max_concurrent} \
    dataset=simpleqa-verified \
    dataset.split=eval \
    dataset.verification_mode="string_match" \
    model=openai_compat \
    model.model_id=${model_id} \
    model.base_url=${base_url} \
    hydra.job_logging.root.level=INFO \
    inference.max_tokens=${max_tokens} \
    inference.temperature=${temperature} \
    inference.top_p=${top_p} \
    hydra.run.dir="outputs/simpleqa-verified__${model_name}"
# then, run evaluation with LLM-as-a-judge
uv run python scripts/construct_eval_datasets.py \
    experiment=default \
    experiment.sampling.k=${k} \
    experiment.async_.max_concurrent=1000 \
    dataset=simpleqa-verified \
    dataset.split=eval \
    dataset.verification_mode="llm_judge" \
    dataset.llm_judge_model=${llm_judge_model} \
    dataset.llm_judge_base_url=${llm_judge_base_url} \
    model=openai_compat \
    model.model_id=${model_id} \
    model.base_url=${base_url} \
    hydra.job_logging.root.level=INFO \
    inference.max_tokens=${max_tokens} \
    inference.temperature=${temperature} \
    inference.top_p=${top_p} \
    hydra.run.dir="outputs/simpleqa-verified__${model_name}"

# GSM8K (test)
uv run python scripts/construct_eval_datasets.py \
    experiment=default \
    experiment.sampling.k=${k} \
    experiment.async_.max_concurrent=${max_concurrent} \
    dataset="gsm8k" \
    dataset.split="test" \
    model="openai_compat" \
    model.model_id=${model_id} \
    model.base_url=${base_url} \
    hydra.job_logging.root.level="INFO" \
    inference.max_tokens=${max_tokens} \
    inference.temperature=${temperature} \
    inference.top_p=${top_p} \
    hydra.run.dir="outputs/gsm8k-test__${model_name}"

# MATH-500
# first, run inference with math-verify
uv run python scripts/construct_eval_datasets.py \
    experiment=default \
    experiment.sampling.k=${k} \
    experiment.async_.max_concurrent=${max_concurrent} \
    dataset=math-500 \
    dataset.split=test \
    dataset.verification_mode="math_verify" \
    model=openai_compat \
    model.model_id=${model_id} \
    model.base_url=${base_url} \
    hydra.job_logging.root.level=INFO \
    inference.max_tokens=${max_tokens} \
    inference.temperature=${temperature} \
    inference.top_p=${top_p} \
    hydra.run.dir="outputs/math-500__${model_name}"
# then, run evaluation with LLM-as-a-judge
uv run python scripts/construct_eval_datasets.py \
    experiment=default \
    experiment.sampling.k=${k} \
    experiment.async_.max_concurrent=1000 \
    dataset=math-500 \
    dataset.split=test \
    dataset.verification_mode="llm_judge" \
    dataset.llm_judge_model=${llm_judge_model} \
    dataset.llm_judge_base_url=${llm_judge_base_url} \
    model=openai_compat \
    model.model_id=${model_id} \
    model.base_url=${base_url} \
    hydra.job_logging.root.level=INFO \
    inference.max_tokens=${max_tokens} \
    inference.temperature=${temperature} \
    inference.top_p=${top_p} \
    hydra.run.dir="outputs/math-500__${model_name}"

# AIME25
uv run python scripts/construct_eval_datasets.py \
    experiment=default \
    experiment.sampling.k=${k} \
    experiment.async_.max_concurrent=${max_concurrent} \
    dataset=aime25 \
    dataset.split=test \
    model=openai_compat \
    model.model_id=${model_id} \
    model.base_url=${base_url} \
    hydra.job_logging.root.level=INFO \
    inference.max_tokens=${max_tokens} \
    inference.temperature=${temperature} \
    inference.top_p=${top_p} \
    hydra.run.dir="outputs/aime25-test__${model_name}"

# MMLU
uv run python scripts/construct_eval_datasets.py \
    experiment=default \
    experiment.sampling.k=${k} \
    experiment.async_.max_concurrent=${max_concurrent} \
    dataset=mmlu \
    dataset.split=validation \
    model=openai_compat \
    model.model_id=${model_id} \
    model.base_url=${base_url} \
    hydra.job_logging.root.level=INFO \
    inference.max_tokens=${max_tokens} \
    inference.temperature=${temperature} \
    inference.top_p=${top_p} \
    hydra.run.dir="outputs/mmlu-validation__${model_name}"

# GPQA
uv run python scripts/construct_eval_datasets.py \
    experiment=default \
    experiment.sampling.k=${k} \
    experiment.async_.max_concurrent=${max_concurrent} \
    dataset=gpqa \
    dataset.split=train \
    dataset.hf_name=gpqa_diamond \
    model=openai_compat \
    model.model_id=${model_id} \
    model.base_url=${base_url} \
    hydra.job_logging.root.level=INFO \
    inference.max_tokens=${max_tokens} \
    inference.temperature=${temperature} \
    inference.top_p=${top_p} \
    hydra.run.dir="outputs/gpqa-diamond__${model_name}"

# For training
k=10
# TriviaQA (train)
# first, run inference with string matching
uv run python scripts/construct_eval_datasets.py \
    experiment=default \
    experiment.sampling.k=${k} \
    experiment.async_.max_concurrent=${max_concurrent} \
    dataset=triviaqa \
    dataset.split=train \
    dataset.verification_mode="string_match" \
    model=openai_compat \
    model.model_id=${model_id} \
    model.base_url=${base_url} \
    hydra.job_logging.root.level=INFO \
    inference.max_tokens=${max_tokens} \
    inference.temperature=${temperature} \
    inference.top_p=${top_p} \
    hydra.run.dir="outputs/triviaqa-train__${model_name}"
# then, run evaluation with LLM-as-a-judge
uv run python scripts/construct_eval_datasets.py \
    experiment=default \
    experiment.sampling.k=${k} \
    experiment.async_.max_concurrent=1000 \
    dataset=triviaqa \
    dataset.split=train \
    dataset.verification_mode="llm_judge" \
    dataset.llm_judge_model=${llm_judge_model} \
    dataset.llm_judge_base_url=${llm_judge_base_url} \
    model=openai_compat \
    model.model_id=${model_id} \
    model.base_url=${base_url} \
    hydra.job_logging.root.level=INFO \
    inference.max_tokens=${max_tokens} \
    inference.temperature=${temperature} \
    inference.top_p=${top_p} \
    hydra.run.dir="outputs/triviaqa-train__${model_name}"

# GSM8K (train)
uv run python scripts/construct_eval_datasets.py \
    experiment=default \
    experiment.sampling.k=${k} \
    experiment.async_.max_concurrent=${max_concurrent} \
    dataset="gsm8k" \
    dataset.split="train" \
    model="openai_compat" \
    model.model_id=${model_id} \
    model.base_url=${base_url} \
    hydra.job_logging.root.level="INFO" \
    inference.max_tokens=${max_tokens} \
    inference.temperature=${temperature} \
    inference.top_p=${top_p} \
    hydra.run.dir="outputs/gsm8k-train__${model_name}"

# MATH-train
# first, run inference with math-verify
uv run python scripts/construct_eval_datasets.py \
    experiment=default \
    experiment.sampling.k=${k} \
    experiment.async_.max_concurrent=${max_concurrent} \
    dataset=math-train \
    dataset.split=train \
    dataset.verification_mode="math_verify" \
    model=openai_compat \
    model.model_id=${model_id} \
    model.base_url=${base_url} \
    hydra.job_logging.root.level=INFO \
    inference.max_tokens=${max_tokens} \
    inference.temperature=${temperature} \
    inference.top_p=${top_p} \
    hydra.run.dir="outputs/math-train__${model_name}"
# then, run evaluation with LLM-as-a-judge
uv run python scripts/construct_eval_datasets.py \
    experiment=default \
    experiment.sampling.k=${k} \
    experiment.async_.max_concurrent=1000 \
    dataset=math-train \
    dataset.split=train \
    dataset.verification_mode="llm_judge" \
    dataset.llm_judge_model=${llm_judge_model} \
    dataset.llm_judge_base_url=${llm_judge_base_url} \
    model=openai_compat \
    model.model_id=${model_id} \
    model.base_url=${base_url} \
    hydra.job_logging.root.level=INFO \
    inference.max_tokens=${max_tokens} \
    inference.temperature=${temperature} \
    inference.top_p=${top_p} \
    hydra.run.dir="outputs/math-train__${model_name}"

# MATH-valid
# first, run inference with math-verify
uv run python scripts/construct_eval_datasets.py \
    experiment=default \
    experiment.sampling.k=100 \
    experiment.async_.max_concurrent=${max_concurrent} \
    dataset=math-valid \
    dataset.split=validation \
    dataset.verification_mode="math_verify" \
    model=openai_compat \
    model.model_id=${model_id} \
    model.base_url=${base_url} \
    hydra.job_logging.root.level=INFO \
    inference.max_tokens=${max_tokens} \
    inference.temperature=${temperature} \
    inference.top_p=${top_p} \
    hydra.run.dir="outputs/math-valid__${model_name}"
# then, run evaluation with LLM-as-a-judge
uv run python scripts/construct_eval_datasets.py \
    experiment=default \
    experiment.sampling.k=100 \
    experiment.async_.max_concurrent=1000 \
    dataset=math-valid \
    dataset.split=validation \
    dataset.verification_mode="llm_judge" \
    dataset.llm_judge_model=${llm_judge_model} \
    dataset.llm_judge_base_url=${llm_judge_base_url} \
    model=openai_compat \
    model.model_id=${model_id} \
    model.base_url=${base_url} \
    hydra.job_logging.root.level=INFO \
    inference.max_tokens=${max_tokens} \
    inference.temperature=${temperature} \
    inference.top_p=${top_p} \
    hydra.run.dir="outputs/math-valid__${model_name}"