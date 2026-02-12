# Run PASS-K simulation at the dataset level

cd applications/passk_simulation

uv run python src/run_simulation.py \
    --config config/passk_simulation_aime.yaml

uv run python src/run_simulation.py \
    --config config/passk_simulation_math500.yaml