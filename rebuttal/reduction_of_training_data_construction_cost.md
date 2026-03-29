## Brier Score (↓) on TriviaQA ($k_{eval} = 100$)

| Method / Model | Olmo-3-7B | Qwen3-8B | gpt-oss-20b |
|---|---|---|---|
| Uniform random baseline | 0.2745 | 0.2865 | 0.2639 |
| Verbalized confidence | 0.2624 | 0.2431 | 0.1266 |
| P(True) | 0.1933 | 0.2970 | 0.2101 |
| Probe ($k_{train} = 100$) | 0.1113 | 0.1079 | 0.0845 |
| Probe ($k_{train} = 10$) | 0.1117 | 0.1083 | 0.0859 |

## Brier Score (↓) on GSM8K ($k_{eval} = 100$)

| Method / Model | Olmo-3-7B | Qwen3-8B | gpt-oss-20b |
|---|---|---|---|
| Uniform random baseline | 0.3119 | 0.3144 | 0.3195 |
| Verbalized confidence | 0.0462 | 0.0461 | 0.0268 |
| P(True) | 0.1282 | 0.0482 | 0.0306 |
| Probe ($k_{train} = 100$) | 0.0370 | 0.0368 | 0.0289 |
| Probe ($k_{train} = 10$) | 0.0370 | 0.0371 | 0.0290 |
