# Temperature Sweep Results

## Qwen3-8B-non-thinking, Temperature 0.35

| Method | TriviaQA validation | GSM8K test | MMLU test |
|---|---:|---:|---:|
| Uniform random | 0.3015 | 0.3133 | 0.2909 |
| Verbalized confidence | 0.2573 | 0.0531 | 0.1210 |
| Verbalized confidence + isotonic | 0.1761 | 0.0451 | 0.1078 |
| P(True) | 0.3278 | 0.0515 | 0.1557 |
| P(True) + isotonic | <u>0.1540</u> | <u>0.0428</u> | <u>0.1051</u> |
| Linear probe | **0.1273** | **0.0412** | **0.0993** |

## Qwen3-8B-non-thinking, Temperature 0.175

| Method | TriviaQA validation | GSM8K test | MMLU test |
|---|---:|---:|---:|
| Uniform random | 0.3085 | 0.3201 | 0.2930 |
| Verbalized confidence | 0.2607 | 0.0503 | 0.1260 |
| Verbalized confidence + isotonic | 0.1808 | 0.0460 | 0.1106 |
| P(True) | 0.3326 | 0.0531 | 0.1585 |
| P(True) + isotonic | <u>0.1586</u> | <u>0.0447</u> | <u>0.1078</u> |
| Linear probe | **0.1350** | **0.0429** | **0.1044** |
