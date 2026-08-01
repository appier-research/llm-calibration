# Temperature Sweep Results

## Olmo-3-7B-Instruct, Temperature 0.3

| Method | TriviaQA validation | GSM8K test | MMLU test |
|---|---:|---:|---:|
| Uniform random | 0.2748 | 0.3143 | 0.2594 |
| Verbalized confidence | 0.2878 | 0.0483 | 0.1489 |
| Verbalized confidence + isotonic | 0.1684 | **0.0382** | <u>0.1276</u> |
| P(True) | 0.1794 | 0.1306 | 0.2272 |
| P(True) + isotonic | <u>0.1663</u> | 0.0410 | 0.1283 |
| Linear probe | **0.1186** | <u>0.0383</u> | **0.1036** |

## Olmo-3-7B-Instruct, Temperature 0.15

| Method | TriviaQA validation | GSM8K test | MMLU test |
|---|---:|---:|---:|
| Uniform random | 0.2743 | 0.3205 | 0.2594 |
| Verbalized confidence | 0.2950 | 0.0480 | 0.1576 |
| Verbalized confidence + isotonic | 0.1764 | **0.0389** | 0.1346 |
| P(True) | 0.1858 | 0.1317 | 0.2306 |
| P(True) + isotonic | <u>0.1730</u> | 0.0416 | <u>0.1336</u> |
| Linear probe | **0.1318** | <u>0.0390</u> | **0.1114** |

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

## gpt-oss-20b, Temperature 0.5

| Method | TriviaQA validation | GSM8K test | MMLU test |
|---|---:|---:|---:|
| Uniform random | 0.2720 | 0.3252 | 0.2917 |
| Verbalized confidence | 0.1634 | <u>0.0261</u> | <u>0.0901</u> |
| Verbalized confidence + isotonic | **0.0816** | **0.0259** | **0.0801** |
| P(True) | 0.2653 | 0.0294 | 0.1249 |
| P(True) + isotonic | 0.1493 | 0.0271 | 0.0963 |
| Linear probe | <u>0.1143</u> | 0.0331 | 0.0946 |

## gpt-oss-20b, Temperature 0.25

| Method | TriviaQA validation | GSM8K test | MMLU test |
|---|---:|---:|---:|
| Uniform random | 0.2887 | 0.3244 | 0.3064 |
| Verbalized confidence | 0.1860 | <u>0.0283</u> | <u>0.0945</u> |
| Verbalized confidence + isotonic | **0.0984** | **0.0270** | **0.0847** |
| P(True) | 0.2713 | 0.0332 | 0.1278 |
| P(True) + isotonic | 0.1611 | 0.0295 | 0.0981 |
| Linear probe | <u>0.1208</u> | 0.0365 | 0.1013 |
