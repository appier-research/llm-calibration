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

## Ground-Truth Expected Accuracy Comparisons

Correlations and MSEs align examples by `example_id`. Mean expected accuracy uses all rows present in each ground-truth file.

### Olmo-3-7B-Instruct: Spearman r vs Default Temperature 0.6

| Comparison | TriviaQA validation | GSM8K test | MMLU test |
|---|---:|---:|---:|
| 0.6 <--> 0.3 | 0.9721 | 0.8654 | 0.9763 |
| 0.6 <--> 0.15 | 0.9522 | 0.8443 | 0.9654 |

### Olmo-3-7B-Instruct: MSE vs Default Temperature 0.6

| Comparison | TriviaQA validation | GSM8K test | MMLU test |
|---|---:|---:|---:|
| 0.6 <--> 0.3 | 0.0048 | 0.0012 | 0.0034 |
| 0.6 <--> 0.15 | 0.0106 | 0.0026 | 0.0063 |

### Qwen3-8B-non-thinking: Spearman r vs Default Temperature 0.7

| Comparison | TriviaQA validation | GSM8K test | MMLU test |
|---|---:|---:|---:|
| 0.7 <--> 0.35 | 0.9642 | 0.8680 | 0.9596 |
| 0.7 <--> 0.175 | 0.9348 | 0.8129 | 0.9404 |

### Qwen3-8B-non-thinking: MSE vs Default Temperature 0.7

| Comparison | TriviaQA validation | GSM8K test | MMLU test |
|---|---:|---:|---:|
| 0.7 <--> 0.35 | 0.0052 | 0.0016 | 0.0026 |
| 0.7 <--> 0.175 | 0.0106 | 0.0033 | 0.0059 |

### gpt-oss-20b: Spearman r vs Default Temperature 1.0

| Comparison | TriviaQA validation | GSM8K test | MMLU test |
|---|---:|---:|---:|
| 1.0 <--> 0.5 | 0.9449 | 0.6734 | 0.8784 |
| 1.0 <--> 0.25 | 0.9225 | 0.6414 | 0.8530 |

### gpt-oss-20b: MSE vs Default Temperature 1.0

| Comparison | TriviaQA validation | GSM8K test | MMLU test |
|---|---:|---:|---:|
| 1.0 <--> 0.5 | 0.0120 | 0.0011 | 0.0035 |
| 1.0 <--> 0.25 | 0.0208 | 0.0027 | 0.0063 |

### Olmo-3-7B-Instruct: Mean Expected Accuracy by Temperature

| Temperature | TriviaQA validation | GSM8K test | MMLU test |
|---|---:|---:|---:|
| 0.6 | 0.4884 | 0.9331 | 0.7006 |
| 0.3 | 0.4933 | 0.9340 | 0.6989 |
| 0.15 | 0.4957 | 0.9348 | 0.6989 |

### Qwen3-8B-non-thinking: Mean Expected Accuracy by Temperature

| Temperature | TriviaQA validation | GSM8K test | MMLU test |
|---|---:|---:|---:|
| 0.7 | 0.6040 | 0.9356 | 0.7940 |
| 0.35 | 0.6122 | 0.9355 | 0.7979 |
| 0.175 | 0.6140 | 0.9361 | 0.7976 |

### gpt-oss-20b: Mean Expected Accuracy by Temperature

| Temperature | TriviaQA validation | GSM8K test | MMLU test |
|---|---:|---:|---:|
| 1.0 | 0.6507 | 0.9576 | 0.8444 |
| 0.5 | 0.6625 | 0.9585 | 0.8429 |
| 0.25 | 0.6565 | 0.9566 | 0.8384 |
