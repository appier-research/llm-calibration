# Improved Verbalized Confidence and P(True)
**Caption**: For verbalized confidence and P(True), "train on TriviaQA / GSM8K / MATH" means performing isotonic regression on the training set of TriviaQA / GSM8K / MATH.

## Olmo-3-7B-Instruct


| Method                                    | TriviaQA | SimpleQA | GSM8K  | MATH   | AIME25 | MMLU   | GPQA   |
| ----------------------------------------- | -------- | -------- | ------ | ------ | ------ | ------ | ------ |
| Uniform random baseline                   | 0.2745   | 0.3133   | 0.3119 | 0.2940 | 0.2462 | 0.2565 | 0.2125 |
| Verbalized confidence (training-free)     | 0.2624   | 0.2676   | 0.0462 | 0.0557 | 0.2002 | 0.1561 | 0.2742 |
| Verbalized confidence (few-shot-3)        | 0.2802   | 0.3707   | 0.0423 | 0.0637 | 0.2437 | 0.1677 | 0.2809 |
| Verbalized confidence (train on TriviaQA) | 0.1681   | 0.1267   | 0.1662 | 0.1664 | 0.1454 | 0.1522 | 0.1259 |
| Verbalized confidence (train on GSM8K)    | 0.3173   | 0.6296   | 0.0405 | 0.0553 | 0.3317 | 0.1801 | 0.3487 |
| Verbalized confidence (train on MATH)     | 0.2819   | 0.5290   | 0.0407 | 0.0507 | 0.2586 | 0.1609 | 0.2979 |
| P(True) (training-free)                   | 0.1933   | 0.0419   | 0.1282 | 0.1400 | 0.1854 | 0.2164 | 0.1553 |
| P(True) (train on TriviaQA)               | 0.1668   | 0.1376   | 0.1305 | 0.1378 | 0.1624 | 0.1513 | 0.1291 |
| P(True) (train on GSM8K)                  | 0.3020   | 0.5553   | 0.0394 | 0.0633 | 0.4038 | 0.1668 | 0.3391 |
| P(True) (train on MATH)                   | 0.2970   | 0.6918   | 0.0400 | 0.0584 | 0.3540 | 0.1576 | 0.3104 |
| Probe (train on TriviaQA)                 | 0.1113   | 0.0386   | 0.1180 | 0.1273 | 0.1496 | 0.1300 | 0.1242 |
| Probe (train on GSM8K)                    | 0.2648   | 0.5465   | 0.0370 | 0.0545 | 0.2482 | 0.1200 | 0.1628 |
| Probe (train on MATH)                     | 0.2550   | 0.4846   | 0.0388 | 0.0394 | 0.1411 | 0.1255 | 0.1295 |


## Qwen3-8B


| Method                                    | TriviaQA | SimpleQA | GSM8K  | MATH   | AIME25 | MMLU   | GPQA   |
| ----------------------------------------- | -------- | -------- | ------ | ------ | ------ | ------ | ------ |
| Uniform random baseline                   | 0.2865   | 0.3109   | 0.3144 | 0.2781 | 0.2800 | 0.2868 | 0.2113 |
| Verbalized confidence (training-free)     | 0.2431   | 0.4736   | 0.0461 | 0.0962 | 0.4443 | 0.1293 | 0.2773 |
| Verbalized confidence (few-shot-3)        | 0.2724   | 0.3694   | 0.0429 | 0.0729 | 0.3357 | 0.1343 | 0.2554 |
| Verbalized confidence (train on TriviaQA) | 0.1700   | 0.2552   | 0.1374 | 0.1431 | 0.2113 | 0.1554 | 0.1409 |
| Verbalized confidence (train on GSM8K)    | 0.2652   | 0.6715   | 0.0412 | 0.0973 | 0.5966 | 0.1374 | 0.3140 |
| Verbalized confidence (train on MATH)     | 0.2231   | 0.5593   | 0.0452 | 0.0822 | 0.4522 | 0.1145 | 0.2289 |
| P(True) (training-free)                   | 0.2970   | 0.6072   | 0.0482 | 0.1126 | 0.4957 | 0.1597 | 0.3448 |
| P(True) (train on TriviaQA)               | 0.1505   | 0.2068   | 0.1049 | 0.1190 | 0.1715 | 0.1328 | 0.1129 |
| P(True) (train on GSM8K)                  | 0.2402   | 0.6434   | 0.0388 | 0.0868 | 0.5243 | 0.1216 | 0.2589 |
| P(True) (train on MATH)                   | 0.1837   | 0.3713   | 0.0468 | 0.0744 | 0.2614 | 0.1057 | 0.1623 |
| Probe (train on TriviaQA)                 | 0.1079   | 0.0638   | 0.3177 | 0.3219 | 0.1286 | 0.2006 | 0.1556 |
| Probe (train on GSM8K)                    | 0.1885   | 0.4451   | 0.0368 | 0.0715 | 0.0740 | 0.1176 | 0.1556 |
| Probe (train on MATH)                     | 0.2977   | 0.8297   | 0.0408 | 0.0475 | 0.0831 | 0.1163 | 0.1811 |


## gpt-oss-20b


| Method                                    | TriviaQA | SimpleQA | GSM8K  | MATH   | AIME25 | MMLU   | GPQA   |
| ----------------------------------------- | -------- | -------- | ------ | ------ | ------ | ------ | ------ |
| Uniform random baseline                   | 0.2639   | 0.3010   | 0.3195 | 0.3063 | 0.2369 | 0.3018 | 0.2388 |
| Verbalized confidence (training-free)     | 0.1266   | 0.1957   | 0.0268 | 0.0275 | 0.0460 | 0.0559 | 0.1174 |
| Verbalized confidence (few-shot-3)        | 0.1217   | 0.1583   | 0.0237 | 0.0200 | 0.0584 | 0.0556 | 0.1034 |
| Verbalized confidence (train on TriviaQA) | 0.0825   | 0.0776   | 0.0476 | 0.0486 | 0.0762 | 0.0925 | 0.1184 |
| Verbalized confidence (train on GSM8K)    | 0.1310   | 0.3003   | 0.0263 | 0.0272 | 0.0579 | 0.0547 | 0.1052 |
| Verbalized confidence (train on MATH)     | 0.1128   | 0.1587   | 0.0277 | 0.0262 | 0.0488 | 0.0564 | 0.0931 |
| P(True) (training-free)                   | 0.2101   | 0.6151   | 0.0306 | 0.0321 | 0.1092 | 0.0817 | 0.2082 |
| P(True) (train on TriviaQA)               | 0.1284   | 0.2699   | 0.0968 | 0.0875 | 0.1049 | 0.1008 | 0.1248 |
| P(True) (train on GSM8K)                  | 0.1991   | 0.6933   | 0.0268 | 0.0333 | 0.1064 | 0.0650 | 0.1716 |
| P(True) (train on MATH)                   | 0.1787   | 0.5542   | 0.0279 | 0.0299 | 0.0988 | 0.0681 | 0.1596 |
| Probe (train on TriviaQA)                 | 0.0845   | 0.0600   | 0.0780 | 0.1593 | 0.1457 | 0.0977 | 0.1533 |
| Probe (train on GSM8K)                    | 0.1756   | 0.7048   | 0.0289 | 0.0485 | 0.1213 | 0.0686 | 0.2010 |
| Probe (train on MATH)                     | 0.1577   | 0.5871   | 0.0332 | 0.0267 | 0.1644 | 0.0922 | 0.1363 |
