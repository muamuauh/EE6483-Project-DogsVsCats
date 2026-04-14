# Validation Error Analysis

- Accuracy: `0.9902`
- Correct samples: `4951`
- Incorrect samples: `49`
- Total validation samples: `5000`

## High-confidence incorrect samples

| image | true | predicted | confidence |
|---|---|---|---|
| `cat.11565.jpg` | `cat` | `dog` | `1.0000` |
| `cat.7920.jpg` | `cat` | `dog` | `0.9996` |
| `cat.11149.jpg` | `cat` | `dog` | `0.9973` |
| `dog.9109.jpg` | `dog` | `cat` | `0.9853` |
| `cat.6655.jpg` | `cat` | `dog` | `0.9779` |
| `cat.9882.jpg` | `cat` | `dog` | `0.9727` |

## High-confidence correct samples

| image | true | predicted | confidence |
|---|---|---|---|
| `cat.10280.jpg` | `cat` | `cat` | `1.0000` |
| `cat.10525.jpg` | `cat` | `cat` | `1.0000` |
| `cat.1064.jpg` | `cat` | `cat` | `1.0000` |
| `cat.10656.jpg` | `cat` | `cat` | `1.0000` |
| `cat.1093.jpg` | `cat` | `cat` | `1.0000` |
| `cat.11204.jpg` | `cat` | `cat` | `1.0000` |

## Low-confidence correct samples

| image | true | predicted | confidence |
|---|---|---|---|
| `cat.2159.jpg` | `cat` | `cat` | `0.5000` |
| `dog.6199.jpg` | `dog` | `dog` | `0.5040` |
| `cat.8198.jpg` | `cat` | `cat` | `0.5223` |
| `dog.12353.jpg` | `dog` | `dog` | `0.5287` |
| `cat.7122.jpg` | `cat` | `cat` | `0.5312` |
| `cat.10441.jpg` | `cat` | `cat` | `0.5665` |
