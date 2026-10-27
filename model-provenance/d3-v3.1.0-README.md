---
pipeline_tag: zero-shot-classification
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: transformers
tags:
- zero-shot-classification
- decision-model
- classification
- system-one
- multimodal
- vision
- video
- safetensors
---

![d3: Decision 3.0](assets/banner.png)

# d3

**d3** is the 27B multimodal foundation decision model of Decision 3.0, the decision models of [vLLM Semantic Router](https://github.com/vllm-project/semantic-router). Give it an input (text or JSON, optionally with images and videos) and the questions you need answered: pick one of several options, say yes or no, or rate on a scale. It answers them all in one call and returns a probability for every answer, without generating text.

| | |
| --- | --- |
| **Parameters** | 26.09B, including the 0.46B vision encoder |
| **Inputs** | Text or JSON, plus images and videos (several per request) |
| **Decision types** | Choice · Yes / No · Score |
| **License** | Apache-2.0 |

## Highlights

- **Jev Decision Index 0.3, public suite: 65.45**, measured with the official 0.3 kit on the released weights: all 140,178 public requests answered, none unsupported.
- **+8.5 on the public suite over Decision 2.0** (its 27B model: 56.97 on the board), ahead in all five areas.
- **Reads images:** multiple images per request (PNG, JPEG or WebP), given as paths, URLs, PIL images or base64 data URLs; every question of the request sees all of them.
- **Reads videos:** multiple videos per request (MP4, WebM, MOV or MKV), given as paths, URLs, base64 data URLs or frame arrays, read at 2 frames per second; **Perception Test (multiple-choice video QA): 77.4** (internal evaluation).
- **Speed:** text requests take a median of 55 ms, requests with an image a median of 273 ms, and requests with a 10-second video a median of 553 ms, on one AMD Instinct MI325X, one request at a time.
- **Many questions, one call:** Choice, Yes / No and Score questions about the same input are answered together, each from its own forward pass over the input, with a probability for every option.

## Quickstart

```bash
pip install "transformers==5.17.0" torch torchvision pillow opencv-python-headless safetensors accelerate
pip install flash-linear-attention  # optional: fast GPU kernels for the linear-attention layers
```

```python
import json

from huggingface_hub import hf_hub_download
from transformers import AutoModel

model = AutoModel.from_pretrained("vllm-sr/d3", trust_remote_code=True)

# Text
result = model.system_one(
    state="The order arrived damaged yesterday. The customer has a receipt and asks for a replacement today.",
    questions={
        "route": {
            "type": "choice",
            "instructions": "Which team should handle this request?",
            "criteria": {
                "returns": "Refunds, replacements and damaged deliveries",
                "billing": "Payments, invoices and charges",
                "technical": "Product setup and faults"
            }
        },
        "receipt": {
            "type": "noul",
            "instructions": "Does the customer have a receipt?"
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this request?",
            "criteria": [
                "Routine",
                "Soon",
                "Today"
            ]
        }
    },
)
print(json.dumps(result["answers"], indent=2))

# Text and an image (or several)
receipt = hf_hub_download("vllm-sr/d3", "assets/example-receipt.png")
result = model.system_one(
    state="The customer says the blender arrived cracked and attached the receipt.",
    images=[receipt],  # local paths, http(s) URLs, PIL images or base64 data URLs
    questions={
        "route": {
            "type": "choice",
            "instructions": "Which team should handle this request?",
            "criteria": {
                "returns": "Refunds, replacements and damaged deliveries",
                "billing": "Payments, invoices and charges",
                "technical": "Product setup and faults"
            }
        },
        "on_receipt": {
            "type": "noul",
            "instructions": "Does the receipt list the blender?"
        },
        "payment": {
            "type": "choice",
            "instructions": "How was the order paid?",
            "criteria": {
                "card": None,
                "cash": None,
                "gift card": None
            }
        }
    },
)
print(json.dumps(result["answers"], indent=2))

# Text and a video (or several)
clip = hf_hub_download("vllm-sr/d3", "assets/example-video.mp4")
result = model.system_one(
    state="A short clip from a test camera.",
    videos=[clip],  # local paths, http(s) URLs, base64 data URLs or frame arrays
    questions={
        "direction": {
            "type": "choice",
            "instructions": "Which way does the square move?",
            "criteria": {
                "right": "From left to right",
                "left": "From right to left",
                "still": "It does not move"
            }
        },
        "color_change": {
            "type": "noul",
            "instructions": "Does the square change color?"
        }
    },
)
print(json.dumps(result["answers"], indent=2))

# Or as a pipeline:
# transformers.pipeline("decision", model="vllm-sr/d3", trust_remote_code=True)(state=..., questions=..., images=..., videos=...)
```

Images go before the text of the request, each read at up to 1.6 megapixels; videos follow the images, read at 2 frames per second (at most 32 frames spread over the whole video, each at up to 0.2 megapixels). Every question of the request sees all of them.

## Evaluation

### Text: Jev Decision Index 0.3.1

| Model | Jev Decision Index ↑ | Public ↑ | Same-skill tests ↑ | New-domain tasks ↑ |
| --- | ---: | ---: | ---: | ---: |
| **d3** | **64.7** | **65.5** | **62.8** | **56.6** |
| Perplexity Decider v1.1 (27B) | 62.8 | 62.3 | 61.1 | 55.6 |
| Fastino GLiDE no-thinking (28B) | 60.2 | 59.1 | 59.5 | 52.9 |
| Jev | 60.1 | 58.0 | 58.0 | 55.0 |
| Torchcast Decision 27B | 59.9 | 65.1 | 58.1 | 50.8 |
| Decision 2.0 (27B) | 55.9 | 57.0 | 55.7 | 47.9 |

![Jev Decision Index against model size](assets/index-pareto.png)

![Jev Decision Index by area: d3 and Decision 2.0 (27B)](assets/index-areas.png)

### Images: Jev Decision Index vision board 0.3.1

| Model | Vision Index ↑ | Public ↑ | Private ↑ |
| --- | ---: | ---: | ---: |
| **d3** | **71.6** | **74.2** | **68.9** |
| Perplexity Decider v1.1 (27B) | 70.6 | 73.2 | 67.9 |
| JEV-27B-VL | 69.6 | 72.8 | 66.4 |
| Solomon v1.1 (27B) | 66.8 | 69.9 | 63.8 |

d3 on the public vision benchmarks († approximate rebuild):

| Benchmark | d3 |
| --- | ---: |
| CV-Bench | 76.2 |
| BLINK | 59.9 |
| RealWorldQA | 73.4 |
| CharXiv † | 91.7 |
| InfographicVQA † | 96.7 |
| Mind2Web † | 90.0 |
| Winoground | 85.8 |
| KIE (CORD+FUNSD) † | 97.7 |
| Moderation (Hateful Memes) | 43.9 |
| R-Bench-M | 33.0 |
| MMMU-Pro vision | 47.0 |

### Videos: Perception Test

| Perception Test (validation) | d3 |
| --- | ---: |
| **All 19,140 questions** | **77.4** |
| Memory | 80.3 |
| Abstraction | 71.2 |
| Physics | 72.5 |
| Semantics | 88.8 |

Multiple-choice video question answering (three options per question, chance 33.3): top-1 accuracy (%), videos read with the defaults above.

<sub>d3: internal evaluation. Others: live board data, text 2026-10-10, vision 2026-10-09.</sub>

## License

Apache-2.0 ([LICENSE](LICENSE)). Built on [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0).

## Citation

```bibtex
@misc{d3_2026,
  title        = {{d3}: A Multimodal Foundation Decision Model},
  author       = {{vLLM Semantic Router Team}},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/vllm-sr/d3}}
}
```

Trained on AMD Instinct MI325X GPUs.
