---
pipeline_tag: zero-shot-classification
license: apache-2.0
base_model: Qwen/Qwen3.5-9B
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

![d3-flash: Decision 3.0](assets/banner.png)

# d3-flash

**d3-flash** is the 9B multimodal foundation decision model of Decision 3.0, the decision models of [vLLM Semantic Router](https://github.com/vllm-project/semantic-router). Give it an input (text or JSON, optionally with images and videos) and the questions you need answered: pick one of several options, say yes or no, or rate on a scale. It answers them all in one call and returns a probability for every answer, without generating text.

| | |
| --- | --- |
| **Parameters** | 8.39B, including the 0.46B vision encoder |
| **Inputs** | Text or JSON, plus images and videos (several per request) |
| **Decision types** | Choice · Yes / No · Score |
| **License** | Apache-2.0 |

## Highlights

- **Jev Decision Index 0.3, public suite: 57.79**, measured with the official 0.3 kit on the released weights: all 140,178 public requests answered, none unsupported.
- **+11.0 on the public suite over Decision 2.0** (its 9B model: 46.76 on the board), ahead in all five areas.
- **Reads images:** multiple images per request (PNG, JPEG or WebP), given as paths, URLs, PIL images or base64 data URLs; every question of the request sees all of them.
- **Reads videos:** multiple videos per request (MP4, WebM, MOV or MKV), given as paths, URLs, base64 data URLs or frame arrays, read at 2 frames per second; **Perception Test (multiple-choice video QA): 73.3** (internal evaluation).
- **Speed:** a median of 19.1 ms for a text request, 149.4 ms for a request with an image and 407.6 ms for a request with a 10-second video, on one AMD Instinct MI325X GPU, one request at a time.
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

model = AutoModel.from_pretrained("vllm-sr/d3-flash", trust_remote_code=True)

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
receipt = hf_hub_download("vllm-sr/d3-flash", "assets/example-receipt.png")
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
clip = hf_hub_download("vllm-sr/d3-flash", "assets/example-video.mp4")
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
# transformers.pipeline("decision", model="vllm-sr/d3-flash", trust_remote_code=True)(state=..., questions=..., images=..., videos=...)
```

Images go before the text of the request, each read at up to 1.6 megapixels; videos follow the images, read at 2 frames per second (at most 32 frames spread over the whole video, each at up to 0.2 megapixels). Every question of the request sees all of them.

## Evaluation

### Text: Jev Decision Index 0.3.1

| Model | Jev Decision Index ↑ | Public ↑ | Same-skill tests ↑ | New-domain tasks ↑ |
| --- | ---: | ---: | ---: | ---: |
| **d3-flash** | **59.0** | **57.8** | **57.5** | **52.9** |
| Bespoke Nimble 9B v3 | 54.7 | 57.2 | 52.3 | 49.1 |
| Nace.AI Drex v1.5 | 50.8 | 58.1 | 51.9 | 39.7 |
| AJev lora5 (Gemma 4 12B) | 50.0 | 51.7 | 51.4 | 41.5 |
| Winnow-12B | 49.3 | 50.7 | 48.1 | 44.2 |
| Decision 2.0 (9B) | 48.2 | 46.8 | 47.4 | 44.5 |

![Jev Decision Index against model size](assets/index-pareto.png)

![Jev Decision Index by area: d3-flash and Decision 2.0 (9B)](assets/index-areas.png)

### Images: Jev Decision Index vision board 0.3.1

| Model | Vision Index ↑ | Public ↑ | Private ↑ |
| --- | ---: | ---: | ---: |
| **d3-flash** | **66.9** | **69.5** | **64.3** |
| JPT-9B | 61.7 | 65.4 | 58.0 |
| Winnow-12B | 58.8 | 63.9 | 53.6 |
| Jev-Omni (12B) | 56.1 | 60.1 | 52.1 |

d3-flash on the public vision benchmarks († approximate rebuild):

| Benchmark | d3-flash |
| --- | ---: |
| CV-Bench | 75.2 |
| BLINK | 55.5 |
| RealWorldQA | 63.4 |
| CharXiv † | 83.0 |
| InfographicVQA † | 95.1 |
| Mind2Web † | 85.1 |
| Winoground | 83.3 |
| KIE (CORD+FUNSD) † | 96.3 |
| Moderation (Hateful Memes) | 41.3 |
| R-Bench-M | 20.8 |
| MMMU-Pro vision | 38.7 |

### Videos: Perception Test

| Perception Test (validation) | d3-flash |
| --- | ---: |
| **All 19,140 questions** | **73.3** |
| Memory | 75.5 |
| Abstraction | 68.7 |
| Physics | 66.1 |
| Semantics | 86.1 |

Multiple-choice video question answering (three options per question, chance 33.3): top-1 accuracy (%), videos read with the defaults above.

<sub>d3-flash: internal evaluation. Others: live board data, text 2026-10-10, vision 2026-10-09.</sub>

## License

Apache-2.0 ([LICENSE](LICENSE)). Built on [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) (Apache-2.0).

## Citation

```bibtex
@misc{d3_flash_2026,
  title        = {{d3-flash}: A Multimodal Foundation Decision Model},
  author       = {{vLLM Semantic Router Team}},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/vllm-sr/d3-flash}}
}
```

Trained on AMD Instinct MI325X GPUs.
