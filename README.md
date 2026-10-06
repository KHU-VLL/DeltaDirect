# Which Way Did It Move? Diagnosing and Overcoming Directional Motion Blindness in Video-LLMs

[Jongseo Lee](https://jong980812.github.io/)<sup>1†</sup>, [Hyuntak Lee](https://hyuntak03.github.io/)<sup>1†</sup>, Sunghun Kim<sup>1</sup>, Sooa Kim<sup>1</sup>, Jihoon Chung<sup>2</sup>, [Jinwoo Choi](https://sites.google.com/site/jchoivision/home?authuser=0)<sup>1\*</sup>

**<sup>1</sup>Kyung Hee University, <sup>2</sup>Princeton University**

<sup>†</sup>Equal contribution, <sup>\*</sup>Corresponding author

NeurIPS 2026

[![arXiv](https://img.shields.io/badge/arXiv-2605.22823-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.22823) [![Project Page](https://img.shields.io/badge/Project-Page-4285F4?logo=googlechrome&logoColor=white)](https://jong980812.github.io/which-way-did-it-move/) [![Dataset](https://img.shields.io/badge/Dataset-MoDirect-FFD21E?logo=huggingface)](https://huggingface.co/datasets/KHUjongseo/Modirect-family) [![GitHub stars](https://img.shields.io/github/stars/KHU-VLL/DeltaDirect?style=flat&label=Stars&color=e3b341&logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxNiAxNiI+PHBhdGggZmlsbD0iI2UzYjM0MSIgZD0iTTggLjI1YS43NS43NSAwIDAgMSAuNjczLjQxOGwxLjg4MiAzLjgxNSA0LjIxLjYxMmEuNzUuNzUgMCAwIDEgLjQxNiAxLjI3OWwtMy4wNDYgMi45Ny43MTkgNC4xOTJhLjc1MS43NTEgMCAwIDEtMS4wODguNzkxTDggMTIuMzQ3bC0zLjc2NiAxLjk4YS43NS43NSAwIDAgMS0xLjA4OC0uNzlsLjcyLTQuMTk0TC44MTggNi4zNzRhLjc1Ljc1IDAgMCAxIC40MTYtMS4yOGw0LjIxLS42MTFMNy4zMjcuNjY4QS43NS43NSAwIDAgMSA4IC4yNVoiLz48L3N2Zz4=)](https://github.com/KHU-VLL/DeltaDirect/stargazers) ![visitors](https://visitor-badge.laobi.icu/badge?page_id=KHU-VLL.DeltaDirect&left_text=Visitors)

![DeltaDirect overview](img/github-thumbnail.svg?raw=true)

PyTorch implementation of **DeltaDirect** for instruction tuning [LLaVA-Video-7B](https://huggingface.co/lmms-lab/LLaVA-Video-7B-Qwen2) on MoDirect-Inst, with a baseline trained without it. For details, see the paper: **[Which Way Did It Move? Diagnosing and Overcoming Directional Motion Blindness in Video-LLMs](https://arxiv.org/abs/2605.22823)**.

Video-LLMs often cannot tell which way an object moves, a failure we call *directional motion blindness*. We trace it to a *direction binding gap*: the model sees the direction but fails to say it. **DeltaDirect** is a simple training objective that teaches the model direction from frame-to-frame feature changes. Trained only on synthetic videos, it improves accuracy from 25.9% to 85.9% on SynBench and by 21.4 points on RealBench.

## Installation

```bash
conda create -n deltadirect python=3.11 -y
conda activate deltadirect

git clone https://github.com/KHU-VLL/DeltaDirect.git && cd DeltaDirect
pip install -e ".[train]"
```

## Getting Started

### Data preparation.

Download MoDirect into `data/MoDirect`.

```bash
huggingface-cli download KHUjongseo/Modirect-family --repo-type dataset --local-dir data/MoDirect
```

### Training

```bash
bash scripts/train/train.sh scripts/train/configs/llava_video_7b_deltadirect.sh
```

Hyperparameters are set in [the config](scripts/train/configs/llava_video_7b_deltadirect.sh). Set `USE_DELTA_DIRECT=false` to train the baseline without DeltaDirect.

### Loading a trained checkpoint

```python
from llava.model.builder import load_pretrained_model

tokenizer, model, image_processor, _ = load_pretrained_model(
    model_path="work_dirs/<EXP_NAME>",
    model_base="lmms-lab/LLaVA-Video-7B-Qwen2",
    model_name="llava-qwen-lora",
    device_map="cuda",
)
```

## MoDirect Dataset

MoDirect is available on [Hugging Face](https://huggingface.co/datasets/KHUjongseo/Modirect-family) and consists of two parts:

- **MoDirect-Inst**: 100K synthetic videos with instruction-tuning conversations and per-frame 2D motion vectors
- **MoDirect-Bench**: a multiple-choice benchmark for motion direction, with SynBench and RealBench

<details>
<summary><b>Composition</b></summary>

| | Subset | # QA | Choices |
|:---|:---:|---:|:---:|
| **MoDirect-Inst** | | 100,000 | |
| **SynBench** | P-Syn | 6,000 | 4-way |
| | P-Real | 6,000 | 4-way |
| | C-Syn | 6,000 | 4-way |
| | C-Real | 6,000 | 4-way |
| **RealBench** | SSv2 | 722 | 2-way |
| | KTH | 899 | 2-way |
| | TOMATO | 403 | 3-7-way |

</details>

<details>
<summary><b>Annotation format</b></summary>

`MoDirect-Inst.json` follows the LLaVA conversation format with an additional `direction_gt` field.

```json
{
  "id": "modirect_inst_000000",
  "video": "videos/000/000000.mp4",
  "conversations": [
    {"from": "human", "value": "<image>\nWhich direction does the object move? Options: A. No Movement, B. Right, C. Lower-Right, D. Leftward, E. Up Answer with the option letter only."},
    {"from": "gpt", "value": "C"}
  ],
  "direction_gt": [[0.728, 0.685], [0.619, 0.785], [0.736, 0.677], [0.709, 0.705], [0.672, 0.740], [0.704, 0.710], [0.735, 0.678]]
}
```

- `direction_gt` is the unit motion vector (x, y) between each pair of adjacent frames, so an 8-frame video has 7 vectors.
- Videos of a static object have no `direction_gt` and are trained with the language modeling loss only.

</details>

## License

The code is licensed under the [Apache 2.0 license](LICENSE).

## Acknowledgements

This project is built upon the following works:
- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT): Base codebase and LLaVA-Video model
- [COCO](https://cocodataset.org), [Places365](http://places2.csail.mit.edu), [Something-Something V2](https://www.qualcomm.com/developer/software/something-something-v-2-dataset), [KTH](https://www.csc.kth.se/cvap/actions/), [TOMATO](https://github.com/yale-nlp/TOMATO): Source data of MoDirect-Bench

We thank all authors who contributed to these foundational projects.

## Citing DeltaDirect

If you use DeltaDirect or MoDirect in your research, please use the following BibTeX entry.

```bibtex
@inproceedings{lee2026whichway,
  title     = {Which Way Did It Move? Diagnosing and Overcoming Directional Motion Blindness in Video-LLMs},
  author    = {Lee, Jongseo and Lee, Hyuntak and Kim, Sunghun and Kim, Sooa and Chung, Jihoon and Choi, Jinwoo},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
}
```
