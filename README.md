
## Coverage-aware Semantic Representation Learning for Underrepresented Visual Concepts

This repository contains the official implementation of **SemCovNet**, a coverage-aware semantic representation learning framework for underrepresented visual concepts.

We study **Semantic Coverage Imbalance (SCI)**, where empirical ***class–concept*** coverage is uneven across a dataset. SemCovNet addresses this problem through **Coverage-Calibrated Semantic Sharing (CCSS)**, which adapts semantic sharing using pair support and representation uncertainty.

Paper: [preprint arxiv](https://arxiv.org/pdf/2602.16917)

## Citation

If you find this work useful, please consider citing:

```BibTex
@misc{ahammed2026semcovnet,
      title={Coverage-aware Semantic Representation Learning for Underrepresented Visual Concepts}, 
      author={Sakib Ahammed and Xia Cui and Wenqi Lu and Xinqi Fan and Bill Cassidy and Xueli Chen and Moi Hoon Yap},
      year={2026},
      eprint={2602.16917},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.16917}, 
}
```


# **Detailed instructions will be available soon.**


### DINOv3 Hugging Face Access Token

The implementation uses the pretrained **DINOv3 ViT-B/16 LVD-1689M** backbone from Hugging Face.
Before running experiments with DINOv3, obtain access to the model on Hugging Face and provide your Hugging Face Hub token in:

```text
models/encoders/dinov3_encoder.py
```

Set the following variable:

```python
token = "<HF hub token for DINOv3 LVD-1689M>"  # HF hub token for DINOv3 LVD-1689M
```

Model and access instructions are available at:
https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m

Make sure that your Hugging Face account has permission to access the model before running the training or evaluation scripts.

