# MMDiffuzzy: Fuzzy Memory Guided Diffusion for Uncertainty-Aware Multimodal Fusion in WSIs Analysis

Thank you very much for your interest in our work! This repository contains the research implementation of MMDiffuzzy.

## 1. Project Overview

As illustrated in the framework, MMDiffuzzy consists of three core components:


## 2. Repository Structure

```python
train.py          # Training pipeline
eval.py           # Evaluation pipeline
model.py          # Diffusion backbone and wrapper
dfmp.py           # Dynamic Fuzzy Memory Particle module
unet_fmg.py       # UNet with fuzzy memory guidance
diffusion.py      # Diffusion scheduler
metrics.py        # C-index implementation
utils.py          # Utility functions
```

This version reflects the exact research implementation used in the experiments reported in the manuscript.

## 3. Environment Setup

Install dependencies:

```python
pip install -r requirements.txt
```


## 4. Dataset Preparation

To validate our model, we selected 4 projects from [TCGA](https://www.cancer.gov/ccg/research/genome-sequencing/tcga) and divided them into 3 datasets, which include matched diagnostic slides, copy number variation, simple nucleotide variation, and transcriptome profiling, as detailed below:
* SARC: A soft tissue sarcoma cohort from TCGA-SARC;
* BLCA: A bladder cancer cohort from TCGA-BLCA;
* GBMLGG: A combined glioma cohort from TCGA-GBM and TCGA-LGG.

## 5. Training

The training pipeline includes:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MMDiffuzzy(config).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(epochs):

    model.train()
    for batch in train_loader:
        loss_dict = model.loss(batch)
        total_loss = loss_dict["total"]

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}: Loss = {total_loss.item():.4f}")
```

The loss function combines: Diffusion reconstruction loss, Auxiliary regularization terms

## 6. Implementation Notes

This repository corresponds to the research implementation used during experimentation.

While all core algorithmic components are fully implemented, the current version:
* Follows a research-oriented scripting layout
* May contain intermediate utilities used during development
* Has not yet undergone full structural refactoring

Future updates will provide:
* Code refactoring and modular reorganization
* Extended documentation
* Additional usage examples

These improvements are structural in nature and do not affect algorithmic correctness or experimental reproducibility.







