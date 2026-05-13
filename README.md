# MMDiffuzzy: Fuzzy Memory Guided Diffusion for Uncertainty-Aware Multimodal Fusion in WSIs Analysis

Thank you very much for your interest in our work! This repository contains the research implementation of MMDiffuzzy.

## 1. Project Overview

As illustrated in the framework, MMDiffuzzy consists of three core components:
* Latent Diffusion Module progressively refines multimodal representations through a time-dependent denoising process.
* Dynamic Fuzzy Memory Particle (DFMP) Module constructs evolving fuzzy memory particles to model structured cross-modal uncertainty.
* Fuzzy Memory Guidance (FMG) Module injects step-aware fuzzy memory into the diffusion backbone via cross-attention mechanisms for survival risk prediction.

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

You can install the environment with the following code:

```conda env create -f environment.yaml```

[environment.yaml](https://github.com/Choutyear/FMDNN/blob/main/Files/encironment.yaml)


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



## 7. 
Due to the strict page limitation (35 pages) imposed by the journal, only the compact numerical summary is included in the manuscript. For completeness and reproducibility, we provide the detailed diffusion-step analysis and additional experimental results in this repository.



## Diffusion Step Analysis (Expand from Table 4)

| Methods      | Metric   | 10    | 20    | 50    | 100   | 200   |
|---------------|----------|-------|-------|-------|-------|-------|
| w/o Fuzzy    | F1       | 66.30 | 69.89 | 72.89 | 75.94 | 75.57 |
|               | c-index  | 0.521 | 0.536 | 0.566 | 0.599 | 0.614 |
| w/o Diffu    | F1       | 74.26 | 74.89 | 74.56 | 75.29 | 76.01 |
|               | c-index  | 0.703 | 0.706 | 0.711 | 0.716 | 0.719 |
| M-Direct     | F1       | 45.92 | 48.14 | 50.97 | 52.36 | 53.10 |
|               | c-index  | 0.609 | 0.624 | 0.652 | 0.663 | 0.665 |
| M-CNN        | F1       | 62.02 | 63.06 | 69.21 | 69.34 | 72.08 |
|               | c-index  | 0.664 | 0.690 | 0.727 | 0.725 | 0.738 |
| M-Res        | F1       | 67.10 | 69.70 | 73.88 | 74.86 | 75.32 |
|               | c-index  | 0.705 | 0.726 | 0.754 | 0.758 | 0.753 |
| U-ADD        | F1       | 72.73 | 75.33 | 80.13 | 81.55 | 81.42 |
|               | c-index  | 0.767 | 0.770 | 0.795 | 0.807 | 0.802 |
| U-CA         | F1       | 73.96 | 75.23 | 79.58 | 81.88 | 82.18 |
|               | c-index  | 0.767 | 0.786 | 0.816 | 0.821 | 0.823 |
| MMDiffuzzy   | F1       | 76.02 | 82.47 | 86.19 | 87.75 | 88.10 |
|               | c-index  | 0.741 | 0.785 | 0.838 | 0.844 | 0.851 |


Several important observations can be made:

1. **Effectiveness of diffusion refinement**  
   The proposed MMDiffuzzy consistently improves as the number of diffusion steps increases, demonstrating the effectiveness of iterative diffusion-based multimodal refinement. The performance gain is especially significant from small to moderate steps, while gradually saturating at larger steps.

2. **Necessity of the diffusion backbone**  
   The variant without diffusion (w/o Diffu) exhibits only marginal improvements as steps increase, indicating that simply increasing computation or repeated forwarding cannot reproduce the benefits brought by diffusion refinement. This supports our claim that the performance gains mainly arise from the structured diffusion process rather than additional computation alone.

3. **Complementary role of fuzzy modeling**  
   Removing fuzzy modeling (w/o Fuzzy) leads to a substantial performance degradation across all diffusion steps, particularly in c-index, suggesting that fuzzy-guided uncertainty modeling is critical for stable multimodal alignment and survival prediction.

4. **Behavior of conventional fusion variants**  
   Traditional multimodal fusion strategies (M-Direct, M-CNN, M-Res, U-ADD, and U-CA) also benefit from increased refinement steps to some extent. However, their performance remains consistently below MMDiffuzzy, demonstrating the advantage of combining diffusion refinement with fuzzy memory-guided multimodal interaction.


## Calibration Analysis on BLCA Multimodal Classification

| Model      | ECE  | Brier Score  |
|------------|------:|---------------:|
| MC-Fusion  | 0.081 | 0.174          |
| MMDiffuzzy | 0.056 | 0.142          |

The calibration-related evaluation suggests that MMDiffuzzy produces more reliable probabilistic predictions than conventional multimodal fusion approaches. Specifically, the proposed fuzzy memory-guided diffusion framework achieves lower Expected Calibration Error (ECE) and Brier Score, indicating improved consistency between prediction confidence and empirical correctness. These results further support the uncertainty-aware property of the proposed framework.


## Failure Case Visualization

![Failure Case Visualization](https://github.com/Choutyear/MMDiffuzzy/blob/main/data/figcam.jpg)

Representative CAM-like visualization from the BLCA multimodal classification task using the proposed MMDiffuzzy framework. Compared with relatively concentrated and stable activation patterns, failure-prone cases exhibit diffuse and ambiguous attention distributions with weaker structural focus on discriminative tumor regions. Such challenging cases are commonly associated with heterogeneous morphology, unclear tumor boundaries, or weak multimodal consistency, which increase the difficulty of reliable multimodal alignment and prediction.

## Table 4. Sensitivity analysis of diffusion steps and memory particle size.

| Setting | Value | F1 (%) | C-index |
|---|---:|---:|---:|
| Diffusion steps \(T\) | 10  | 76.02 | 0.741 |
|  | 20  | 82.47 | 0.785 |
|  | 50  | 86.19 | 0.838 |
|  | 100 | 87.75 | 0.844 |
|  | 200 | 88.10 | 0.851 |
| Memory particles \(K\) | 4  | 84.80 | 0.829 |
|  | 8  | 86.52 | 0.839 |
|  | 16 | 87.86 | 0.848 |
|  | 32 | 87.39 | 0.847 |




