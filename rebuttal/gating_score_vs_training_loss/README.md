# Convergence Analysis and Visualization

## Description

This folder contains the scripts for analyzing the convergence patterns of different datasets using node gate variability and training loss trends. The focus is on understanding how variability in node gate distributions correlates with training loss during model training.

### Key Objectives:
1. Analyze node gate variability during training for multiple datasets.
2. Compare variability trends with training loss to identify patterns of convergence.
3. Quantify relationships between variability and loss using statistical metrics (e.g., Pearson correlation).
4. Visualize these patterns through stacked area plots and dual-axis charts.

---

## What Has Been Done

1. **Data Preparation**:
   - Loaded node gate score data for 7 different input types (`Logits`, `Features`, `Degrees`, `[Logits, Features]`, `[Features, Degrees]`, `[Logits, Degrees]`, `[Logits, Features, Degrees]`) over training epochs.
   - Loaded training loss data corresponding to the same training epochs.

2. **Variability Computation**:
   - Calculated variability (e.g., standard deviation) of node gate distributions over training epochs to capture how the model’s focus shifts during training. See file `PR.Plot_Gating_Scores.ipynb`.

3. **Quantitative Analysis**:
   - Calculated the Pearson correlation coefficient between node gate variability and training loss to quantify their relationship. See file `PR.Analyze_Gating_vs_Training_Loss.ipynb`.

---

## Key Findings

- Datasets like `computers`, `CS`, and `photo` show significant variability in node gate focus during the training process, aligning with rapid changes in training loss.
- Datasets like `pubmed` and `physics` exhibit early stabilization, with minimal variability in node gates and smoother training loss trends.
