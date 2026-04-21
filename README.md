# AI 100 Final Project — CIFAR-10 CNN Classifier with Bug Case Analysis

**Course:** AI 100  
**Deadline:** April 24, 2026  
**Group Members:** Nawaf Alqahtani · Nasser Almalki · Mohammed Ajwah · Yazeed Alshehri · Maria Almansour

---

## Project Overview

This project builds a CNN image classifier on CIFAR-10 and intentionally introduces 10 bugs to explore how AI systems fail and how GenAI (Claude) can serve as a Socratic debugging partner.

## Repository Structure

```
├── train.py           # Base AI system — CNN classifier on CIFAR-10
├── bug_cases.xlsx     # Google Sheet / Excel with all 10 bug cases
├── final_report.pdf   # PDF report: lessons about GenAI and AI systems
└── README.md
```

## The AI System

- **Architecture:** 3-layer CNN with BatchNorm, Dropout → 2 fully connected layers
- **Dataset:** CIFAR-10 (60,000 images, 10 classes)
- **Framework:** PyTorch
- **Optimizer:** Adam (lr=0.001), StepLR scheduler

## Running the Base System

```bash
pip install torch torchvision
python train.py
```

## Bug Cases Summary

| Case | Student | Bug Description | GenAI Label |
|------|---------|-----------------|-------------|
| 1 | Nawaf Alqahtani | Wrong Linear layer input size (4×4 → 8×8) | Bad |
| 2 | Nawaf Alqahtani | Removed optimizer.zero_grad() | Bad |
| 3 | Nasser Almalki | CrossEntropyLoss → MSELoss | Bad |
| 4 | Nasser Almalki | Wrong normalization statistics | Bad |
| 5 | Mohammed Ajwah | MaxPool2d(2,2) → MaxPool2d(3,3) | Bad |
| 6 | Mohammed Ajwah | Swapped backward() and step() | Bad |
| 7 | Yazeed Alshehri | model.train() → model.eval() during training | Bad |
| 8 | Yazeed Alshehri | DataLoader shuffle=True → shuffle=False | Bad |
| 9 | Maria Almansour | ReLU → Sigmoid activations | Bad |
| 10 | Maria Almansour | Learning rate 0.001 → 10.0 | Bad |

## Submission

Submitted via Canvas. GitHub URL included in submission.
