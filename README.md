# FGSM adversarial attack on an MNIST classifier (PyTorch)

A small learning project: I trained a simple neural network on MNIST in PyTorch and attacked it with the **Fast Gradient Sign Method (FGSM)** (Goodfellow, Shlens & Szegedy, 2015) to see how quickly its accuracy collapses under small, targeted pixel changes.

## What I did

1. Trained a fully connected network (784 → 512 → 512 → 10, ReLU) on MNIST for 5 epochs with SGD (learning rate 0.01, batch size 64). Clean test accuracy: **92.0 %**.
2. Implemented FGSM:
   `x_adv = clip(x + ε · sign(∇x Loss(model(x), y)), 0, 1)`
3. Evaluated the attacked model on the full MNIST test set (10,000 images) for ε ∈ {0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30}.

## Design choices

- **No input normalization:** pixel values stay in [0, 1], so the attack and the clipping work directly in image space (no de-normalization step needed).
- **Batched attack:** 64 images per batch instead of one image at a time, which makes the evaluation much faster.
- **White-box setting:** the attacker uses the model's own gradients.

## Results

| ε (attack strength) | Accuracy under attack |
|---|---|
| 0.00 | 92.0 % |
| 0.05 | 72.3 % |
| 0.10 | 32.5 % |
| 0.15 | 8.1 % |
| 0.20 | 1.2 % |
| 0.25 | 0.1 % |
| 0.30 | 0.0 % |

Examples at ε = 0.2 (true label → prediction after the attack): 7 → 3, 2 → 5, 1 → 2, 0 → 5, 4 → 7. The digits are still easy for a human to read. The plots are in the notebook.

## What I learned and the limits

- A model with 92 % clean accuracy loses most of its accuracy at ε = 0.1 and almost all of it at ε = 0.2.
- On MNIST's black background, the perturbation is visible as grey noise at the larger ε values, so it is not "invisible" at these strengths.
- Limits: a small fully connected model (not a CNN), one dataset, one attack (FGSM), one training run, and no defense yet.
- Next steps I want to try: a CNN, adversarial training as a defense, and checking how *confident* the model is on adversarial inputs, which relates to the confidence-gated automation in my MSc thesis.

## References

- Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). *Explaining and Harnessing Adversarial Examples*. ICLR 2015.
- Built while following the official PyTorch tutorials (Quickstart; Adversarial Example Generation), adapted: my own trained model, no normalization, batched evaluation.

## How to run

Open `fgsm-mnist-pytorch.ipynb` in Google Colab and choose **Runtime → Run all**.
