# Lab 5 — Scikit-learn Digits CNN with WandB Experiment Tracking

A convolutional neural network trained on the **scikit-learn Digits** dataset with full experiment tracking, sample prediction logging, and confusion matrix visualization via **Weights & Biases**.

## Dataset

**sklearn.datasets.load_digits** — 1,797 grayscale images (8 × 8 pixels) of handwritten digits (0–9). Pixel values range from 0 to 16. The dataset is split 80/20 into training and test sets using stratified sampling.

Unlike Fashion MNIST or CIFAR-10, this dataset is loaded entirely from scikit-learn with no external download required.

## Model Architecture

| Layer | Details |
|---|---|
| Conv2D Block 1 | Conv2D (32 filters, 3×3, ReLU, same padding) → MaxPool 2×2 → Dropout 0.25 |
| Conv2D Block 2 | Conv2D (64 filters, 3×3, ReLU, same padding) → MaxPool 2×2 → Dropout 0.25 |
| Dense Head | Flatten → Dense 128 (ReLU) → Dropout 0.25 → Dense 10 (Softmax) |

**Optimizer:** SGD with Nesterov momentum (lr=0.01, momentum=0.9)

The architecture is intentionally lightweight to match the small 8×8 input resolution. Two pooling layers reduce the spatial dimensions from 8×8 → 4×4 → 2×2 before flattening.

## WandB Integration

The following are logged every epoch:

- **Training and validation loss/accuracy** (via `WandbMetricsLogger`)
- **Model checkpoints** (via `WandbModelCheckpoint`)
- **Learning rate** (custom `LogLRCallback`)
- **Sample predictions table** with images (custom `LogSamplesCallback`)
- **Confusion matrix** on validation set (custom `ConfusionMatrixCallback`)
- **Dataset metadata** (train/test split sizes, number of classes, image shape)
- **Final model artifact** (saved `.h5` + model summary text)

## Setup and Usage

### 1. Clone the repo

```bash
git clone https://github.com/kavin1317/wandb_lab-5.git
cd W&B_LAB-5
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Login to WandB

```bash
wandb login
```

You will be prompted for your API key. Get it from [https://wandb.ai/authorize](https://wandb.ai/authorize).

### 4. Run the notebook

```bash
jupyter notebook wandb.ipynb
```

Run all cells. Training progress and visualizations will appear in your WandB dashboard under the project **Lab2-Sklearn-Digits-CNN**.

## Hyperparameters

| Parameter | Default |
|---|---|
| `conv1_filters` | 32 |
| `conv2_filters` | 64 |
| `dense_units` | 128 |
| `dropout` | 0.25 |
| `learn_rate` | 0.01 |
| `momentum` | 0.9 |
| `epochs` | 30 |
| `batch_size` | 32 |
| `test_size` | 0.2 |
| `random_state` | 42 |

All hyperparameters are tracked in the WandB config and can be modified in the `DigitsTrainer.__init__` method.

## Project Structure

```
lab2-sklearn-digits-wandb/
├── wandb.ipynb          # Main training notebook
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── .gitignore          # Git ignore rules
```

## References

- [Scikit-learn Digits Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
- [Weights & Biases Docs](https://docs.wandb.ai/)
- [Keras Documentation](https://keras.io/)