# Fashion Item Classification Using a Custom CNN

**Course:** Computer Vision and Pattern Recognition  
**Student Name:** MD.Fahim Abdullah  
**Student ID:** 23-50691-1  
**Framework:** Python, PyTorch, Google Colab  
**Dataset:** Fashion-MNIST (Zalando Research; Kaggle / Hugging Face / torchvision)

---

## Project Overview

This project implements a custom Convolutional Neural Network (CNN) in PyTorch to classify grayscale images of clothing items into 10 categories. The notebook covers dataset exploration, preprocessing, data augmentation, CNN design, training, validation, testing, and performance analysis.

## Dataset

- **Total images:** 70,000
- **Classes:** 10 (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)
- **Input size:** 28 × 28 grayscale (1 channel)
- **Split:** Official 60,000 training / 10,000 test split. The training set is further divided 80% / 20% into training and validation (random seed 42). The classes are perfectly balanced (6,000 training images per class).

| Subset     | Number of images |
|------------|------------------|
| Training   | 48,000           |
| Validation | 12,000           |
| Testing    | 10,000           |

## Preprocessing and Augmentation

- Convert images to tensors and normalize with mean 0.2860 and standard deviation 0.3530.
- Training only: random horizontal flip and random rotation (±10°).
- Validation and test images use deterministic preprocessing without augmentation.

## Custom CNN Architecture

The CNN is built from scratch and contains:

1. Convolution (1 → 64 channels, 3×3), ReLU, max pooling
2. Convolution (64 → 128 channels, 3×3), ReLU, max pooling
3. Flatten → fully connected layer (256 units) → ReLU → dropout (0.3)
4. Output fully connected layer (10 class logits)

**Trainable parameters:** 896,522

## Training Configuration

| Setting        | Value                                |
|----------------|--------------------------------------|
| Loss function  | CrossEntropyLoss                     |
| Optimizer      | Adam                                 |
| Learning rate  | 0.001                                |
| Batch size     | 64                                   |
| Epochs         | 10                                   |
| Model saved    | Final trained weights (`state_dict`) |

## Results

The values below are from the latest recorded training and evaluation run in the project notebook.

| Metric                    | Result     |
|---------------------------|------------|
| Final training accuracy   | 91.67%     |
| Final validation accuracy | 91.36%     |
| Test loss                 | 0.2658     |
| **Test accuracy**         | **90.85%** |
| Macro precision           | 91%        |
| Macro recall              | 91%        |
| Macro F1-score            | 91%        |

Validation accuracy rose from 85.41% (epoch 1) to a peak of 91.67% (epoch 9), and the gap between training and validation accuracy stayed small, so there is no strong sign of over-fitting.

## Class-wise Findings

- The highest class-level F1-scores are for Trouser (0.99), Sandal (0.98) and Bag (0.98).
- Shirt was the most challenging class (F1-score 0.70, recall 64.10%).
- The largest directional confusion was Shirt → T-shirt/top, with 116 test images. Shirt was also confused with Coat (112 images) and Pullover (98 images).
- Visually similar upper-body items (Shirt, T-shirt/top, Pullover, Coat) are the hardest to separate at 28 × 28 resolution, while footwear, bags and trousers are recognised most easily.
- A high overall accuracy does not imply equally strong performance across all classes.

| Class       | Per-class accuracy |
|-------------|--------------------|
| Bag         | 98.20%             |
| Trouser     | 97.50%             |
| Sandal      | 97.40%             |
| Sneaker     | 97.20%             |
| Ankle boot  | 96.30%             |
| Dress       | 91.80%             |
| Coat        | 90.80%             |
| Pullover    | 90.40%             |
| T-shirt/top | 84.80%             |
| Shirt       | 64.10%             |

## Repository Files

```
Fashion-MNIST-CNN/
├── CNN_23-50691-1.ipynb       # Complete project notebook
├── CNN_23-50691-1.pth         # Trained model weights (3.59 MB)
└── README.md                  # Project documentation
```

Dataset files are not included in this repository because they are downloaded automatically through `torchvision`.

## How to Run

1. Open `CNN_23-50691-1.ipynb` in Google Colab.
2. Select a GPU runtime if available: Runtime → Change runtime type → GPU.
3. Run the cells in order. Fashion-MNIST downloads automatically, so no Kaggle credentials are needed.
4. The notebook creates the train/validation split, trains the CNN, and saves the weights.
5. Run the test-evaluation and visualization cells to reproduce the metrics and graphs.

> **Note:** Retraining may give slightly different results depending on random state, GPU environment, and library versions. The results reported above are the recorded results in the submitted notebook.

## Notebook Sections

1. Introduction and Import Libraries
2. Dataset Loading and Exploration
3. Dataset Description
4. Data Preprocessing and Augmentation
5. Dataset Summary
6. Hyperparameter Settings
7. DataLoader Creation
8. CNN Model Architecture
9. Model Summary
10. Training and Validation Functions
11. Model Training
12. Training and Validation Loss
13. Training and Validation Accuracy
14. Model Evaluation on Test Dataset
15. Classification Report
16. Confusion Matrix
17. Model Saving
18. Conclusion and Future Work

## Limitations and Future Work

- Fashion-MNIST images are small (28 × 28) and grayscale, so some classes look very similar and may not represent real-world product photos.
- Results come from a single training run with a fixed seed, so variation across runs is not measured.
- Future improvements include Batch Normalization, deeper architectures, stronger augmentation, a learning-rate scheduler, early stopping, analysis of misclassified images, and transfer learning with a pre-trained network.
