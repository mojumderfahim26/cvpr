CNN-Based Image Classification on Fashion-MNIST

Course: Computer Vision & Pattern Recognition
Student Name: MD.Fahim Abdullah
Student ID: 23-50691-1

Overview

This project designs and trains a custom Convolutional Neural Network (CNN) in PyTorch to classify grayscale clothing images into 10 categories. The notebook covers the full pipeline: data loading, exploration, preprocessing, model building, training, evaluation, and model saving.

Dataset
Name: Fashion-MNIST (Zalando Research), loaded via torchvision.datasets. It is also available on Kaggle and Hugging Face.
Images: 70,000 in total, 28×28 pixels, 1 channel (grayscale).
Split: 60,000 training images (divided 80% train / 20% validation) and 10,000 test images.
Classes (10, balanced): T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.
Preprocessing
Training augmentation: random horizontal flip and random rotation (±10°).
Conversion to tensors and normalisation with the dataset mean (0.2860) and standard deviation (0.3530).
Validation and test data are only normalised, with no augmentation.
Model Architecture
Layer	Details
Conv2d + ReLU + MaxPool	1 → 64 filters, 3×3 kernel (28→26→13)
Conv2d + ReLU + MaxPool	64 → 128 filters, 3×3 kernel (13→11→5)
Flatten	128 × 5 × 5 = 3200 features
Linear + ReLU + Dropout	3200 → 256, dropout 0.3
Linear (output)	256 → 10 logits

The model has about 0.9 million trainable parameters (896,522).

Design justification: The first convolution learns simple patterns such as edges and textures, and the second combines them into larger shapes such as sleeves, straps and soles. Small 3×3 kernels are efficient, and max pooling reduces computation while adding tolerance to small shifts. Dropout reduces over-fitting in the large fully connected layer.

Training Setup
Hyperparameter	Value
Optimizer	Adam
Learning rate	0.001
Loss function	CrossEntropyLoss
Batch size	64
Epochs	10
Dropout	0.3
Random seed	42 (for reproducibility)
Evaluation
Training and validation loss/accuracy curves over 10 epochs.
Test loss and accuracy on the unseen test set: XX.XX% (replace after running).
Per-class precision, recall and F1-score (classification report).
Confusion matrix, per-class accuracy, and examples of misclassified images.
Observation: Visually similar classes (Shirt, T-shirt/top, Pullover, Coat) are the hardest to separate at 28×28 resolution. Footwear, bags and trousers are recognised most easily.
Model Saving

The trained weights are saved as CNN_23-50691-1.pth (about 3.5 MB). The notebook reloads them into a fresh model to confirm the test accuracy is unchanged.

Files
CNN_23-50691-1.ipynb – full notebook
CNN_23-50691-1.pth – trained model weights
Future Work

Batch Normalization, deeper networks, stronger augmentation, a learning-rate scheduler, early stopping, and transfer learning with a pre-trained network.
