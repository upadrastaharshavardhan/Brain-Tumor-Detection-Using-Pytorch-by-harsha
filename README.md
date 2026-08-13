# Brain Tumor Detection Using Machine Learning and Deep Learning Models

A PyTorch-based brain MRI image classification project that compares traditional machine learning algorithms, neural networks, and transfer-learning CNN architectures for brain tumor detection.

> **Important:** This README is based on the uploaded Jupyter Notebook `Brain_Tumor_Detection_Using_Pytorch_with_10_ml_models_.ipynb`.
---
<img width="853" height="846" alt="image" src="https://github.com/user-attachments/assets/d3609244-091f-4e91-9390-3b30d0ae28fb" />

---
<img width="1156" height="734" alt="image" src="https://github.com/user-attachments/assets/838a6de3-33e8-48f8-a8c9-1474413b8da3" />

---
<img width="1280" height="543" alt="image" src="https://github.com/user-attachments/assets/486bf993-15fc-4d81-933c-baaf4670a1d2" />

---
<img width="1280" height="543" alt="image" src="https://github.com/user-attachments/assets/2c98f486-f3bb-4343-be09-824a137fc4c3" />

---
<img width="1280" height="728" alt="image" src="https://github.com/user-attachments/assets/bfe03be4-87a5-4d54-9c93-21a955d52d6c" />

---
<img width="1280" height="543" alt="image" src="https://github.com/user-attachments/assets/45c801fd-4dac-4291-83a6-78ffba881d8d" />

---
<img width="1280" height="745" alt="image" src="https://github.com/user-attachments/assets/d4cf6fb6-fc69-4cdc-970f-7bb021652d82" />

---
<img width="1280" height="543" alt="image" src="https://github.com/user-attachments/assets/8a520fda-b29e-4f44-95a2-71ac4b01ed4d" />

---
<img width="1280" height="543" alt="image" src="https://github.com/user-attachments/assets/4aa4a9a2-0cbf-439a-b0c1-297f9858a3e5" />

---
<img width="1280" height="728" alt="image" src="https://github.com/user-attachments/assets/1ecf0926-129d-4864-a55e-7a61ad0071c8" />

---
<img width="1280" height="543" alt="image" src="https://github.com/user-attachments/assets/a156982c-470c-4f99-bd3e-1c6a1280753c" />

---
<img width="1280" height="543" alt="image" src="https://github.com/user-attachments/assets/e324b41c-c444-42a5-8bf4-cd139a254b81" />

---
<img width="1189" height="716" alt="image" src="https://github.com/user-attachments/assets/67d28163-7e6b-49ff-806a-6ede762a98e6" />


---
## Overview

The project classifies brain MRI images and compares multiple machine learning and deep learning approaches. The notebook includes:

- Image loading and preprocessing using `torchvision`
- Corrupted-image handling
- Train/test splitting
- Traditional machine learning models
- Custom neural networks
- Pre-trained CNNs using transfer learning
- Accuracy comparison and visualization
- Grad-CAM explainability for visualizing important image regions
- Advanced CNN training with data augmentation and architecture improvements

The notebook's main project description states that the goal is to compare different approaches for brain tumor detection, from SVM and Random Forest to CNN and ImageNet-pretrained architectures.

## Models

The notebook contains implementations or sections for the following 10 models:

| # | Model | Approach |
|---|---|---|
| 1 | FFNN / MLP | Fully connected neural network |
| 2 | SVM | Traditional machine learning |
| 3 | Random Forest | Ensemble machine learning |
| 4 | VGGNet | Pre-trained CNN section |
| 5 | ResNet18 | Transfer learning |
| 6 | LeNet | CNN trained from scratch |
| 7 | EfficientNet-B0 | Transfer learning |
| 8 | Custom CNN | CNN trained from scratch |
| 9 | MobileNetV2 | Transfer learning |
| 10 | DenseNet121 | Transfer learning |

### Additional CNN experiments

The notebook also contains progressively improved Custom CNN experiments:

- **CNN Alpha 1.0:** Custom CNN + Grad-CAM
- **CNN Alpha 2.0:** Advanced data augmentation + larger CNN architecture
- **CNN Alpha 3.0:** Further CNN architecture refinement

## Dataset

The notebook loads the dataset from Google Drive:

```text
/content/drive/MyDrive/Brain Tumor Data Set
```

Images are loaded using `torchvision.datasets.ImageFolder`.

### Expected dataset structure

For binary classification, the dataset should be organized with one directory per class:

```text
Brain Tumor Data Set/
├── Brain Tumor/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── Healthy/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

The exact directory names can be changed, but `ImageFolder` requires separate class directories.

### Dataset size observed in the notebook

The notebook processed **4,601 images** and created:

- Training samples: **3,680**
- Testing samples: **921**
- Split: **80% training / 20% testing**

## Preprocessing

The baseline preprocessing pipeline performs:

1. Resize images to `64 × 64`
2. Convert images to PyTorch tensors
3. Apply ImageNet normalization

```python
transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

A custom image loader is also used to catch image-loading errors and skip corrupted images.

## Advanced Data Augmentation

The advanced CNN experiment applies:

- `RandomResizedCrop`
- `RandomHorizontalFlip`
- `RandomRotation`
- `ColorJitter`
- Tensor conversion
- ImageNet normalization

Example:

```python
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

## Requirements

The notebook uses Python libraries including:

- Python 3
- PyTorch
- Torchvision
- scikit-learn
- NumPy
- Matplotlib
- Pillow
- tqdm
- OpenCV
- `pytorch-grad-cam`
- Google Colab / Google Drive support

Install the main dependencies with:

```bash
pip install torch torchvision scikit-learn matplotlib numpy pillow tqdm opencv-python grad-cam
```

For the Grad-CAM section, the notebook uses:

```python
from pytorch_grad_cam import GradCAM
```

## Running the Project

### 1. Open the notebook

Open:

```text
Brain_Tumor_Detection_Using_Pytorch_with_10_ml_models_.ipynb
```

The notebook is designed around Google Colab and mounts Google Drive.

### 2. Mount Google Drive

The notebook uses:

```python
from google.colab import drive

drive.mount('/content/drive')
```

### 3. Configure the dataset path

Update:

```python
dataset_path = '/content/drive/MyDrive/Brain Tumor Data Set'
```

so that it points to the directory containing the class folders.

### 4. Run data loading and preprocessing

The notebook:

- Loads images with `ImageFolder`
- Filters corrupted images
- Applies transformations
- Creates an 80/20 train/test split
- Creates PyTorch `DataLoader` objects

The baseline batch size is:

```python
batch_size=32
```

### 5. Train the models

Run the model sections in notebook order.

The deep-learning models generally use:

```python
nn.CrossEntropyLoss()
```

and Adam optimization:

```python
optim.Adam(model.parameters(), lr=0.001)
```

### 6. Compare model performance

The notebook collects available test accuracies and creates comparison plots.

It also contains plots for:

- Model test accuracy
- Training loss over epochs
- Test accuracy over epochs
- Individual model performance

## Reported Results in the Notebook

The notebook contains the following recorded results from executed cells:

| Model | Recorded result |
|---|---:|
| FFNN | 100.00% test accuracy |
| Random Forest | 100.00% test accuracy |
| ResNet18 | 100.00% test accuracy |
| LeNet | 100.00% test accuracy |
| EfficientNet-B0 | 100.00% test accuracy |
| SVM | Training failed because only one class was detected |
| VGGNet | Training section is commented/not included in the executed comparison |
| MobileNetV2 | Model section exists; no executed result is recorded in the inspected outputs |
| DenseNet121 | Model section exists; no executed result is recorded in the inspected outputs |
| Custom CNN | Later experiment achieved 97.50% |
| Advanced Custom CNN | Later experiment achieved 98.70% |

### Important interpretation note

The recorded 100% results should **not automatically be treated as evidence of real-world diagnostic performance**.

In the notebook's baseline data-loading output, `ImageFolder` reported:

```text
Classes: ['Brain Tumor Data Set']
```

That means the executed dataset structure was detected as **one class**, which also explains the SVM error:

```text
ValueError: The number of classes has to be greater than one; got 1 class
```

Before using the reported accuracy values as a meaningful binary classification evaluation, verify that the dataset is correctly organized into separate tumor and healthy class directories.

## Custom CNN

The Custom CNN uses convolutional layers with:

- ReLU activation
- Batch Normalization
- Max Pooling
- Fully connected layers
- Dropout

The baseline Custom CNN uses convolution channels:

```text
3 → 32 → 64 → 128
```

and a fully connected layer beginning with:

```text
128 × 8 × 8
```

## Advanced Custom CNN

The improved CNN increases the convolutional filters to:

```text
3 → 64 → 128 → 256
```

and increases the first fully connected layer to 512 neurons.

The advanced experiment also adds stronger data augmentation.

The notebook records a final test accuracy of:

```text
98.70%
```

for this advanced Custom CNN experiment.

## Grad-CAM Explainability

The project includes Grad-CAM to visualize which regions of an MRI image contributed most to the CNN prediction.

The notebook:

1. Loads the trained Custom CNN
2. Selects the final convolutional layer
3. Runs Grad-CAM
4. Generates an activation heatmap
5. Overlays the heatmap on the MRI image

The selected target layer is the final convolutional layer of the Custom CNN:

```python
target_layer = cnn_model.conv_layers[6]
```

Grad-CAM is useful because it provides a visual explanation of where the model is focusing rather than returning only a class label.

## Project Workflow

```text
MRI Dataset
     │
     ▼
ImageFolder
     │
     ▼
Image Validation
     │
     ▼
Resize + Tensor + Normalization
     │
     ▼
Train / Test Split
     │
     ├───────────────┐
     ▼               ▼
Traditional ML    Deep Learning
     │               │
 ┌───┴────┐      ┌───┴───────────────────────────────┐
 │        │      │                                   │
SVM   Random   FFNN  LeNet  CNN  ResNet  EfficientNet
Forest                       │          MobileNet
                             │          DenseNet
                             ▼
                         Grad-CAM
                             │
                             ▼
                    Visual Explanation
                             │
                             ▼
                    Model Comparison
```

## Project Structure

A recommended repository structure is:

```text
brain-tumor-detection/
│
├── README.md
├── Brain_Tumor_Detection_Using_Pytorch_with_10_ml_models_.ipynb
├── requirements.txt
│
├── dataset/
│   ├── Brain Tumor/
│   └── Healthy/
│
├── models/
│   └── trained_models/
│
├── results/
│   ├── accuracy_plots/
│   ├── loss_plots/
│   └── gradcam/
│
└── images/
    └── sample_mri_images/
```

## Key Technologies

| Technology | Purpose |
|---|---|
| Python | Programming language |
| PyTorch | Deep learning framework |
| Torchvision | Image processing and pretrained models |
| scikit-learn | SVM, Random Forest and evaluation |
| NumPy | Numerical processing |
| Matplotlib | Visualization |
| Pillow | Image loading |
| OpenCV | Image processing |
| Grad-CAM | Model explainability |
| Google Colab | Notebook execution |
| Google Drive | Dataset storage |

## Limitations

This project is an experimental machine-learning implementation and should not be considered a clinical diagnostic system.

Important limitations include:

- The dataset structure must be verified before interpreting accuracy.
- Accuracy alone is insufficient for medical image classification.
- The notebook does not establish clinical validity.
- A larger and independently validated dataset would be required for robust evaluation.
- Additional metrics such as precision, recall, F1-score, specificity, sensitivity, ROC-AUC, and confusion matrices should be considered.
- The 64×64 image resolution may discard useful medical-image detail.
- Results can vary depending on the train/test split and random seed.

## Recommended Improvements

For a stronger research implementation:

1. Correctly verify the class directory structure.
2. Use stratified train/validation/test splits.
3. Fix random seeds for reproducibility.
4. Add a validation set.
5. Report confusion matrices.
6. Report precision, recall, F1-score and ROC-AUC.
7. Evaluate sensitivity and specificity.
8. Use cross-validation where appropriate.
9. Test on an independent dataset.
10. Compare Grad-CAM explanations across correctly and incorrectly classified images.
11. Save trained model checkpoints.
12. Track experiments and hyperparameters.
13. Consider higher image resolutions when computational resources permit.

## Reproducibility

For reproducible experiments, set seeds for Python, NumPy and PyTorch:

```python
import random
import numpy as np
import torch

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
```

## Disclaimer

This project is intended for **educational and research purposes only**. It is not a substitute for professional medical diagnosis, radiological interpretation, or clinical decision-making.

## Source Notebook

The implementation and experiments described in this README are based on:

```text
Brain_Tumor_Detection_Using_Pytorch_with_10_ml_models_.ipynb
```
