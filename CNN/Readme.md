# Pneumonia Detection from Chest X-Rays using Custom CNN

## Problem Statement
Build a deep learning model to classify chest X-ray images into **NORMAL** or **PNEUMONIA** using a custom convolutional neural network architecture. The task requires learning spatial hierarchical features from grayscale medical images while handling class imbalance and achieving high generalization performance.

---

## Dataset Source
Kaggle – Chest X-Ray Images (Pneumonia)  
[Link](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

**Aspect Details**:
- **Classes**: 2 (NORMAL, PNEUMONIA)
- **Train images**: 5,216 (after merging original tiny val set)
- **Test images**: 624
- **Image size**: Resized to 224 × 224 grayscale
- **Class distribution**: ~26% NORMAL, ~74% PNEUMONIA (imbalanced)
- **Preprocessing**: Grayscale, resize, augmentation (rotation, flip, affine), normalization (mean=0.485, std=0.229)

**Data split strategy**:
- Original tiny val folder (16 images) merged into train
- Final train → stratified 80/20 split → new train (~4,173) / val (~1,043)
- Official test set used only for final evaluation

---

## Why This Dataset?
- Real-world class imbalance challenge
- Grayscale X-rays → suitable for lightweight custom CNNs
- Clinically relevant task (early pneumonia detection)

---

## Model architecture diagram

[Baseline CNN](https://github.com/vaibhavreddy0226/ACM-tasks-/blob/main/CNN/Baseline_CNN_Diagram.jpeg)  
[Custom CNN](https://github.com/vaibhavreddy0226/ACM-tasks-/blob/main/CNN/Custom_CNN_Diagram.jpeg)

---

## Training strategy and setup

**Baseline model**
- Simple 3-layer CNN
- Layers: 3 Conv blocks (32 → 64 → 128) + MaxPool + Flatten → FC(128) → Dropout(0.5) → 2 classes
- Parameters: ~12.9M
- Regularization: Dropout 0.5 in final FC layer only

**Custom model – PneumoNet**  
**Overall style**   
Deep VGG-inspired architecture  
**Layer-by-layer breakdown & simple justification**

| Block | Output Channels | Layers inside the block                                      | Why we chose this (simple explanation)                                                                 |
|-------|------------------|----------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| 1     | 32               | 2 × Conv(3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout 0.1 | Early layers detect basic edges and textures. Low channels keep it fast and light. Small dropout prevents early overfitting. |
| 2     | 64               | 2 × Conv(3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout 0.2 | Doubling channels lets the model learn more complex patterns. Slightly higher dropout as features get richer and more prone to noise. |
| 3     | 128              | **3** × Conv(3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout 0.3 | Middle layers combine simple features into shapes (lung edges, small opacities). Extra conv layer helps refine features — common in deeper VGG models. |
| 4     | 256              | 2 × Conv(3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout 0.4 | Deeper layers focus on higher-level patterns (consolidations, ground-glass areas). Strong dropout protects against memorizing training noise. |
| 5     | 512              | 2 × Conv(3×3) → BatchNorm → ReLU → **Global Average Pooling** | Very deep features capture the most abstract concepts. Global Average Pooling replaces huge flatten + FC layers → saves millions of parameters and greatly reduces overfitting (widely used in modern CNNs like ResNet, EfficientNet, MobileNet). |

**Final classification head**  
- After Global Average Pooling: 512 features  
- Linear(512 → 256) → BatchNorm → ReLU → Dropout(0.5)  
- Linear(256 → 2 classes)  

**Why this head?**  
One hidden layer helps mix features non-linearly before the final decision. BatchNorm + high dropout strongly regularizes the classification step.

**Key design choices – explained simply**

- **Why 3×3 kernels everywhere?**  
  Small 3×3 filters are very efficient and powerful. Stacking many of them gives a large receptive field with more non-linearity and fewer parameters than big kernels. This has been the standard since VGGNet (2014) and is used in almost every successful modern CNN.

- **Why channels double each time (32 → 64 → 128 → 256 → 512)?**  
  Classic pattern from VGG and most CNNs. Early layers need fewer filters for simple features; deeper layers need many more to capture complex medical patterns.

- **Why extra conv in block 3?**  
  Deeper layers benefit from more processing before pooling. This is exactly how VGG-16 and VGG-19 work — extra convs help refine features.

- **Why BatchNorm after every conv?**  
  Makes training much more stable, allows higher learning rates, and reduces problems with shifting activations. Almost every modern CNN uses it.

- **Why increasing Dropout (0.1 → 0.4) in conv blocks?**  
  Early layers have simple features → less regularization needed. Deeper layers have very powerful features → higher risk of overfitting → stronger dropout.

- **Why Global Average Pooling instead of big fully-connected layers?**  
  Drastically reduces parameters (from millions to almost none in that step), lowers overfitting, and improves generalization. This technique became popular after Network in Network (2013) and is now standard in ResNet, EfficientNet, MobileNet, and many medical imaging models.

---
## Training Setup
- **Framework**: PyTorch
- **Loss function**: CrossEntropyLoss (with class weights)
- **Optimizer**: Adam (learning rate = 0.001)
- **Batch size**: 32
- **Number of epochs**: 20
- **Data loaders**: `torch.utils.data.DataLoader` (shuffle only on train)
- **Evaluation metric**: Accuracy, Precision, Recall, F1 on validation and test sets

### Results Snapshot (from training logs)

| Model       | Best Val Accuracy | Final Val Accuracy | Test Accuracy | Parameters |
|-------------|-------------------|---------------------|---------------|------------|
| BaselineCNN | 95.22%            | 95.22%             | ~86.54%       | 12.9M      |
| CustomCNN   | 96.47%            | 94.75%             | ~87.34%       | 5.0M       |

---

## Deliverables

### Model architecture diagram
See links above.

### Parameter count and efficiency analysis
- BaselineCNN: 12,938,114 parameters
- PneumoNet: 4,995,810 parameters
- **Efficiency gain**: PneumoNet is **2.59× smaller** (thanks to Global Average Pooling replacing large FC layers)

### Training and validation curves  
**BaseLine train and validation curves**  
![Baseline_curves](https://github.com/vaibhavreddy0226/ACM-tasks-/blob/main/CNN/Baseline_curves.png)  

**BaseLine train and validation curves**  
![customCNN_curves](https://github.com/vaibhavreddy0226/ACM-tasks-/blob/main/CNN/custom_curves.png)

### Confusion matrix
![Baseline Confusion Matrix](https://github.com/vaibhavreddy0226/ACM-tasks-/blob/main/CNN/Baseline_confusion_matrix.png)  
![CustomCNN Confusion Matrix](https://github.com/vaibhavreddy0226/ACM-tasks-/blob/main/CNN/Custom_confusion_matrix.png)

### Error visualization samples
![Baseline Misclassifications](https://github.com/vaibhavreddy0226/ACM-tasks-/blob/main/CNN/Baseline_missclassification.png)  
![CustomCNN Misclassifications](https://github.com/vaibhavreddy0226/ACM-tasks-/blob/main/CNN/Custom_missclassification.png)

