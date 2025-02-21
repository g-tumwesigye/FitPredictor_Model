# FitPredictor 

## Project Overview
This project aims to classify individuals's BMI cases into different categories using a dataset containing individuals' health metrics.I implemented both **Machine Learning (ML) algorithms** and **Neural Networks (NNs)** to determine the most effective classification method. I tested different optimization techniques, regularization methods and training strategies to optimize performance.

## Dataset 
The dataset consists of features such as **Weight, Height, BMI, Age, Gender and BMI-to-Weight ratio**. The target variable is **BMIcase** which categorizes individuals BMI cases into seven classes: **Severe Thinness, Moderate Thinness, Mild Thinness, Normal, Overweight, Obese, and Severe Obese.**

**Dataset Source:** [FitPredictor Dataset on Kaggle](https://www.kaggle.com/datasets/mustafa20635/fitness-exercises-using-bfp-and-bmi)

## Model Performance & Findings

### Comparison of Model Results
| Model | Accuracy | Precision | Recall | F1 Score | AUC Score |
|--------|------------|------------|--------|---------|----------|
| Simple Neural Network | 0.8680 | 0.8680 | 0.8680 | 0.8674 | 0.9922 |
| Logistic Regression | 0.8330 | 0.8412 | 0.8330 | 0.8353 | 0.9837 |
| Adam Optimizer Model | 0.8730 | 0.8736 | 0.8730 | 0.8726 | 0.9931 |
| RMSprop Optimizer Model | 0.8720 | 0.8702 | 0.8720 | 0.8709 | N/A |
| SGD + Momentum Optimizer Model | 0.8660 | 0.8685 | 0.8660 | 0.8669 | N/A |

---
## Summary of Results

### Which Combination Worked Best?
The **Adam Optimizer Model** performed best with an **accuracy of 0.8730**, a **high F1-score of 0.8726**, and **an AUC Score of 0.9931**. This indicates **strong classification ability**, likely due to Adam's effective adaptive learning rate adjustments.

### ML vs. Neural Network Performance
- The **Neural Network models** outperformed **Logistic Regression** across all metrics.
- Logistic Regression had an **accuracy of 0.8330**, which is **lower than all neural network models**.
- **Adam and RMSprop optimizers performed better** than the other NN models, demonstrating the importance of optimizer selection.

### Logistic Regression Hyperparameters Used
- **Solver:** lbfgs
- **Max Iterations:** 1000
- **Class Weight:** Balanced
- **Multi-class Setting:** Multinomial

---
## Training Adjustments & Results Table
The table below documents various **training instances**, including changes in optimization settings and their impact on model performance.

| Training Instance | Optimizer Used | Regularizer | Epochs | Early Stopping | No. of Layers | Learning Rate | Accuracy | F1 Score | Recall | Precision | Loss |
|------------------|---------------|-------------|--------|---------------|---------------|--------------|----------|---------|--------|----------|------|
| Instance 1 | Default (TF) | None | 100 | No | 3 | Default | 0.8600 | 0.8590 | 0.8600 | 0.8612 | 0.412 |
| Instance 2 | Adam | L2 (0.001) | 100 | Yes | 4 | 0.0005 | 0.8730 | 0.8726 | 0.8730 | 0.8736 | 0.367 |
| Instance 3 | RMSprop | L2 (0.001) | 100 | Yes | 4 | 0.0005 | 0.8720 | 0.8709 | 0.8720 | 0.8702 | 0.371 |
| Instance 4 | SGD + Momentum | L2 (0.001) | 100 | Yes | 4 | 0.0005 | 0.8660 | 0.8669 | 0.8660 | 0.8685 | 0.382 |
| Instance 5 | Nadam | L2 (0.002) | 120 | No | 5 | 0.0003 | TBD | TBD | TBD | TBD | TBD |

**TBD** = Training not yet completed.

---
## Discussion of Findings
Each optimization technique influenced model performance differently:
- **Instance 1:** Default settings showed lower accuracy, likely due to the lack of fine-tuned optimization.
- **Instance 2 (Adam):** Performed best due to adaptive learning rate and L2 regularization controlling overfitting.
- **Instance 3 (RMSprop):** Showed strong performance but slightly lower than Adam, indicating it benefits from a more fine-tuned learning rate.
- **Instance 4 (SGD + Momentum):** Performed well, but its stability issues may explain why it lagged slightly behind Adam and RMSprop.
- **Instance 5 (Nadam - TBD):** Expected to combine the benefits of both Adam and RMSprop.

---
## Final Thoughts
1. **Adam optimizer performed best, followed by RMSprop.**  
2. **Neural Networks significantly outperformed Logistic Regression.**  
3. **Early stopping helped avoid overfitting, while L2 regularization improved stability.**

This project highlights the importance of optimizer selection, regularization, and hyperparameter tuning in improving classification performance.


