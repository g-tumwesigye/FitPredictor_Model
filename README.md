# FitPredictor 

## Problem Statement
FitPredictor aims to classify individuals into different BMI categories based on various physical attributes and recommend suitable exercise plans. The dataset used contains features like weight, height, BMI, body fat percentage, gender, and age. This project evaluates different optimization techniques to enhance classification accuracy and model performance.

## Dataset Used
- **Source:** [Kaggle - Fitness Exercises using BFP and BMI](https://www.kaggle.com/datasets/mustafa20635/fitness-exercises-using-bfp-and-bmi)
- **Size:** 5000 samples with 9 features

## Experimental Setup
I trained multiple neural network instances, each with different optimization techniques, hyperparameter tuning and regularization strategies. Below is a summary of the five training instances, including optimizers, regularization methods, learning rates, dropout rates, and evaluation metrics.

### Performance Comparison Table

| Training Instance | Optimizer Used | Regularizer Used | Epochs | Early Stopping | Number of Layers | Learning Rate | Accuracy | F1 Score | Recall | Precision |
|------------------|---------------|-----------------|--------|----------------|----------------|--------------|----------|----------|--------|-----------|
| Instance 1      | Default        | None            | 100    | No             | 3              | Default      | 86.40%   | 86.42%   | 86.40% | 86.49%    |
| Instance 2      | Adam           | L2 (0.001)      | 100    | Yes            | 4              | 0.0005       | 86.60%   | 86.70%   | 86.60% | 86.86%    |
| Instance 3      | RMSprop        | L2 (0.001)      | 100    | Yes            | 4              | 0.0005       | 87.10%   | 86.80%   | 87.10% | 86.74%    |
| Instance 4      | SGD + Momentum | L2 (0.001)      | 100    | Yes            | 4              | 0.0005       | 87.50%   | 87.66%   | 87.50% | 87.95%    |
| Instance 5      | AdamW          | L1 (0.001)      | 100    | Yes            | 5              | 0.0003       | 87.80%   | 87.95%   | 87.80% | 88.10%    |

## Findings 
- **Impact of optimization techniques:**
  - The baseline model (Instance 1) achieved an accuracy of **86.40%** using the default optimizer without early stopping or regularization.
  - The **Adam optimizer with L2 regularization (Instance 2)** showed slight improvement, increasing accuracy to **86.60%**.
  - **RMSprop (Instance 3)** further improved performance to **87.10%**, likely due to its adaptive learning rate capabilities.
  - **SGD with Momentum (Instance 4)** provided better generalization, yielding an accuracy of **87.50%**.
  - The **best performing model (Instance 5) used AdamW with L1 regularization**, reaching **87.80% accuracy**, benefiting from efficient weight decay and adaptive optimization.

- **Regularization and Learning Rate Impact:**
  - L2 regularization helped improve generalization in Adam and RMSprop models.
  - The combination of L1 regularization and AdamW in Instance 5 yielded the best performance.
  - Learning rates around **0.0005** worked well for Adam, RMSprop, and SGD, while AdamW performed better at **0.0003**.

## Machine Learning vs. Neural Network
- The logistic regression model achieved **83.30% accuracy**, lower than all neural network models.
- The neural network model's ability to capture non-linear relationships made it superior.
- Logistic regression worked well with class weights and balanced training but lacked deep feature representation power.
- Hyperparameter tuning in neural networks significantly enhanced performance.

## Conclusion
The **AdamW optimizer with L1 regularization** provided the best results. **SGD with Momentum** also performed well and had fewer computational requirements. Future improvements could involve hyperparameter tuning via GridSearch and experimenting with additional regularization techniques such as DropConnect and data augmentation strategies.

---

## Video 
A video discussion of these results can be found here: **[Video link: ]**

