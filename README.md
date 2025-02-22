# FitPredictor: Predicting BMIcase Using Machine Learning

## Overview
FitPredictor is a machine learning solution designed to predict BMIcase classifications using individual health metrics such as weight, height, BMI, Gender & Age.   
This project aims to provide personalized health insights to tackle the rising prevalence of non-communicable diseases.  
It leverages deep learning techniques along with a traditional ML baseline to empower individuals and healthcare providers.  
The dataset comprises 5,000 samples sourced from Kaggle.

## Dataset
The dataset used for this project is available on Kaggle:
[Fitness Exercises Using BFP and BMI](https://www.kaggle.com/datasets/mustafa20635/fitness-exercises-using-bfp-and-bmi)  
It includes features such as Weight, Height, BMI, Body Fat Percentage, Gender, Age and BMIcase, among others.

## Data Preprocessing & Feature Engineering
- **Missing Value Check:** Verified that there are no missing values.
- **Correlation Analysis:** A heatmap was generated to identify redundant features.
- **Feature Engineering:** Dropped highly correlated features (Body Fat Percentage, BFPcase, Exercise Recommendation Plan) and created a new feature (`BMI_to_Weight`).
- **Standardization & Encoding:** Standardized numerical features and label-encoded categorical variables (Gender and BMIcase).
- **Data Splitting:** Data split into 70% training, 15% validation, and 15% testing.
- **Class Imbalance Handling:** SMOTE was applied to balance the training data.

## Experimental Setup
I built a training pipeline using a custom function (`train_save_evaluate`) that trains models for 100 epochs, saves the trained model, predicts on the test set and computes evaluation metrics (Accuracy, F1-score, Recall, Precision and Loss). The following table summarizes five training instances.

| Instance    | Optimizer Used                  | Regularizer Used     | Epochs | Early Stopping | # Layers               | Learning Rate | Accuracy | F1 Score | Recall | Precision |
|-------------|---------------------------------|----------------------|--------|----------------|------------------------|---------------|----------|----------|--------|-----------|
| **Instance 1**<br>(Simple NN)  | Default (no explicit optimizer)  | None                 | 100    | No             | 3 (Basic architecture)   | Default       | 86.27%   | 85.87%   | 86.27% | 85.93%    |
| **Instance 2**<br>(Adam with L2)  | Adam                           | L2 (0.001)           | 100    | Yes (patience=10) | 4 (Dense + BN + Dropout)   | 0.0005        | **86.40%**   | **86.19%**   | **86.40%** | **86.17%**    |
| **Instance 3**<br>(RMSprop)    | RMSprop                        | L2 (0.001)           | 100    | No             | 4 (Dense + BN + Dropout)   | 0.0005        | 84.53%* | 84.84%* | 84.53%* | 85.86%*  |
| **Instance 4**<br>(SGD + Momentum) | SGD with Momentum              | L2 (0.001)           | 100    | Yes (patience=20) | 4 (Dense + BN + Dropout)   | 0.0005        | 85.60%   | 85.78%   | 85.60% | 86.17%    |
| **Instance 5**<br>(Logistic Regression) | Logistic Regression (ML baseline) | Class weight balancing; multinomial | N/A    | N/A            | N/A                    | N/A           | 82.40%   | 82.73%   | 82.40% | 83.63%    |

## Discussion of Findings
- **Neural Network vs. Logistic Regression:**  
  The deep learning models consistently outperform the logistic regression baseline, highlighting their ability to capture complex, non-linear relationships in the data. While logistic regression produced an excellent AUC (0.9820), its overall accuracy and F1-score were lower.

- **Optimizer Impact:**  
  - The **Simple NN** (Instance 1) serves as a robust baseline.
  - The **Adam optimizer** (Instance 2) with L2 regularization and EarlyStopping (patience=10) achieved the best performance, with 86.40% accuracy, by preventing overfitting and ensuring the best weights were restored.
  - The **RMSprop** (Instance 3) and **SGD with Momentum** (Instance 4) configurations yielded slightly lower performance. The SGD model, with a longer patience (20 epochs) to accommodate its slower convergence, still lagged behind the Adam configuration.

- **Early Stopping Role:**  
  EarlyStopping is critical in halting training when no further improvement is observed, thus preventing overfitting and reducing training time. Its application in both the Adam and SGD experiments ensured the best model weights were retained.

- **Best Combination:**  
  Overall, the Adam-based model (Instance 2) with L2 regularization and EarlyStopping is the most effective, outperforming other neural network configurations and the logistic regression baseline.

## ML Algorithm Hyperparameters (Logistic Regression)
- **Max Iterations:** 1000  
- **Class Weight:** Balanced  
- **Solver:** lbfgs  
- **Multi-class Strategy:** Multinomial

## Video Presentation
A video presentation (with the camera on) is included in this repository. In the presentation, I discuss the experimental table, explain the rationale behind the choice of optimizers and hyperparameters, and provide an in-depth error analysis of the models' performance.

