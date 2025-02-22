# FitPredictor: Predicting BMIcase Using Machine Learning

## Overview
FitPredictor is a machine learning solution designed to predict BMIcase classifications using individual health metrics such as weight, height, BMI, Gender & Age. 

This project addresses the rising prevalence of non-communicable diseases where obesity and undernutrition are growing concerns—by providing personalized and actionable health insights.  
Leveraging deep learning alongside traditional machine learning, FitPredictor aims to empower individuals and healthcare providers with data-driven results for improved health outcomes.  
The model demonstrates the practical application of ML in solving real-world health challenges, with potential to drive significant health impact among communities.

- **Motivation:**  
With a significant rise in non-communicable diseases especially in Rwanda, there is an urgent need for personalized health tools that go beyond generic BMI calculators. Current tools do not offer the detailed, actionable insights needed to drive preventive health care.  
FitPredictor aims to bridge the gap between health awareness and informed decision-making.

- **Problem Statement:**  
Many individuals in Rwanda lack awareness of their BMI classification and its associated health implications due to limited access to tailored health tools. Existing solutions provide only basic BMI values without context. FitPredictor addresses this gap by leveraging machine learning to deliver accurate and relevant BMI classifications along with health insights.

## Dataset
The dataset used for this project is available on Kaggle:
[Fitness Exercises Using BFP and BMI](https://www.kaggle.com/datasets/mustafa20635/fitness-exercises-using-bfp-and-bmi)  
- **Size:** 5000 samples with 9 features!

## Data Preprocessing & Feature Engineering
- **Missing Value Check:** Verified that there are no missing values.
- **Correlation Analysis:** A heatmap was generated to identify redundant features.
- **Feature Engineering:** Dropped highly correlated features (Body Fat Percentage, BFPcase, Exercise Recommendation Plan) and created a new feature (`BMI_to_Weight`).
- **Standardization & Encoding:** Standardized numerical features and label-encoded categorical variables (Gender and BMIcase).
- **Data Splitting:** Data split into 70% training, 15% validation, and 15% testing.
- **Class Imbalance Handling:** SMOTE was applied to balance the training data.

## Experimental Setup
I trained multiple neural network instances, each with different optimization techniques, hyperparameter tuning and regularization strategies. Below is a summary of the five training instances (Performance Comparison Table)

| Instance    | Optimizer Used                  | Regularizer Used     | Epochs | Early Stopping | # Layers               | Learning Rate | Accuracy | F1 Score | Recall | Precision |
|-------------|---------------------------------|----------------------|--------|----------------|------------------------|---------------|----------|----------|--------|-----------|
| **Instance 1**<br>(Simple NN)  | Default (no explicit optimizer)  | None                 | 100    | No             | 3 (Basic architecture)   | Default       | 86.27%   | 85.87%   | 86.27% | 85.93%    |
| **Instance 2**<br>(Adam with L2)  | Adam                           | L2 (0.001)           | 100    | Yes (patience=10) | 4 (Dense + BN + Dropout)   | 0.0005        | **86.40%**   | **86.19%**   | **86.40%** | **86.17%**    |
| **Instance 3**<br>(RMSprop)    | RMSprop                        | L2 (0.001)           | 100    | No             | 4 (Dense + BN + Dropout)   | 0.0005        | 84.53%* | 84.84%* | 84.53%* | 85.86%*  |
| **Instance 4**<br>(SGD + Momentum) | SGD with Momentum              | L2 (0.001)           | 100    | Yes (patience=20) | 4 (Dense + BN + Dropout)   | 0.0005        | 85.60%   | 85.78%   | 85.60% | 86.17%    |
| **Instance 5**<br>(Logistic Regression) | Logistic Regression (ML baseline) | Class weight balancing; multinomial | N/A    | N/A            | N/A                    | N/A           | 82.40%   | 82.73%   | 82.40% | 83.63%    |

## Findings
- **Neural Network vs the Logistic Regression**  
  The deep learning models consistently outperform the logistic regression, highlighting their ability to capture complex, non-linear relationships in the data while logistic regression produced an excellent AUC (0.9820), its overall accuracy and F1-score were lower.

- **Optimizer impact:**  
  - The **Simple NN** (Instance 1) serves as a robust baseline.
  - The **Adam optimizer** (Instance 2) with L2 regularization and EarlyStopping (patience=10) achieved the best performance with 86.40% accuracy by preventing overfitting and ensuring the best weights were restored.
  - The **RMSprop** (Instance 3) and **SGD with Momentum** (Instance 4) configurations yielded slightly lower performance. The SGD model, with a longer patience (20 epochs) to accommodate its slower convergence still lagged behind the Adam configuration.

- **Early Stopping Role:**  
  EarlyStopping is critical in halting training when no further improvement is observed, thus preventing overfitting and reducing training time. Its application in both the Adam and SGD experiments ensured the best model weights were retained.

- I noted that the Adam-based model (Instance 2) with L2 regularization and EarlyStopping is the most effective, outperforming other neural network configurations and the logistic regression Model.

## Logistic Regression Model
- **Max Iterations:** 1000  
- **Class Weight:** Balanced  
- **Solver:** lbfgs  
- **Multi-class Strategy:** Multinomial

## Video 
A video discussion of these results can be found here: **[Video link: ]**
