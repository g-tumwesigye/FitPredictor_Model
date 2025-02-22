%%writefile README.md
# FitPredictor

## Overview

- **Problem Statement:**  
  The goal of FitPredictor is to classify individuals into BMI categories using physical attributes such as weight, height, and derived features.  
  We aim to improve predictive accuracy by experimenting with different neural network architectures and hyperparameters.  
  The project compares deep learning models with a traditional machine learning baseline.  
  Insights are drawn from multiple training instances to identify the best combination of hyperparameters.

- **Dataset Used:**  
  The dataset was obtained from Kaggle. You can find it here: [Fitness Exercises Using BFP and BMI](https://www.kaggle.com/datasets/mustafa20635/fitness-exercises-using-bfp-and-bmi).  
  It contains 5,000 samples with features including Weight, Height, BMI, Age, and other derived variables.

## Experiments & Findings

The following table summarizes the key training instances for the optimized neural network. Each instance varies in the choice of optimizer, regularizer, learning rate, and other hyperparameters. A machine learning baseline using Logistic Regression is also provided.

| Instance   | Optimizer Used                | Regularizer Used         | Epochs | Early Stopping | Number of Layers                           | Learning Rate | Accuracy | F1 Score | Recall | Precision |
|------------|-------------------------------|--------------------------|--------|----------------|--------------------------------------------|---------------|----------|----------|--------|-----------|
| Instance 1 | Default (no explicit optimizer) | None (default settings)  | 100    | No             | 3 (Simple NN architecture)                 | Default       | 0.8627   | 0.8587   | 0.8627 | 0.8593    |
| Instance 2 | Adam                          | L2 (λ = 0.001)           | 100    | No             | 4 (Dense + BatchNorm + Dropout layers)       | 0.0005        | 0.8640   | 0.8619   | 0.8640 | 0.8617    |
| Instance 3 | RMSprop                       | L2 (λ = 0.001)           | 100    | No             | 4 (Dense + BatchNorm + Dropout layers)       | 0.0005        | 0.8387   | 0.8429   | 0.8387 | 0.8539    |
| Instance 4 | SGD + Momentum (momentum=0.9) | L2 (λ = 0.001)           | 100    | No             | 4 (Dense + BatchNorm + Dropout layers)       | 0.0005        | 0.8440   | 0.8456   | 0.8440 | 0.8500    |
| Instance 5 | Logistic Regression (ML baseline) | Class weight balancing; solver=lbfgs, multi_class=multinomial | N/A  | N/A            | N/A                                        | N/A           | 0.8240   | 0.8273   | 0.8240 | 0.8363    |

*Note: Instance 5 represents the traditional ML approach used for comparison.*

## Summary of Findings

- **Best Neural Network Configuration:**  
  Among the neural network models, the configuration using the **Adam optimizer with L2 regularization (λ = 0.001) and a learning rate of 0.0005** achieved the highest performance (Accuracy: 86.40%).

- **Neural Network vs. ML Algorithm:**  
  The neural network implementations consistently outperformed the Logistic Regression baseline across all evaluation metrics. This indicates that a well-optimized deep learning model is better suited for this BMI classification task.

- **ML Algorithm Hyperparameters:**  
  The Logistic Regression model was tuned with a maximum of 1000 iterations, balanced class weights, and the multinomial setting with the lbfgs solver. Despite its strong AUC (0.9820), its overall accuracy (82.40%) was lower compared to the neural network models.

## Implementation Details

- **Data Preprocessing & Visualization:**  
  - Missing value checks and correlation analysis (heatmap).  
  - Feature engineering by dropping highly correlated features and creating the `BMI_to_Weight` ratio.  
  - Standardization of numerical features and label encoding of categorical features.  
  - Data split into training (70%), validation (15%), and test (15%) sets, with class imbalance addressed using SMOTE.

- **Model Architectures:**  
  - **Simple Neural Network:** A baseline NN without a specifically defined optimizer, using default settings.  
  - **Optimized Neural Networks:**  
    Configurations using Adam, RMSprop, and SGD with Momentum were tested. Each model integrated Batch Normalization and Dropout layers for improved generalization.  
  - **Logistic Regression:**  
    Used as a baseline machine learning model for comparison.

- **Training and Evaluation:**  
  Models were trained for 100 epochs, and performance was evaluated using accuracy, precision, recall, F1 score, ROC curves, and AUC score. Models were saved in the `/content/saved_models` directory.

- **Visualization:**  
  Confusion matrices, ROC curves, and training versus validation loss curves are generated to visualize model performance.

## Video Presentation

A video presentation is included in the repository. In the video, the presenter (with the camera on) discusses the neural network diagram, walks through the experimental table above, and explains how the different combinations of hyperparameters affect model performance.

Conclusion
The experiments demonstrate that a carefully tuned neural network (particularly with the Adam optimizer and L2 regularization) outperforms a baseline Logistic Regression model for BMI classification. The project highlights the importance of hyperparameter tuning in achieving optimal performance in deep learning models.

