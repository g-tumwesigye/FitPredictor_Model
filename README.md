# **FitPredictor (Predicting BMIcase)**

**Author:** Geofrey Tumwesigye  
**African Leadership University**

## **Overview**

FitPredictor is a machine learning solution designed to predict BMIcase classifications (e.g., normal, overweight, obese, underweight) using individual health metrics such as weight, height, BMI, body fat percentage, gender and age.  
This project addresses the rising prevalence of non-communicable diseases where obesity and undernutrition are growing concerns—by providing personalized and actionable health insights.  
Leveraging deep learning alongside traditional machine learning, FitPredictor aims to empower individuals and healthcare providers with data-driven results for improved health outcomes.  
The model demonstrates the practical application of ML in solving real-world health challenges, with potential to drive significant health impact among Rwandan communities.

## **Dataset**

- **Source:** The dataset was obtained from Kaggle.  
  [Fitness Exercises Using BFP and BMI](https://www.kaggle.com/datasets/mustafa20635/fitness-exercises-using-bfp-and-bmi)
- **Description:** Contains 5,000 samples with features including weight, height, BMI, body fat percentage, gender, age, and derived features for BMI classification.

## Project Motivation & Problem Statement

- **Motivation:**  
  With a significant rise in non-communicable diseases in Rwanda, there is an urgent need for personalized health tools that go beyond generic BMI calculators. Current tools do not offer the detailed, actionable insights needed to drive preventive health care.  
  FitPredictor was inspired by Rwanda’s growing health challenges and aims to bridge the gap between health awareness and informed decision-making.

- **Problem Statement:**  
  Many individuals in Rwanda lack awareness of their BMI classification and its associated health implications due to limited access to tailored health tools. Existing solutions provide only basic BMI values without context or personalized recommendations. FitPredictor addresses this gap by leveraging machine learning to deliver accurate and relevant BMI classifications along with health insights.

## Findings

The following table summarizes key training instances.

| Instance   | Optimizer Used                | Regularizer Used         | Epochs | Early Stopping | Number of Layers                           | Learning Rate | Accuracy | F1 Score | Recall | Precision |
|------------|-------------------------------|--------------------------|--------|----------------|--------------------------------------------|---------------|----------|----------|--------|-----------|
| Instance 1 | Default                         | None                     | 100    | No             | 3 (Simple NN architecture)                 | Default       | 0.8627   | 0.8587   | 0.8627 | 0.8593    |
| Instance 2 | Adam                          | L2                       | 100    | No             | 4 (Dense + BatchNorm + Dropout layers)       | 0.0005        | 0.8640   | 0.8619   | 0.8640 | 0.8617    |
| Instance 3 | RMSprop                       | L2                       | 100    | No             | 4 (Dense + BatchNorm + Dropout layers)       | 0.0005        | 0.8387   | 0.8429   | 0.8387 | 0.8539    |
| Instance 4 | SGD + Momentum (momentum=0.9) | L2                       | 100    | No             | 4 (Dense + BatchNorm + Dropout layers)       | 0.0005        | 0.8440   | 0.8456   | 0.8440 | 0.8500    |
| Instance 5 | Logistic Regression               | Class weight balancing; solver=lbfgs, multi_class=multinomial | N/A  | N/A            | N/A                                        | N/A           | 0.8240   | 0.8273   | 0.8240 | 0.8363    |


## Summary of Findings

- **Best Neural Network Configuration:**  
  The neural network using the **Adam optimizer with L2 regularization and a learning rate of 0.0005** achieved the highest performance with an accuracy of 86.40%.
  
- **Comparative Performance:**  
  The deep learning models consistently outperformed the Logistic Regression baseline, underscoring the importance of tailored hyperparameter tuning and network design for BMI classification tasks.
  
- **ML Baseline Details:**  
  The Logistic Regression model was configured with 1000 iterations, balanced class weight and a multinomial setting using the lbfgs solver. Despite a high AUC, its overall accuracy was lower compared to the neural network models.


- **Model Architectures:**  
  - **Simple Neural Network:** A baseline architecture using default settings.  
  - **Optimized Neural Networks:**  
    Models employing Adam, RMSprop and SGD with Momentum optimizers were experimented with, each featuring Batch Normalization and Dropout for improved generalization.  
  - **Logistic Regression:**  
    Served as the ML baseline for comparative evaluation.

- **Training & Evaluation:**  
  Models were trained for 100 epochs and evaluated using metrics such of accuracy, precision, recall, F1 score, ROC curves and AUC score. Models are in the saved_models` directory.

- **Visualization:**  
  Confusion matrices, ROC curves and training versus validation loss curves were generated to illustrate model performance.

## Video 
A video discussion of these results can be found here: **[Video link: ]**
![image](https://github.com/user-attachments/assets/390fa9ea-5cf8-4c86-9053-d798bfe177e4)


