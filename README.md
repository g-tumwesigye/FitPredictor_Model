FitPredictor
Overview
Problem Statement:
The goal of FitPredictor is to classify individuals into BMI categories using physical attributes such as weight, height, and derived features.
We aim to improve predictive accuracy by experimenting with different neural network architectures and hyperparameters.
The project compares deep learning models with a traditional machine learning baseline.
Insights are drawn from multiple training instances to identify the best combination of hyperparameters.

Dataset Used:
The dataset was obtained from Kaggle. You can find it here: Fitness Exercises Using BFP and BMI.
It contains 5,000 samples with features including Weight, Height, BMI, Age, and other derived variables.

Experiments & Findings
The following table summarizes the key training instances for the optimized neural network. Each instance varies in the choice of optimizer, regularizer, learning rate, and other hyperparameters. A machine learning baseline using Logistic Regression is also provided.

Instance	Optimizer Used	Regularizer Used	Epochs	Early Stopping	Number of Layers	Learning Rate	Accuracy	F1 Score	Recall	Precision
Instance 1	Default (no explicit optimizer)	None (default settings)	100	No	3 (Simple NN architecture)	Default	0.8627	0.8587	0.8627	0.8593
Instance 2	Adam	L2 (λ = 0.001)	100	No	4 (Dense + BatchNorm + Dropout layers)	0.0005	0.8640	0.8619	0.8640	0.8617
Instance 3	RMSprop	L2 (λ = 0.001)	100	No	4 (Dense + BatchNorm + Dropout layers)	0.0005	0.8387	0.8429	0.8387	0.8539
Instance 4	SGD + Momentum (momentum=0.9)	L2 (λ = 0.001)	100	No	4 (Dense + BatchNorm + Dropout layers)	0.0005	0.8440	0.8456	0.8440	0.8500
Instance 5	Logistic Regression (ML baseline)	Class weight balancing; solver=lbfgs, multi_class=multinomial	N/A	N/A	N/A	N/A	0.8240	0.8273	0.8240	0.8363
Note: Instance 5 represents the traditional ML approach used for comparison.

Summary of Findings
Best Neural Network Configuration:
Among the neural network models, the configuration using the Adam optimizer with L2 regularization (λ = 0.001) and a learning rate of 0.0005 achieved the highest performance (Accuracy: 86.40%).

Neural Network vs. ML Algorithm:
The neural network implementations consistently outperformed the Logistic Regression baseline across all evaluation metrics. This indicates that a well-optimized deep learning model is better suited for this BMI classification task.

ML Algorithm Hyperparameters:
The Logistic Regression model was tuned with a maximum of 1000 iterations, balanced class weights, and the multinomial setting with the lbfgs solver. Despite its strong AUC (0.9820), its overall accuracy (82.40%) was lower compared to the neural network models.
