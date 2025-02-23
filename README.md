# FitPredictor (Predicting BMIcase)

## Overview
FitPredictor is a machine learning solution designed to predict BMIcase classifications using individual health metrics such as weight, height, BMI, gender, and age.  
This project addresses the rising prevalence of non-communicable diseases—where obesity and undernutrition are growing concerns by providing personalized health insights.  
FitPredictor empowers both individuals and healthcare providers with data-driven results to improve health outcomes.  

**Problem Statement**  
There is a sharp increase in non-communicable diseases. Traditional BMI calculators provide only basic values without the context needed for effective preventive healthcare. Many individuals lack awareness of their BMI classification and the associated health implications because current tools are generic and fail to offer tailored insights. FitPredictor leverages machine learning to deliver accurate BMI classifications and personalized health guidance thus bridging this gap by delivering detailed BMI classifications and personalized health insights.

## Dataset
The dataset for this project is available on Kaggle:  
[Fitness Exercises Using BFP and BMI](https://www.kaggle.com/datasets/mustafa20635/fitness-exercises-using-bfp-and-bmi)  
- **Size:** 5000 samples with 9 features

## Data Preprocessing & Feature Engineering
- **Missing values check:** I confirmed that there were no missing values.
- **Correlation analysis:** Generated a heatmap to identify redundant features.
- **Feature engineering:** I dropped highly correlated features (Body Fat Percentage, BFPcase and Exercise Recommendation Plan) and created a new feature (**BMI_to_Weight**).
- **Standardization & Encoding:** I standardized numerical features and label-encoded categorical variables (Gender and BMIcase).
- **Data Splitting:** I split data into 70% training, 15% validation and 15% testing.
- **Class imbalance handling:** SMOTE was applied to handle the class imbalance.

## Comparison & performance evaluation
Multiple neural network instances were trained using various optimization and regularization stratefies & also Logistic regression. Their performance is summarized in the table below:

| **Instance**                                      | **Optimizer Used**            | **Regularizer Used** | **Epochs** | **Early Stopping**         |  **Layers**         | **Learning Rate** | **Accuracy** | **F1 Score** | **Recall** | **Precision** |
|---------------------------------------------------|-------------------------------|----------------------|------------|----------------------------|----------------------|-------------------|--------------|--------------|------------|---------------|
| **Instance 1 (Simple NN)**                        | Default (no optomizer)          | None                 | 100        | No                         | 3 (Dense-only)       | Default           | 86.53%       | 86.46%       | 86.53%     | 86.44%        |
| **Instance 2 (Adam with L2)**                     | Adam                          | L2 (0.001)           | 100        | Yes (patience=10)          | 4 (Dense+BN+Dropout) | 0.0005            | 85.60%       | 85.78%       | 85.60%     | 85.62%        |
| **Instance 3 (RMSprop)**                          | RMSprop                       | L2 (0.001)           | 100        | No                         | 4 (Dense+BN+Dropout) | 0.0005            | 84.13%       | 84.84%       | 84.13%     | 85.86%        |
| **Instance 4 (SGD + Momentum)**                   | SGD + Momentum                | L2 (0.001)           | 100        | Yes (patience=20)          | 4 (Dense+BN+Dropout) | 0.0005            | **87.47%**   | **87.55%**   | **87.47%** | **87.76%**    |
| **Instance 5 (Logistic Regression - Baseline)**   | Logistic Regression           | (Class weight balancing) |   -    |   -                        |   -                  |   -               | 82.40%       | 82.73%       | 82.40%     | 83.63%        |

## Summary of findings
  - Neural network models outperformed Logistic Regression in overall accuracy and F1 score though Logistic Regression had an excellent ROC AUC.   
  - The **Simple NN** showed robust performance with no optimizer.  
  - The **Adam model** (instance 2) with L2 and EarlyStopping (patience=10) performed well but was outperformed by SGD+mpmentum with L2.  
  - The **RMSprop model** (instance 3) lagged behind.  
  - The **SGD + Momentum model** (instance 4) achieved the highest accuracy of 87.47% and best F1 score, recall and precision, making it the best model. The SGD + Momentum model with L2 regularization and EarlyStopping (patience = 20) delivered the best overall performance.
  - EarlyStopping helped to stop training when validation loss stopped improving & this prevented overfitting and ensured that the best weights are retained.

## Hyperparameters of the Logistic Regression Model
- **Max iterations:** 1000  
- **Class weight:** Balanced in oredr to address the class imbalance.  
- **Solver:** lbfgs  
- **Multi-class:** Multinomial
The hyperparameters ensured that the model converged reliably

## How to run the notebook,load & evaluatee the best saved model

1. **Running the Notebook:**
   - Open the notebook in Google Colab.
   - Make sure you have uploaded the dataset file (e.g., `fitpredictor_dataset.csv`) to your Colab environment.
   - Run all cells sequentially. You can click on **Runtime > Run all** to execute the notebook from start to finish.
   - The notebook will automatically preprocess the data, train multiple models using various optimization and regularization techniques and save each trained model in the `/content/saved_models` directory.

2. **Loading and evaluating the best model:**
   - After running all cells, review the comparison table printed in the notebook to determine the best performing model based on the evaluation metrics.
   - The notebook is set up to save the best model and in my case, the SGD + Momentum model as `/content/saved_models/sgd_momentum_model.h5`.
   - To load the best model and make predictions on the test set, run the following code cell at the end of the notebook:

   ```python
   from tensorflow.keras.models import load_model
   from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
   import matplotlib.pyplot as plt
   import seaborn as sns
   import numpy as np

   best_model_path = "/content/saved_models/sgd_momentum_model.h5"
   model = load_model(best_model_path)
   pred_probs = model.predict(X_test)
   pred_classes = np.argmax(pred_probs, axis=1)
   true_classes = np.argmax(y_test, axis=1)

   acc = accuracy_score(true_classes, pred_classes)
   prec = precision_score(true_classes, pred_classes, average='weighted')
   rec = recall_score(true_classes, pred_classes, average='weighted')
   f1_val = f1_score(true_classes, pred_classes, average='weighted')
   roc_auc = roc_auc_score(y_test, pred_probs, multi_class='ovr', average='weighted')

   print("Accuracy: {:.4f}".format(acc))
   print("Precision: {:.4f}".format(prec))
   print("Recall: {:.4f}".format(rec))
   print("F1 Score: {:.4f}".format(f1_val))
   print("ROC AUC: {:.4f}".format(roc_auc))

   cm_matrix = confusion_matrix(true_classes, pred_classes)
   print("Confusion Matrix:")
   print(cm_matrix)

   plt.figure(figsize=(8,6))
   sns.heatmap(cm_matrix, annot=True, fmt="d", cmap="Blues",
               xticklabels=np.unique(true_classes), yticklabels=np.unique(true_classes))
   plt.xlabel("Predicted Labels")
   plt.ylabel("True Labels")
   plt.title("Confusion Matrix - Best Model")
   plt.show()
   
- Running the above cell will load the best saved model, make predictions on the test set, compute evaluation metrics (accuracy, precision, recall, F1 score and ROC AUC) and display the confusion matrix along with the loss curves.

## Video 
A video discussion of these results can be found here: **[Video link: https://drive.google.com/file/d/1X7hcpUp5ZXKMHcypwt0VrA8SdfD-F5gj/view?usp=drive_link]**
