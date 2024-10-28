# Loan Prediction System

This project aims to predict loan approvals based on applicant information using several classification models like Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbors. The dataset includes features like gender, marital status, education, income, and more. 

## Table of Contents
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Setup](#setup)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [License](#license)

## Technologies Used

- Python 3.x
- Pandas
- Numpy
- Seaborn
- Matplotlib
- Scikit-learn
- Imbalanced-learn

## Dataset

The dataset contains loan application details and is used to train various models to predict whether a loan will be approved or not. The key features include:

- **Gender**: Applicant's gender (Male/Female)
- **Married**: Applicant's marital status (Yes/No)
- **Dependents**: Number of dependents
- **Education**: Applicant's education level (Graduate/Non-Graduate)
- **Self_Employed**: Whether the applicant is self-employed (Yes/No)
- **ApplicantIncome**: Applicant's income
- **LoanAmount**: Loan amount requested
- **Loan_Status**: Whether the loan was approved (Y/N) [Target Variable]

Ensure the dataset is downloaded and placed in the project directory.

## Setup

To set up the project locally, follow these steps:


## Usage
- Preprocess the data:

  - Load the dataset using Pandas.
  - Handle missing values by using median and mode imputation for numerical and categorical features.
  - Apply log transformation for skewed features like LoanAmount, TotalIncome, and others.
  - Encode categorical features using LabelEncoder.
  - Split the dataset into training and testing sets (e.g., 75% training, 25% testing).

- Train different models:
  
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - K-Nearest Neighbors
  - 
- Run the models: Each model is trained on the training dataset, and predictions are made on the test dataset. You can also apply cross-validation to check the generalization of the models.
  
- Use oversampling to handle imbalance: If your dataset is imbalanced, apply RandomOverSampler from imbalanced-learn to balance the dataset before training.

## Model Evaluation
We evaluate the performance of the models using the following metrics:

- Accuracy: Measures the proportion of correctly predicted instances out of the total instances.

- Confusion Matrix: A table used to describe the performance of a classification model by showing the correct and incorrect predictions made.

- Learning Curve: This helps to understand how the model improves with more training data and whether it is underfitting or overfitting.

- Feature Importance: For models like Decision Trees and Random Forests, feature importance shows which features were most influential in predicting the loan status.

## Results
After training the models and evaluating their performance, the results (accuracy and other metrics) are displayed for each model. Below are the models tested:

- Logistic Regression: Provides a baseline performance with reasonable accuracy.
  ![image](https://github.com/user-attachments/assets/577f401f-cd02-4721-b468-ef62dbf1b2a0)
  
- Decision Tree Classifier: Performs well but can sometimes overfit the data.
  ![image](https://github.com/user-attachments/assets/57ddab2a-dc5e-4a2d-a55e-ff820ad32815)

- Random Forest Classifier: Provides robust performance by averaging multiple decision trees.
  ![image](https://github.com/user-attachments/assets/98af9c02-6bce-4e5e-9295-068c3f9eca0c)

- K-Nearest Neighbors: Simpler model but might not perform well with imbalanced datasets.
  ![image](https://github.com/user-attachments/assets/6ba9c5c6-6dfe-45f5-bcdc-f4b6d0097474)

## LICENSE
This project is licensed under the MIT License - see the LICENSE file for details.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/loan-prediction-system.git
   cd loan-prediction-system
