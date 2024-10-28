# Loan-Eligibility-Prediction

# Loan Prediction System

This project aims to predict loan approvals based on applicant information. The dataset used contains various features such as gender, marital status, education, income, and more. Several classification models like Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbors are employed to analyze the data.

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

The dataset consists of information related to loan applicants. Key columns include:
- **Gender**: Male/Female
- **Married**: Yes/No
- **Dependents**: Number of dependents
- **Education**: Applicant's education level
- **Self_Employed**: Whether the applicant is self-employed
- **ApplicantIncome**: Applicant's income
- **LoanAmount**: Loan amount requested
- **Loan_Status**: Whether the loan was approved or not (target variable)

Download the dataset from the appropriate source or use the provided CSV file within the repository.

## Setup

To set up the project locally, follow these steps:
Install the required libraries:

bash
Copy code
pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn
Open the project in Jupyter Notebook or your preferred IDE.

## Usage
- Load the dataset:

df = pd.read_csv('/content/sample_data/Loan_data.csv')

-Preprocess the data:

  -Handle missing values.
  -Create new features like Total_Income.
  -Apply transformations like log scaling.
  - Perform label encoding on categorical variables.

df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
# Log transformation for skewed data
df['ApplicantIncomelog'] = np.log(df['ApplicantIncome'] + 1)

- Model training:

  -Split the dataset into training and testing sets.
  -Train classifiers such as Logistic Regression, Decision Tree, Random Forest, and KNN.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train Logistic Regression
model1 = LogisticRegression()
model1.fit(X_train, y_train)
Evaluate the models:

python
Copy code
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred_model1)
print("Logistic Regression Accuracy:", accuracy*100)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/loan-prediction-system.git
   cd loan-prediction-system
