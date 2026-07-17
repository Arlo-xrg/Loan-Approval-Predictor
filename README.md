# 🏦 Loan Approval Predictor

An intelligent Loan Approval Prediction system built with **Python**, **pandas**, and **Scikit-Learn** using the **Random Forest Classifier** algorithm. The project uses applicant financial metrics to predict loan eligibility with high accuracy (>98%).

---

## 📌 Features

- **High Precision Training Loop:** Trains multiple Random Forest models dynamically until it achieves a target accuracy of over **98.7%** before exporting the model.
- **Automated Feature Engineering:** Calculates the **DTI (Debt-to-Income) Ratio** on the fly to boost classification precision.
- **Interactive CLI Interface:** Allows users to input financial details directly in the terminal and receive instant decision feedback along with confidence probabilities.
- **Model Persistence:** Uses `joblib` for easy model serialization and deployment.

---

## 📊 Dataset Features

The model evaluates applicants using 6 key financial attributes:

1. `Monthly_Income_M`: Monthly Income (in millions)
2. `Credit_Score`: Credit Score (ranging from 300 to 850)
3. `Loan_Amount_M`: Requested Loan Amount (in millions)
4. `Work_Experience_Years`: Total work experience (in years)
5. `Has_Prior_Default`: History of prior defaults (`0` = No, `1` = Yes)
6. `DTI_Ratio`: Calculated Debt-to-Income Ratio `(Loan_Amount * 0.035) / Monthly_Income`

---

## 🛠️ Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/Arlo-xrg/Loan-Approval-Predictor.git](https://github.com/Arlo-xrg/Loan-Approval-Predictor.git)
   cd Loan-Approval-Predictor
Install Dependencies:

Bash
pip install pandas scikit-learn joblib
🚀 Usage
1️⃣ Train and Save the Model
Run the training script to generate and save your high-accuracy model (my_model.pkl):

Bash
python create_model.py
Note: The script continuously evaluates Random Forest iterations until accuracy exceeds 98.7%, then automatically saves my_model.pkl.

2️⃣ Predict Loan Approval
Run the inference script to evaluate a new applicant via terminal prompts:

Bash
python predict.py
Example Terminal Output:

Plaintext
Monthly income : 60
credit_score : 720
loan_amount : 200
work_exp : 5
has_default : 0

-----------------Answer-------------------
The loan has been approved. (89.4%)
📁 Directory Structure
Plaintext
├── Dataset/
│   └── loan_dataset.csv          # Training dataset  
├── Models/
│   └── my_model.pkl              # Saved model binary
├── create_model.py               # Model training script
├── main.py                    # Inference script
└── README.md                     # Project documentation
🧠 Tech Stack
Language: Python 3.10+

Data Processing: pandas

Machine Learning: Scikit-Learn (Random Forest Classifier)

Model Export: Joblib
