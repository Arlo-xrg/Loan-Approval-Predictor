import pandas as pd
import joblib

modle_addres  = "Models\my_model.pkl"
model = joblib.load(modle_addres)


income = int(input("Monthly income : "))         
credit_score = int(input("credit_score : "))   
loan_amount = int(input("loan_amount : "))     
work_exp = int(input("work_exp : "))        
has_default = int(input("has_default : "))      

dti_ratio = (loan_amount * 0.035) / income

applicant_data = pd.DataFrame([{
    'Monthly_Income_M': income,
    'Credit_Score': credit_score,
    'Loan_Amount_M': loan_amount,
    'Work_Experience_Years': work_exp,
    'Has_Prior_Default': has_default,
    'DTI_Ratio': dti_ratio
}])

prediction = model.predict(applicant_data)[0]
probability = model.predict_proba(applicant_data)[0][1] * 100

print("-----------------Answer-------------------")
if prediction == 1:
    print(f"The loan has been approved. {probability:.1f}%)")
else:
    print(f"The loan application was rejected. {probability:.1f}%)")
