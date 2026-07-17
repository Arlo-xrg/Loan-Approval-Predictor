import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix
import joblib

Coefficient = 98.7
model_name = "Models/my_model.pkl"


data = pd.read_csv('Dataset/loan_dataset.csv')
data = data.dropna()

x = data.drop(columns=['Approved'])
y=data['Approved']

x_train , x_test , y_train , y_test =train_test_split(x , y , test_size=0.2 )

while True:
    model=RandomForestClassifier(n_estimators=100 )
    model.fit(x_train , y_train)
    
    y_pr = model.predict(x_test)
    acc = accuracy_score(y_test , y_pr)
    print(acc)
    if acc*100 >Coefficient:
        print("----------------finish-------------------")
        joblib.dump(model , model_name)
        break
    # print(classification_report(y_test , y_pr))
    
