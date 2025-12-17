# Diabetes Prediction ML Project

## Objective  
Predict whether a person has diabetes using physiological measurements and demographic data — useful for early detection and preventive healthcare.

---

## Dataset  
- *Name:* Pima Indians Diabetes Database  
- *Source:* Kaggle — [Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database)  
- *File used:* diabetes.csv  
- *Target column:* Outcome (0 = no diabetes, 1 = diabetes)  
- *Features:* Plasma glucose, Blood pressure, BMI, Skin thickness, Insulin, Age, etc.

---

## Project Steps  
1. Load data safely.  
2. Clean data: handle missing or invalid values (replace zeros in certain columns).  
3. Split data into features (X) and target (y).  
4. Train-test split: 80% train, 20% test.  
5. Train model: Random Forest Classifier.  
6. Evaluate model: accuracy, classification report, confusion matrix.  
7. Visualizations: Confusion matrix, feature importance.

---

## Plot Preview(Visualization)

Confusion Matrix

<br>
<br>
<img src="https://github.com/pratimadhende/Customer-Churn-Prediction/blob/ed704043d4240a57a2ae943ff89673902f6cf605/Confusion_matrix.png" alt="Image Description" width="600">
<br>
<br>
 Confusion Matrix
 <br>
 <br>
<img src="https://github.com/pratimadhende/Customer-Churn-Prediction/blob/ed704043d4240a57a2ae943ff89673902f6cf605/Confusion_matrix.png" alt="Image Description" width="600">
<br>
<br>
 Confusion Matrix
 <br>
 <br>
<img src="https://github.com/pratimadhende/Diabetes-Prediction/blob/adcf4adea7e42841707a76f166123ba3441c85eb/Confusion_matrix.png" alt="Image Description" width="600">
<br>
<br>
 Feature vs Outcome
 <br>
 <br>
<img src="https://github.com/pratimadhende/Diabetes-Prediction/blob/adcf4adea7e42841707a76f166123ba3441c85eb/Feature%20vs%20outcome.png" alt="Image Description" width="600">
<br>
<br>
 Outcome count plot
 <br>
 <br>
<img src="https://github.com/pratimadhende/Diabetes-Prediction/blob/adcf4adea7e42841707a76f166123ba3441c85eb/Outcome%20count%20plot(Target%20Distribution).png" alt="Image Description" width="600">
<br>
<br>
 Feature importance
 <br>
 <br>
<img src="https://github.com/pratimadhende/Diabetes-Prediction/blob/adcf4adea7e42841707a76f166123ba3441c85eb/feature_importance.png" alt="Image Description" width="600">
<br>
<br>
 Distribution Plot
 <br>
 <br>
<img src="https://github.com/pratimadhende/Diabetes-Prediction/blob/adcf4adea7e42841707a76f166123ba3441c85eb/histogram%20or%20distribution%20plot.png" alt="Image Description" width="600">

---

## Requirements  
- Python 3.x  
- pandas, numpy, matplotlib, seaborn, scikit-learn  

Install dependencies:

```bash

pip install pandas numpy matplotlib seaborn scikit-learn
