# Diabetes Prediction ML Project

## Objective  
Predict whether a person has diabetes using physiological measurements and demographic data — useful for early detection and preventive healthcare.

## Dataset  
- *Name:* Pima Indians Diabetes Database  
- *Source:* Kaggle — [Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database)  
- *File used:* diabetes.csv  
- *Target column:* Outcome (0 = no diabetes, 1 = diabetes)  
- *Features:* Plasma glucose, Blood pressure, BMI, Skin thickness, Insulin, Age, etc.

## Project Steps  
1. Load data safely.  
2. Clean data: handle missing or invalid values (replace zeros in certain columns).  
3. Split data into features (X) and target (y).  
4. Train-test split: 80% train, 20% test.  
5. Train model: Random Forest Classifier.  
6. Evaluate model: accuracy, classification report, confusion matrix.  
7. Visualizations: Confusion matrix, feature importance.

## Requirements  
- Python 3.x  
- pandas, numpy, matplotlib, seaborn, scikit-learn  

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn