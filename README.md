# Loan Approval Prediction

A Streamlit app that predicts whether a loan application will be approved,
using a random forest trained on the classic loan-prediction dataset
(614 labelled applications). Built for learning purposes during my M.S. in
Analytics at Dakota State University (DSU).

## How it works

The notebook [`loan_prediction.ipynb`](loan_prediction.ipynb) covers the full
workflow:

1. **Exploration** of categorical (gender, marital status, education,
   property area…) and numerical (income, loan amount, term) features
2. **Missing values** filled in (mode for categorical columns and credit
   history, median for loan amount and term)
3. **One-hot encoding** of categorical variables, with the target `Loan_Status`
   mapped Y → 1, N → 0
4. **Model:** `RandomForestClassifier(n_estimators=1000, max_depth=5,
   max_features=15)` trained on an 85/15 split
5. **Result:** **87.1% accuracy** on the held-out 15%
6. The trained model is saved to `RF.sav` and loaded by the app

## Run the app

```bash
pip install -r requirements.txt
streamlit run run1.py
```

Choose **Prediction** in the sidebar, enter the applicant's details, and press
**Predict**.

`RF.sav` was saved with scikit-learn 1.4.1, so `requirements.txt` pins that
version. Newer versions warn about the mismatch and may fail to load the model.

## Files

| File | Purpose |
|---|---|
| `run1.py` | Streamlit app |
| `loan_prediction.ipynb` | Data exploration, preprocessing and model training |
| `RF.sav` | Trained random forest (pickle) |
| `train.csv`, `test.csv` | Loan applications (labelled / unlabelled) |
| `*.gif`, `loan_image.jpg` | Images used by the app |
