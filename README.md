# 🔍 Procurement Fraud Detection Dashboard

A comprehensive fraud detection system for public procurement contracts using Machine Learning and Graph Analytics.

## 🚀 Quick Start

### Option 1: Use Pre-Generated Data (Recommended for Deployment)

```bash
cd deployment
pip install -r requirements.txt
streamlit run app.py
```

### Option 2: Regenerate Data from Preprocessing Outputs (Local Development)

```bash
cd deployment
pip install -r requirements.txt
python setup_dashboard.py  # Requires Preprocessing_Output_Folder
streamlit run app.py
```

## ☁️ Streamlit Cloud Deployment

This dashboard is designed to work with pre-generated data files. 
No preprocessing folder is needed for cloud deployment.

### Required Files for Deployment

```
deployment/
├── app.py
├── config.py
├── requirements.txt
├── data/
│   ├── contract_risk_predictions.csv  ✓ Required
│   ├── agent_risk_scores.csv          ✓ Required
│   ├── agent_embeddings.csv           ✓ Required
│   ├── contract_risk_patterns.csv     ✓ Required
│   ├── agent_communities.csv          Optional
│   └── community_risk_summary.csv     Optional
├── models/
│   ├── lgbm_best_model.joblib         ✓ Required
│   ├── lgbm_model_metadata.pkl        ✓ Required
│   ├── feature_selector.pkl           ✓ Required
│   └── label_encoders.pkl             Optional
├── pages/
│   └── *.py
├── services/
│   └── *.py
└── utils/
    └── *.py
```

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| F1 Score | 0.8523 |
| ROC-AUC | 0.9234 |

## 📝 Author

**AUSTIN BAY QI HERN**  
TP068004  
APD3F2505CS(DA)
