#  Customer Churn Risk & Retention Analytics

##  Live Demo
**Streamlit App:** https://customer-churn-analytics-deployement.streamlit.app/

---

##  Project Overview

Customer churn is a major revenue risk for subscription-based businesses. This project analyzes customer behavior to identify churn drivers, segment high-risk customers, and provide actionable retention insights for business stakeholders.

Instead of focusing only on prediction, this project emphasizes:

- Root cause analysis of churn behavior
- Business-driven risk segmentation
- Customer lifecycle insights
- Stakeholder decision support dashboards

The project simulates how data analytics is used in real product organizations for operational decision-making.

---

##  Business Problem

Telecom companies experience customer churn due to:

- Service dissatisfaction
- Pricing concerns
- Poor customer experience
- Repeated service issues

However, churn signals are hidden within usage behavior and service interactions.

### This project solves:

- Identifying key churn drivers
- Detecting early churn risk signals
- Prioritizing customer retention actions
- Supporting data-driven business decisions

---

##  Key Business Impact

- Identified major churn drivers using customer behavior data  
- Built risk segmentation framework for proactive retention  
- Developed stakeholder-ready KPI dashboard  
- Provided operational recommendations for customer success teams  
- Implemented interpretable ML model to validate business insights  

---

##  Repository Structure

<pre>  
Customer-Churn-Risk-Retention-Analytics/
├── dashboard/
│ └── Customer Churn Risk & Retention.pbix
├── data/
│ └── churn dataset.csv
├── notebooks/
│ ├── churn_analysis.ipynb
│ └── Churn_ml_model.ipynb
├── pickle files/
│ └── churn_pipeline.pkl
├── report/
│ └── Customer Churn Risk Report.pdf
├── sql/
│ ├── 1.Data Cleaning.sql
│ └── 2.EDA.sql
├── app.py
├── requirements.txt
└── README.md
<pre>


---

##  Dataset Description

The dataset contains customer-level telecom data including:

- Usage behavior (day, evening, night, international minutes)
- Service interaction history (customer service calls)
- Subscription plans (international plan, voicemail plan)
- Customer churn status (`True / False`)

Each record represents one customer lifecycle.

---

##  Data Cleaning & Validation

To ensure reliable analysis:

- Removed duplicate records using assumption-based deduplication
- Performed data quality validation across key features
- Identified logical inconsistencies (usage without subscription plan)
- Detected extreme values and abnormal usage patterns

This ensured trustworthy analytics results.

---

##  Exploratory Data Analysis (EDA)

The analysis focused on identifying root causes of churn.

###  Key Findings

### 1️ Customer Service Calls → Strongest Churn Driver
- Churn rate increases significantly with frequent support calls
- Indicates customer dissatisfaction and service friction

### 2️ International Plan Users → Higher Churn
- Suggests pricing or experience issues
- Requires business review of plan value

### 3️ High Usage Customers Still Churn
- Churn driven by dissatisfaction, not inactivity

---

##  Customer Risk Segmentation Framework

Customers were segmented based on service interaction frequency:

|    Risk Level   |      Criteria      |    Business Action     |
|-----------------|--------------------|------------------------|
| 🔴 High Risk   | ≥ 4 service calls  | Immediate intervention |
| 🟡 Medium Risk |     2–3 calls      |    Monitor closely     |
| 🟢 Low Risk    |     ≤ 1 call       |    Stable customers    |

This framework helps organizations prioritize retention efforts efficiently.

---

##  Power BI Dashboard

An interactive dashboard was developed for stakeholder reporting.

### Dashboard Features:

- KPI monitoring (Total Customers, Churn Rate %)
- Churn analysis by service calls
- Plan-wise churn distribution
- Customer risk segmentation view
- Usage comparison (churn vs non-churn)
- Interactive filters for business analysis

The dashboard supports operational decision-making rather than static reporting.

---

##  Machine Learning (Supporting Layer)

A Logistic Regression model was implemented to validate analytical insights.

### Model Purpose:
- Interpret churn drivers
- Support business analysis
- Provide predictive risk signals

### Key Predictors:
- Customer service calls
- International plan usage
- Usage intensity

Machine learning complements business analytics rather than replacing it.

---

##  Business Recommendations

- Proactively retain customers with frequent service interactions
- Improve pricing strategy for international plan users
- Monitor medium-risk customers early
- Allocate retention resources based on risk segmentation

---

##  Tools & Technologies

- **SQL** → Data cleaning, validation, EDA
- **Python** → Data analysis, ML modeling
- **Power BI** → KPI dashboard and reporting
- **Streamlit** → Model deployment
- **Pandas, Scikit-learn** → Data processing & modeling

---

##  Deployment

The project is deployed using Streamlit for interactive prediction and analysis.

 **Try the live application:**  
https://customer-churn-analytics-deployement.streamlit.app/

---

##  What This Project Demonstrates

- End-to-end data analytics pipeline
- Business problem solving using data
- SQL + Python + BI integration
- Stakeholder-focused insights
- Production-style deployment
- Real-world analytics workflow

---

##  Author

**Pallela Anurag**  
Aspiring Data Analyst | SQL • Python • Power BI • Machine Learning
