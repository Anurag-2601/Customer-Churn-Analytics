## Customer Churn Risk & Retention Analytics

### Business Problem

Customer churn directly impacts revenue and growth in subscription-based telecom services.
The objective of this project is to **identify churn drivers, segment customer risk, and support proactive retention strategies** using data analytics.

---

### Dataset Overview

* Telecom customer usage and service interaction data
* Key features include call usage, service calls, plan details, and churn status
* Dataset cleaned and validated before analysis

---

### Data Cleaning & Validation (SQL)

* Removed duplicate records using assumption-based deduplication due to missing customer IDs
* Audited and handled null values across numerical and categorical columns
* Identified outliers in usage metrics and validated abnormal usage scenarios
* Performed logical consistency checks (e.g., international usage without international plans)

---

### Exploratory Data Analysis (SQL)

* Calculated overall churn rate and segment-wise churn metrics
* Analyzed churn trends across customer service calls and international plan usage
* Compared usage behavior between churned and non-churned customers
* Applied window functions and CTEs for trend and cohort-based analysis

---

###  Risk Segmentation Logic

Customers were segmented based on service interaction intensity:

* **High Risk – Immediate Intervention:** ≥4 customer service calls
* **Medium Risk – Monitor:** 2–3 customer service calls
* **Low Risk – Stable:** ≤1 customer service call

This segmentation helps prioritize retention efforts efficiently.

---

### Power BI Dashboard

The interactive dashboard includes:

* Key KPIs: Total Customers, Customers Churned, Churn Rate (%)
* Churn analysis by customer service calls and international plan usage
* Risk-wise churn rate and customer distribution
* Usage behavior comparison between churned and non-churned customers
* Slicers for state, plan type, and churn status

**Dashboard Insight Highlights:**

* Customers with higher service call frequency show significantly higher churn
* International plan users churn ~4× more than non-international users
* Churned customers exhibit higher usage, indicating dissatisfaction rather than low engagement

---

### Business Insights & Recommendations

* Prioritize proactive retention for customers with frequent service calls
* Review pricing and service experience for international plan users
* Monitor medium-risk customers early to prevent escalation into high-risk churn

---

###  Tools Used

* **SQL:** Data cleaning, EDA, window functions
* **Power BI:** DAX measures, KPI design, interactive dashboards

---






