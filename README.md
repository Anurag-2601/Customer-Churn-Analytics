#  Customer Churn Risk & Retention Analytics

##  Overview
Customer churn is a critical business problem for subscription-based and product-led companies.  
This project focuses on **understanding why customers churn**, identifying **early risk signals**, and helping teams **prioritize retention actions and stakeholder decision-making** using business-driven data analytics.

Instead of treating churn purely as a prediction problem, this project emphasizes **behavioral analysis, root cause identification, and actionable insights** that support **product, operations, and customer success teams**.

---

##  Problem Statement
Telecom customers interact with services through usage behavior, subscription plans, and customer support.  
Repeated service issues or pricing dissatisfaction often lead to churn, but these signals are not always visible without structured, end-to-end analysis.

The objectives of this project are to:
- Identify key churn drivers using business metrics  
- Segment customers based on churn risk  
- Support proactive **retention strategy and customer lifecycle management**  
- Enable **data-driven decision-making** for stakeholders  

---

##  Repository Structure

<pre>
Customer-Churn-Risk-Retention-Analytics/
├── dashboard/
│   └── Customer Churn Risk & Retention.pbix
├── data/
│   └── churn dataset.csv
├── notebooks/
│   ├── churn_analysis.ipynb
│   └── Churn_ml_model.ipynb
├── report/
│   └── Customer Churn Risk Report.pdf
├── sql/
│   ├── 1.Data Cleaning.sql
│   └── 2.EDA.sql
└── README.md
</pre>


---


##  Data Understanding
The dataset contains customer-level data covering:
- Usage patterns (day, evening, night, international minutes)
- Service interaction history (customer service calls)
- Plan information (international plan, voice mail plan)
- Churn status (`True / False`)

Each row represents an individual customer within the customer lifecycle.

---

##  Data Cleaning & Validation
Data preparation focused on **data quality and reliability** before analysis:
- Removed duplicate records using assumption-based deduplication due to missing customer IDs  
- Performed data quality checks and validation across all key columns  
- Identified logical inconsistencies (e.g., international usage without an international plan)  
- Reviewed extreme usage values to flag abnormal behavior  

This ensured the analytics pipeline was built on **clean, trustworthy data**.

---

##  Exploratory Analysis & Key Findings
The exploratory analysis focused on **root cause analysis** rather than surface-level reporting.

### Key Drivers Identified:
- **Customer Service Calls:** Churn rate increases sharply with frequent service interactions, indicating service friction  
- **International Plan Usage:** International plan users churn at a significantly higher rate, suggesting pricing or experience gaps  
- **Usage Behavior:** Churned customers show equal or higher usage, indicating dissatisfaction rather than disengagement  

These findings highlight actionable operational insights.

---

##  Risk Segmentation Framework
Customers were segmented based on service interaction intensity to support prioritization:

- **High Risk – Immediate Intervention:** ≥ 4 customer service calls  
- **Medium Risk – Monitor:** 2–3 customer service calls  
- **Low Risk – Stable:** ≤ 1 customer service call  

This risk framework aligns with **real-world retention strategy planning** and enables teams to focus on **who to act on first**.

---

##  Power BI Dashboard
The analytical insights were translated into an **interactive Power BI dashboard** designed for stakeholder consumption.

The dashboard supports:
- KPI monitoring (Total Customers, Customers Churned, Churn Rate %)  
- Churn analysis by service calls and plan type  
- Risk-wise churn rate and customer distribution  
- Usage comparison between churned and non-churned customers  
- Filters for state, plan type, and churn status  

The dashboard is designed to support **data-driven decision-making and operational reviews**, not just static reporting.

---

##  Optional Machine Learning Extension
A supplementary machine learning notebook explores churn prediction as part of the **end-to-end analytics pipeline**.

- Implemented using Logistic Regression for interpretability  
- Used to validate analytical insights rather than replace them  
- Predictive signals aligned with key churn drivers:
  - Customer service calls
  - International plan usage
  - Usage intensity  

Machine learning is included as a **supporting analytical layer**, not a standalone solution.

---

##  Business Insights & Recommendations
- Prioritize proactive retention for customers with frequent service calls  
- Review pricing and service experience for international plan users  
- Monitor medium-risk customers early to prevent escalation into high-risk churn  
- Use risk segmentation to allocate retention efforts efficiently  

---

##  Tools & Technologies
- **SQL:** Data cleaning, validation, EDA, business metrics  
- **Power BI:** KPI design, interactive dashboards, stakeholder reporting  
- **Python:** Exploratory analysis and supplementary churn modeling  

---

##  Conclusion
This project demonstrates an **analytics-first, business-focused approach** to customer churn.  
By combining clean data, structured analysis, and stakeholder-ready visualization, it supports **operational decision-making and retention strategy execution**.

The project reflects how analytics is applied in **real product environments**, not just academic settings.
