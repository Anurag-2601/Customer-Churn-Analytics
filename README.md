# Customer Churn Analytics

A comprehensive data analytics project for predicting and analyzing customer churn using machine learning techniques.

## 📋 Project Overview

Customer churn (customer attrition) is when customers stop doing business with a company. This project provides a complete analytics solution to:
- Analyze customer behavior patterns
- Identify key factors that influence churn
- Build predictive models to forecast customer churn
- Provide actionable insights for customer retention strategies

## 🎯 Features

- **Data Generation**: Creates realistic customer churn datasets for analysis
- **Exploratory Data Analysis**: Comprehensive visualization of customer patterns and churn factors
- **Feature Engineering**: Advanced feature creation for improved model performance
- **Multiple ML Models**: Comparison of various algorithms including:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - XGBoost
- **Model Evaluation**: Detailed performance metrics and visualizations
- **Automated Pipeline**: End-to-end workflow from data to insights

## 📁 Project Structure

```
Customer-Churn-Analytics/
├── data/                           # Data directory
│   ├── customer_churn_data.csv    # Raw customer data
│   └── processed_customer_churn_data.csv  # Processed data
├── src/                            # Source code
│   ├── generate_data.py           # Data generation script
│   ├── preprocess_data.py         # Data preprocessing and feature engineering
│   ├── exploratory_analysis.py    # EDA and visualizations
│   ├── train_models.py            # Model training and evaluation
│   └── main_pipeline.py           # Main pipeline orchestration
├── output/                         # Generated visualizations and results
├── models/                         # Saved models
├── notebooks/                      # Jupyter notebooks (optional)
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Anurag-2601/Customer-Churn-Analytics.git
cd Customer-Churn-Analytics
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### Option 1: Run Complete Pipeline

Execute the entire analysis workflow with a single command:

```bash
cd src
python main_pipeline.py
```

This will:
1. Generate sample customer data
2. Perform exploratory data analysis
3. Preprocess and engineer features
4. Train multiple ML models
5. Generate comparison visualizations
6. Save the best model

#### Option 2: Run Individual Components

**Generate Sample Data:**
```bash
cd src
python generate_data.py
```

**Run Exploratory Data Analysis:**
```bash
python exploratory_analysis.py
```

**Preprocess Data:**
```bash
python preprocess_data.py
```

**Train and Evaluate Models:**
```bash
python train_models.py
```

## 📊 Key Visualizations

The project generates various visualizations including:
- Churn distribution and rates
- Customer demographic analysis
- Feature correlation heatmaps
- ROC curves for model comparison
- Confusion matrices
- Feature importance plots
- Contract and service analysis

All visualizations are saved in the `output/` directory.

## 🤖 Models and Performance

The project trains and compares multiple machine learning models:

| Model | Key Characteristics |
|-------|---------------------|
| Logistic Regression | Fast, interpretable baseline model |
| Random Forest | Ensemble method, handles non-linear relationships |
| Gradient Boosting | Sequential ensemble, high accuracy |
| XGBoost | Advanced boosting, excellent performance |

Performance metrics include:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Cross-validation scores

## 📈 Key Insights

The analysis typically reveals:
- Contract type significantly impacts churn rates
- Tenure is inversely related to churn probability
- Monthly charges affect customer retention
- Support services reduce churn likelihood
- Payment method correlates with customer loyalty

## 🛠️ Technologies Used

- **Python 3.8+**: Core programming language
- **pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **scikit-learn**: Machine learning models and evaluation
- **XGBoost**: Advanced gradient boosting
- **matplotlib & seaborn**: Data visualization
- **Jupyter**: Interactive analysis (optional)

## 📝 Dataset Features

The customer dataset includes:
- **Demographics**: Age, Gender
- **Account Information**: Tenure, Contract Type
- **Services**: Internet Service, Tech Support, Streaming, etc.
- **Billing**: Monthly Charges, Total Charges, Payment Method
- **Target**: Churn (Yes/No)

## 🔍 Future Enhancements

- [ ] Deep learning models (Neural Networks)
- [ ] Real-time prediction API
- [ ] Interactive dashboard with Plotly/Dash
- [ ] A/B testing simulation
- [ ] Customer segmentation clustering
- [ ] Time series analysis for churn trends

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Anurag**
- GitHub: [@Anurag-2601](https://github.com/Anurag-2601)

## 📧 Contact

For questions or feedback, please open an issue in the GitHub repository.

---

**Note**: This project uses synthetic data for demonstration purposes. For production use, replace with actual customer data while ensuring compliance with data privacy regulations.