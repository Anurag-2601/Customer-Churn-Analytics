"""
Exploratory Data Analysis (EDA) for Customer Churn
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class ChurnEDA:
    def __init__(self, filepath):
        """Initialize EDA with data file path"""
        self.df = pd.read_csv(filepath)
        
    def plot_churn_distribution(self):
        """Plot churn distribution"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Count plot
        churn_counts = self.df['Churn'].value_counts()
        axes[0].bar(churn_counts.index, churn_counts.values, color=['#2ecc71', '#e74c3c'])
        axes[0].set_xlabel('Churn Status', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title('Churn Distribution (Count)', fontweight='bold', fontsize=14)
        axes[0].grid(alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(churn_counts.values):
            axes[0].text(i, v + 10, str(v), ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        colors = ['#2ecc71', '#e74c3c']
        axes[1].pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%',
                   colors=colors, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
        axes[1].set_title('Churn Distribution (Percentage)', fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        plt.savefig('../output/churn_distribution.png', dpi=300, bbox_inches='tight')
        print("Churn distribution plot saved!")
        plt.close()
        
    def plot_numerical_features(self):
        """Plot distribution of numerical features"""
        numerical_cols = ['Age', 'Tenure', 'MonthlyCharges', 'TotalCharges']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        
        for idx, col in enumerate(numerical_cols):
            # Distribution by churn status
            for churn_status in ['No', 'Yes']:
                data = self.df[self.df['Churn'] == churn_status][col]
                axes[idx].hist(data, alpha=0.6, label=f'Churn: {churn_status}', bins=30)
            
            axes[idx].set_xlabel(col, fontsize=12)
            axes[idx].set_ylabel('Frequency', fontsize=12)
            axes[idx].set_title(f'{col} Distribution by Churn', fontweight='bold', fontsize=12)
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../output/numerical_features_distribution.png', dpi=300, bbox_inches='tight')
        print("Numerical features distribution plot saved!")
        plt.close()
        
    def plot_categorical_features(self):
        """Plot categorical features against churn"""
        categorical_cols = ['Gender', 'Contract', 'InternetService', 'PaymentMethod']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        for idx, col in enumerate(categorical_cols):
            # Create crosstab
            ct = pd.crosstab(self.df[col], self.df['Churn'], normalize='index') * 100
            
            ct.plot(kind='bar', ax=axes[idx], color=['#2ecc71', '#e74c3c'], width=0.7)
            axes[idx].set_xlabel(col, fontsize=12)
            axes[idx].set_ylabel('Percentage (%)', fontsize=12)
            axes[idx].set_title(f'Churn Rate by {col}', fontweight='bold', fontsize=12)
            axes[idx].legend(['No Churn', 'Churn'])
            axes[idx].grid(alpha=0.3, axis='y')
            axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('../output/categorical_features_churn.png', dpi=300, bbox_inches='tight')
        print("Categorical features plot saved!")
        plt.close()
        
    def plot_correlation_matrix(self):
        """Plot correlation matrix for numerical features"""
        # Select numerical columns and encode target
        numerical_cols = ['Age', 'Tenure', 'MonthlyCharges', 'TotalCharges']
        df_corr = self.df[numerical_cols].copy()
        df_corr['Churn'] = (self.df['Churn'] == 'Yes').astype(int)
        
        # Calculate correlation
        corr_matrix = df_corr.corr()
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix - Numerical Features', fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig('../output/correlation_matrix.png', dpi=300, bbox_inches='tight')
        print("Correlation matrix saved!")
        plt.close()
        
    def plot_tenure_vs_charges(self):
        """Plot relationship between tenure and charges"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Scatter plot: Tenure vs Monthly Charges
        for churn_status in ['No', 'Yes']:
            data = self.df[self.df['Churn'] == churn_status]
            axes[0].scatter(data['Tenure'], data['MonthlyCharges'], 
                          alpha=0.5, label=f'Churn: {churn_status}', s=30)
        
        axes[0].set_xlabel('Tenure (months)', fontsize=12)
        axes[0].set_ylabel('Monthly Charges ($)', fontsize=12)
        axes[0].set_title('Tenure vs Monthly Charges', fontweight='bold', fontsize=14)
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Box plot: Tenure by Churn
        self.df.boxplot(column='Tenure', by='Churn', ax=axes[1])
        axes[1].set_xlabel('Churn Status', fontsize=12)
        axes[1].set_ylabel('Tenure (months)', fontsize=12)
        axes[1].set_title('Tenure Distribution by Churn', fontweight='bold', fontsize=14)
        axes[1].get_figure().suptitle('')  # Remove default title
        
        plt.tight_layout()
        plt.savefig('../output/tenure_vs_charges.png', dpi=300, bbox_inches='tight')
        print("Tenure vs charges plot saved!")
        plt.close()
        
    def plot_contract_analysis(self):
        """Detailed analysis of contract types"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Churn rate by contract type
        contract_churn = self.df.groupby('Contract')['Churn'].apply(
            lambda x: (x == 'Yes').sum() / len(x) * 100
        ).sort_values(ascending=False)
        
        axes[0].bar(range(len(contract_churn)), contract_churn.values, 
                   color=['#e74c3c', '#f39c12', '#2ecc71'])
        axes[0].set_xticks(range(len(contract_churn)))
        axes[0].set_xticklabels(contract_churn.index, rotation=45, ha='right')
        axes[0].set_ylabel('Churn Rate (%)', fontsize=12)
        axes[0].set_title('Churn Rate by Contract Type', fontweight='bold', fontsize=14)
        axes[0].grid(alpha=0.3, axis='y')
        
        # Add value labels
        for i, v in enumerate(contract_churn.values):
            axes[0].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
        
        # Contract distribution
        contract_dist = self.df['Contract'].value_counts()
        axes[1].pie(contract_dist.values, labels=contract_dist.index, autopct='%1.1f%%',
                   colors=['#3498db', '#9b59b6', '#1abc9c'], startangle=90,
                   textprops={'fontsize': 11, 'fontweight': 'bold'})
        axes[1].set_title('Contract Type Distribution', fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        plt.savefig('../output/contract_analysis.png', dpi=300, bbox_inches='tight')
        print("Contract analysis plot saved!")
        plt.close()
        
    def generate_summary_statistics(self):
        """Generate and save summary statistics"""
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        # Overall statistics
        total_customers = len(self.df)
        churned_customers = (self.df['Churn'] == 'Yes').sum()
        churn_rate = churned_customers / total_customers * 100
        
        summary = {
            'Total Customers': [total_customers],
            'Churned Customers': [churned_customers],
            'Retained Customers': [total_customers - churned_customers],
            'Churn Rate (%)': [f'{churn_rate:.2f}'],
            'Average Age': [f'{self.df["Age"].mean():.1f}'],
            'Average Tenure (months)': [f'{self.df["Tenure"].mean():.1f}'],
            'Average Monthly Charges ($)': [f'{self.df["MonthlyCharges"].mean():.2f}'],
        }
        
        summary_df = pd.DataFrame(summary).T
        summary_df.columns = ['Value']
        
        print(summary_df)
        summary_df.to_csv('../output/summary_statistics.csv')
        print("\nSummary statistics saved to output/summary_statistics.csv")
        
    def run_complete_eda(self):
        """Run complete exploratory data analysis"""
        print("\n" + "="*60)
        print("STARTING EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        self.plot_churn_distribution()
        self.plot_numerical_features()
        self.plot_categorical_features()
        self.plot_correlation_matrix()
        self.plot_tenure_vs_charges()
        self.plot_contract_analysis()
        self.generate_summary_statistics()
        
        print("\n" + "="*60)
        print("EDA COMPLETED SUCCESSFULLY!")
        print("All visualizations saved to output/ directory")
        print("="*60)


if __name__ == "__main__":
    # Run EDA
    eda = ChurnEDA('../data/customer_churn_data.csv')
    eda.run_complete_eda()
