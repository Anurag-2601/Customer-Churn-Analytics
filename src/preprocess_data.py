"""
Data preprocessing and feature engineering for customer churn analysis
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class ChurnDataPreprocessor:
    def __init__(self, filepath):
        """Initialize preprocessor with data file path"""
        self.df = pd.read_csv(filepath)
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def explore_data(self):
        """Display basic data exploration"""
        print("=" * 50)
        print("DATA EXPLORATION")
        print("=" * 50)
        print(f"\nDataset shape: {self.df.shape}")
        print(f"\nFirst few rows:")
        print(self.df.head())
        print(f"\nData types:")
        print(self.df.dtypes)
        print(f"\nMissing values:")
        print(self.df.isnull().sum())
        print(f"\nBasic statistics:")
        print(self.df.describe())
        print(f"\nChurn distribution:")
        print(self.df['Churn'].value_counts())
        print(f"\nChurn rate: {(self.df['Churn'] == 'Yes').sum() / len(self.df) * 100:.2f}%")
        
    def handle_missing_values(self):
        """Handle missing values in the dataset"""
        # Check for missing values
        missing_values = self.df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"\nHandling missing values...")
            # For numerical columns, fill with median
            numerical_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if self.df[col].isnull().sum() > 0:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
            
            # For categorical columns, fill with mode
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if self.df[col].isnull().sum() > 0:
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        else:
            print("\nNo missing values found.")
    
    def engineer_features(self):
        """Create new features from existing ones"""
        print("\nEngineering features...")
        
        # Create tenure groups
        self.df['TenureGroup'] = pd.cut(self.df['Tenure'], 
                                         bins=[0, 12, 24, 48, 72],
                                         labels=['0-12', '12-24', '24-48', '48-72'])
        
        # Create charge ratio (monthly vs average)
        self.df['ChargeRatio'] = self.df['MonthlyCharges'] / self.df['MonthlyCharges'].mean()
        
        # Create total services count
        service_cols = ['OnlineSecurity', 'TechSupport', 'StreamingTV']
        self.df['TotalServices'] = 0
        for col in service_cols:
            self.df['TotalServices'] += (self.df[col] == 'Yes').astype(int)
        
        # Create contract value (longer = higher value)
        contract_value = {'Month-to-month': 1, 'One year': 2, 'Two year': 3}
        self.df['ContractValue'] = self.df['Contract'].map(contract_value)
        
        print("Features engineered successfully!")
        
    def encode_categorical_features(self):
        """Encode categorical variables"""
        print("\nEncoding categorical features...")
        
        # Identify categorical columns (excluding target)
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        if 'Churn' in categorical_cols:
            categorical_cols.remove('Churn')
        
        # Remove CustomerID if present
        if 'CustomerID' in categorical_cols:
            categorical_cols.remove('CustomerID')
        
        # Encode categorical features
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[col + '_Encoded'] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le
        
        # Encode target variable
        self.df['Churn_Encoded'] = (self.df['Churn'] == 'Yes').astype(int)
        
        print(f"Encoded {len(categorical_cols)} categorical features.")
        
    def prepare_for_modeling(self):
        """Prepare final dataset for modeling"""
        print("\nPreparing data for modeling...")
        
        # Select features for modeling
        feature_cols = [col for col in self.df.columns if col.endswith('_Encoded') or 
                       col in ['Age', 'Tenure', 'MonthlyCharges', 'TotalCharges', 
                              'TotalServices', 'ContractValue', 'ChargeRatio']]
        
        # Remove target from features
        feature_cols = [col for col in feature_cols if col != 'Churn_Encoded']
        
        X = self.df[feature_cols]
        y = self.df['Churn_Encoded']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame for better readability
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        print(f"Training set size: {X_train_scaled.shape}")
        print(f"Test set size: {X_test_scaled.shape}")
        print(f"Number of features: {X_train_scaled.shape[1]}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def get_processed_data(self):
        """Get the processed dataframe"""
        return self.df


if __name__ == "__main__":
    # Load and preprocess data
    preprocessor = ChurnDataPreprocessor('../data/customer_churn_data.csv')
    
    # Explore data
    preprocessor.explore_data()
    
    # Handle missing values
    preprocessor.handle_missing_values()
    
    # Engineer features
    preprocessor.engineer_features()
    
    # Encode categorical features
    preprocessor.encode_categorical_features()
    
    # Prepare for modeling
    X_train, X_test, y_train, y_test = preprocessor.prepare_for_modeling()
    
    # Save processed data
    processed_df = preprocessor.get_processed_data()
    processed_df.to_csv('../data/processed_customer_churn_data.csv', index=False)
    
    print("\n" + "=" * 50)
    print("Data preprocessing completed successfully!")
    print("=" * 50)
