"""
Main pipeline for Customer Churn Analytics
Runs the complete analysis workflow
"""
import os
import sys

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ['data', 'output', 'models', 'notebooks']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def run_pipeline():
    """Run the complete churn analysis pipeline"""
    print("\n" + "="*70)
    print("CUSTOMER CHURN ANALYTICS PIPELINE")
    print("="*70)
    
    # Ensure directories exist
    create_directories()
    
    # Step 1: Generate sample data (if not exists)
    if not os.path.exists('data/customer_churn_data.csv'):
        print("\nStep 1: Generating sample customer data...")
        from generate_data import generate_data
        generate_data()
    else:
        print("\nStep 1: Customer data already exists. Skipping generation.")
    
    # Step 2: Run Exploratory Data Analysis
    print("\nStep 2: Running Exploratory Data Analysis...")
    from exploratory_analysis import ChurnEDA
    eda = ChurnEDA('data/customer_churn_data.csv')
    eda.run_complete_eda()
    
    # Step 3: Preprocess data and train models
    print("\nStep 3: Preprocessing data and training models...")
    from train_models import ChurnModelTrainer
    from preprocess_data import ChurnDataPreprocessor
    
    preprocessor = ChurnDataPreprocessor('data/customer_churn_data.csv')
    preprocessor.handle_missing_values()
    preprocessor.engineer_features()
    preprocessor.encode_categorical_features()
    X_train, X_test, y_train, y_test = preprocessor.prepare_for_modeling()
    
    # Save processed data
    processed_df = preprocessor.get_processed_data()
    processed_df.to_csv('data/processed_customer_churn_data.csv', index=False)
    print("Processed data saved to data/processed_customer_churn_data.csv")
    
    # Step 4: Train and evaluate models
    print("\nStep 4: Training and evaluating models...")
    trainer = ChurnModelTrainer(X_train, X_test, y_train, y_test)
    
    trainer.train_logistic_regression()
    trainer.train_random_forest()
    trainer.train_gradient_boosting()
    trainer.train_xgboost()
    
    # Step 5: Compare models and generate visualizations
    print("\nStep 5: Comparing models and generating visualizations...")
    comparison = trainer.compare_models()
    trainer.plot_roc_curves()
    trainer.plot_confusion_matrices()
    trainer.plot_feature_importance()
    best_model = trainer.save_best_model()
    
    # Final summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nResults Summary:")
    print(f"- Best Model: {best_model}")
    print(f"- All visualizations saved to: output/")
    print(f"- Model saved to: models/best_model.pkl")
    print(f"- Processed data saved to: data/processed_customer_churn_data.csv")
    print("\nNext Steps:")
    print("1. Review the EDA visualizations in output/")
    print("2. Check model comparison results in output/model_comparison.csv")
    print("3. Use the best model for predictions on new data")
    print("="*70)

if __name__ == "__main__":
    run_pipeline()
