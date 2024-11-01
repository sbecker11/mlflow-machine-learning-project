# model.py
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def main():
    # Log the MLflow run
    with mlflow.start_run():
        # Load data
        data = pd.read_csv("data/raw/Stocks.csv")
        X, y = data.drop("target", axis=1), data["target"]
        # Preprocess data (example: fill missing values)
        data.fillna(data.mean(), inplace=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Log metrics
        mse = mean_squared_error(y_test, y_pred)
        mlflow.log_metric("mse", mse)
        
        # Log model
        mlflow.sklearn.log_model(model, "random-forest-model")

if __name__ == "__main__":
    main()
