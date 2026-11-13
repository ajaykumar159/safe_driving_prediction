import pandas as pd
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def train_model():
    print("Loading data...")
    df = pd.read_csv("data/train.csv")
    
    # Calculate absolute correlation with target to find the best predictors
    print("Selecting top features based on correlation...")
    correlations = df.drop(columns=["id"]).corr()["target"].abs().sort_values(ascending=False)
    
    # Select the top 15 features (index 0 is the target itself, so we take 1 to 16)
    best_features = correlations.index[1:16].tolist()
    print(f"Selected Top 15 features: {best_features}")
    
    # Filter dataset to only use the best features
    X = df[best_features]
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Building pipeline...")
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))
    ])
    
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    preds = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    print(f"Validation ROC-AUC: {auc:.4f}")
    
    # Save both the model pipeline and the list of selected features
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/model_pipeline.pkl")
    joblib.dump(best_features, "models/selected_features.pkl")
    print("Model and selected features saved to models/")

if __name__ == "__main__":
    train_model()