"""Machine Learning 5: Data preprocessing and feature scaling."""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Do not fit on test data.
    print("MACHINE LEARNING 5 - DATA PREPROCESSING")
    print("Original data shape:", data.data.shape)
    print("Train/test shapes:", X_train.shape, X_test.shape)
    print("Scaled train shape:", X_train_scaled.shape)
    print("Scaled test shape:", X_test_scaled.shape)
    print("First five scaled feature means:", X_train_scaled.mean(axis=0)[:5].round(3))

if __name__ == "__main__":
    main()
