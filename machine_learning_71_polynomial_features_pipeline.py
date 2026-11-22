import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    X,y=make_friedman1(n_samples=800,n_features=5,noise=2.5,random_state=71)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=71)
    model=make_pipeline(PolynomialFeatures(degree=2,include_bias=False),StandardScaler(),Ridge(alpha=2))
    model.fit(Xtr,ytr)
    p=model.predict(Xte)
    print("Polynomial Features + Ridge")
    print(f"MAE: {mean_absolute_error(yte,p):.3f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(yte,p)):.3f}")
    print(f"R2: {r2_score(yte,p):.3f}")
    print("Original features:",X.shape[1])
    print("Expanded features:",model.named_steps["polynomialfeatures"].n_output_features_)

if __name__ == "__main__":
    main()
