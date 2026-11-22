import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate(name, model, X, y):
    p = model.predict(X)
    print(f"{name}: MAE={mean_absolute_error(y,p):.3f}, RMSE={np.sqrt(mean_squared_error(y,p)):.3f}, R2={r2_score(y,p):.3f}")

def main():
    X,y=make_regression(n_samples=700,n_features=7,noise=18,random_state=69)
    rng=np.random.default_rng(69)
    ids=rng.choice(len(y),25,replace=False)
    y[ids]+=rng.normal(0,180,len(ids))
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=69)
    evaluate("Linear Regression",LinearRegression().fit(Xtr,ytr),Xte,yte)
    evaluate("Huber Regression",HuberRegressor(epsilon=1.35,max_iter=1000).fit(Xtr,ytr),Xte,yte)

if __name__ == "__main__":
    main()
