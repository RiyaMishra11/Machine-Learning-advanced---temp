import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate(name, model, X, y):
    p=model.predict(X)
    print(f"{name}: MAE={mean_absolute_error(y,p):.3f}, RMSE={np.sqrt(mean_squared_error(y,p)):.3f}, R2={r2_score(y,p):.3f}")

def main():
    X,y=make_regression(n_samples=650,n_features=5,noise=12,random_state=70)
    rng=np.random.default_rng(70)
    ids=rng.choice(len(y),35,replace=False)
    y[ids]+=rng.normal(0,220,len(ids))
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=70)
    base=LinearRegression().fit(Xtr,ytr)
    robust=RANSACRegressor(estimator=LinearRegression(),min_samples=.5,residual_threshold=35,max_trials=200,random_state=70).fit(Xtr,ytr)
    evaluate("Linear Regression",base,Xte,yte)
    evaluate("RANSAC Regression",robust,Xte,yte)
    print("Training inliers:",int(robust.inlier_mask_.sum()),"/",len(ytr))

if __name__ == "__main__":
    main()
