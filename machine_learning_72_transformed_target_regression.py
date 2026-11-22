import numpy as np
from sklearn.datasets import make_regression
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    X,signal=make_regression(n_samples=900,n_features=8,noise=25,random_state=72)
    y=np.exp((signal-signal.min())/(signal.max()-signal.min())*3.0)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=72)
    model=TransformedTargetRegressor(
        regressor=RandomForestRegressor(n_estimators=140,min_samples_leaf=2,random_state=72,n_jobs=-1),
        transformer=QuantileTransformer(n_quantiles=100,output_distribution="normal",random_state=72))
    model.fit(Xtr,ytr)
    p=model.predict(Xte)
    print("Transformed Target Regression")
    print(f"MAE: {mean_absolute_error(yte,p):.3f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(yte,p)):.3f}")
    print(f"R2: {r2_score(yte,p):.3f}")
    print("Actual:",np.round(yte[:5],3))
    print("Predicted:",np.round(p[:5],3))

if __name__ == "__main__":
    main()
