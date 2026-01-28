import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(r"C:\Users\Ramya.R\Downloads\archive (9)\bottle.csv", low_memory=False)
features = ['Depthm', 'Salnty', 'O2ml_L']
target = 'T_degC'
data = df[features + [target]].fillna(df[features + [target]].mean())
X, y = data[features], data[target]
X_scaled = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
lr = LinearRegression().fit(X_train, y_train)
y_pred = lr.predict(X_test)

print("Linear Regression Results")
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))
plt.scatter(y_test, y_pred); plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title("Actual vs Predicted"); plt.show()
plt.scatter(y_pred, y_test - y_pred); plt.axhline(0); plt.xlabel("Predicted"); plt.ylabel("Residuals"); plt.title("Residual Errors"); plt.show()
for model, alpha in [(Ridge(alpha=1.0), "Ridge"), (Lasso(alpha=0.01), "Lasso")]:
    model.fit(X_train, y_train)
    print(f"{alpha} R2 Score:", r2_score(y_test, model.predict(X_test)))
