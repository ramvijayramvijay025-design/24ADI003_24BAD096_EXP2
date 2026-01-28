import pandas as pd, numpy as np, zipfile, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
zip_path = r"C:\Users\Ramya.R\Downloads\archive (10).zip"
with zipfile.ZipFile(zip_path, 'r') as z: 
    z.extractall(r"C:\Users\Ramya.R\Downloads\LIC_data")
    csv_path = r"C:\Users\Ramya.R\Downloads\LIC_data\\" + z.namelist()[0]
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
open_col = [c for c in df.columns if 'open' in c][0]
close_col = [c for c in df.columns if 'close' in c][0]
df['price_movement'] = np.where(df[close_col] > df[open_col], 1, 0)
feature_cols = [c for c in df.columns if any(f in c for f in ['open','high','low','volume'])]
X = df[feature_cols].fillna(df[feature_cols].mean())
y = df['price_movement']

X_scaled = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
y_pred = lr.predict(X_test)
y_prob = lr.predict_proba(X_test)[:,1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f'AUC={auc(fpr,tpr):.2f}'); plt.plot([0,1],[0,1],'k--')
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve'); plt.legend(); plt.show()

coef_df = pd.DataFrame({'Feature': feature_cols, 'Coefficient': lr.coef_[0]})
print(coef_df)
sns.barplot(x='Coefficient', y='Feature', data=coef_df); plt.title("Feature Importance"); plt.show()
params = {'C':[0.01,0.1,1,10],'penalty':['l2'],'solver':['liblinear']}
grid = GridSearchCV(LogisticRegression(max_iter=1000), params, cv=5, scoring='accuracy').fit(X_train, y_train)
best_model = grid.best_estimator_
print("Best Params:", grid.best_params_)
print("Tuned Model Accuracy:", accuracy_score(y_test, best_model.predict(X_test)))
