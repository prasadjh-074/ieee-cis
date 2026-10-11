import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

transaction = pd.read_csv('train_transaction.csv')
identity = pd.read_csv('train_identity.csv')
merged = transaction.merge(identity, on='TransactionID', how='left')
merged['has_identity'] = merged['DeviceType'].notna().astype(int)
merged['hour'] = (merged['TransactionDT'] // 3600) % 24
merged['card1_count'] = merged.groupby('card1')['card1'].transform('count')
merged['addr1_count'] = merged.groupby('addr1')['addr1'].transform('count')
merged['device_count'] = merged.groupby('DeviceInfo')['DeviceInfo'].transform('count')

features = ['TransactionAmt', 'hour', 'ProductCD', 'card4', 'card6',
            'has_identity', 'DeviceType', 'C2', 'D7', 'D8', 'V257',
            'card1_count', 'addr1_count', 'device_count']

X = merged[features].copy()
X['V257_missing'] = X['V257'].isnull().astype(int)
X = X.fillna(-999)
X = pd.get_dummies(X, columns=['ProductCD', 'card4', 'card6', 'DeviceType'], drop_first=True)
y = merged['isFraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

train_idx = X_train.index
fraud_train = merged.loc[train_idx][merged.loc[train_idx, 'isFraud'] == 1]
device_fraud_counts = fraud_train['DeviceInfo'].value_counts()
risky_devices = device_fraud_counts[device_fraud_counts >= 20].index
merged['risky_device_flag'] = merged['DeviceInfo'].isin(risky_devices).astype(int)
X_train['risky_device_flag'] = merged.loc[X_train.index, 'risky_device_flag']
X_test['risky_device_flag'] = merged.loc[X_test.index, 'risky_device_flag']

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
model2 = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                            scale_pos_weight=scale_pos_weight, eval_metric='aucpr', random_state=42)
model2.fit(X_train, y_train)

# --- SHAP ---
explainer = shap.TreeExplainer(model2)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150)
print("Saved shap_summary.png")