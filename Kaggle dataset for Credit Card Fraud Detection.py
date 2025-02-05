# Step 1: Install Kaggle CLI
!pip install kaggle

# Step 2: Upload kaggle.json (manually)
from google.colab import files
files.upload()  # Upload kaggle.json when prompted

# Step 3: Move kaggle.json to ~/.kaggle and set permissions
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Step 4: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Step 5: Download Credit Card Fraud Detection Dataset from Kaggle
dataset_name = "mlg-ulb/creditcardfraud"  # Kaggle dataset for Credit Card Fraud Detection
!kaggle datasets download -d {dataset_name} -p /content

# Step 6: Unzip & Store in Google Drive
import shutil
shutil.unpack_archive(f"/content/{dataset_name.split('/')[-1]}.zip", "/content/drive/MyDrive/Kaggle_Datasets/CreditCardFraud")

print("✅ Credit Card Fraud Detection Dataset Downloaded and Stored in Google Drive Successfully!")

path = "/content/drive/MyDrive/Kaggle_Datasets/CreditCardFraud/creditcard.csv"
df = pd.read_csv(path)
df = df.iloc[:9000]
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE, ADASYN
from xgboost import XGBClassifier

X = df.drop(['Class'], axis=1)
y = df['Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

adasyn = ADASYN(random_state=42)
X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)

model = XGBClassifier(scale_pos_weight=100)  
model.fit(X_train_smote, y_train_smote)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

model.fit(X_train_adasyn, y_train_adasyn)
y_pred_adasyn = model.predict(X_test)
print("Accuracy with ADASYN:", accuracy_score(y_test, y_pred_adasyn))
print("Classification Report with ADASYN:\n", classification_report(y_test, y_pred_adasyn))

from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

iso_forest = IsolationForest(contamination=0.01, random_state=42)
y_pred_if = iso_forest.fit_predict(X_scaled)

y_pred_if = [1 if i == -1 else 0 for i in y_pred_if]

print("Isolation Forest Classification Report:")
print(classification_report(y, y_pred_if))

autoencoder = Sequential()
autoencoder.add(Dense(32, activation='relu', input_dim=X_scaled.shape[1], kernel_regularizer=regularizers.l2(0.01)))
autoencoder.add(Dense(16, activation='relu'))
autoencoder.add(Dense(8, activation='relu'))
autoencoder.add(Dense(16, activation='relu'))
autoencoder.add(Dense(32, activation='relu'))
autoencoder.add(Dense(X_scaled.shape[1], activation='sigmoid'))

autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.fit(X_scaled, X_scaled, epochs=5, batch_size=20, validation_split=0.2, verbose=2)

reconstructed = autoencoder.predict(X_scaled)
reconstruction_error = np.mean(np.abs(X_scaled - reconstructed), axis=1)
threshold = np.percentile(reconstruction_error, 95)  
y_pred_ae = [1 if e > threshold else 0 for e in reconstruction_error]

print("Autoencoder Classification Report:")
print(classification_report(y, y_pred_ae))
 
from hyperopt import fmin, tpe, hp, Trials
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def objective(params):
    
    df = pd.read_csv('/content/drive/MyDrive/Kaggle_Datasets/CreditCardFraud/creditcard.csv')
    X = df.drop(['Class'], axis=1)
    y = df['Class']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    
    model = XGBClassifier(
        max_depth=int(params['max_depth']),
        learning_rate=params['learning_rate'],
        n_estimators=int(params['n_estimators']),
        gamma=params['gamma'],
        scale_pos_weight=params['scale_pos_weight'],
        objective='binary:logistic',
        random_state=42
    )
    model.fit(X_train, y_train)
    
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return -accuracy  

space = {
    'max_depth': hp.choice('max_depth', [3, 5, 7, 9]),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.1),
    'n_estimators': hp.choice('n_estimators', [100, 200, 300]),
    'gamma': hp.uniform('gamma', 0, 1),
    'scale_pos_weight': hp.uniform('scale_pos_weight', 1, 10),
}

trials = Trials()

best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=3, trials=trials)

print("Best hyperparameters:", best)

best_model = XGBClassifier(
    max_depth=int(best['max_depth']),
    learning_rate=best['learning_rate'],
    n_estimators=int(best['n_estimators']),
    gamma=best['gamma'],
    scale_pos_weight=best['scale_pos_weight'],
    objective='binary:logistic',
    random_state=42
)

best_model.fit(X_train, y_train)
y_pred_best = best_model.predict(X_test)
print("Best Model Accuracy:", accuracy_score(y_test, y_pred_best))
