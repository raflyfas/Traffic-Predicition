# Generated from: zurich_DSA (4) (1).ipynb
# Converted at: 2026-01-01T20:01:24.012Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# import os
# from google.colab import drive

# os.makedirs('/content', exist_ok=True)
# drive.mount('/content/drive')

# # Data Preparation


df = pd.read_csv("C:/Users/PC/Downloads/zurich.csv")
print(f"Initial data shape: {df.shape}")

df.isnull().sum()

df.head(20)

df['day'] = pd.to_datetime(df['day'], format="%Y-%m-%d")

# ## Analisis Nilai Hilang


missing_values = df.isnull().sum()
missing_percentage = 100 * df.isnull().sum() / len(df)

missing_data = pd.DataFrame({
    'Total Missing Values': missing_values,
    'Percentage': missing_percentage
})

missing_data = missing_data[missing_data['Total Missing Values'] > 0].sort_values(by='Total Missing Values', ascending=False)

print("Analisis Nilai Hilang:")
print(missing_data)

df = df.drop(columns=['speed'], errors='ignore')

df = df.drop_duplicates()

df = df[(df['flow'] >= 0) & (df['occ'] >= 0)]

# ## Statistik Deskriptif


print("Statistik Deskriptif untuk Kolom Numerik Terpilih:")
print(df[['flow', 'occ', 'interval', 'error']].describe())

# ## Outlier clipping (Extreme)


q1 = df['flow'].quantile(0.01)
q99 = df['flow'].quantile(0.99)
df['flow_clip'] = df['flow'].clip(q1, q99)

import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))
plt.hist(df['flow'], bins=100)
plt.title("Distribusi Flow (Raw)")
plt.show()

# Log Transform (Buat dist heavy right tail) karena data sangat skewed ke kanan


df['flow_log'] = np.log1p(df['flow'])

plt.figure(figsize=(12,5))
plt.hist(df['flow_log'], bins=100)
plt.title("Distribusi Flow (Log Transform)")
plt.show()


plt.figure(figsize=(12,5))
plt.hist(df['occ'], bins=100)
plt.title("Distribusi Occ")
plt.show()

df['occ_log'] = np.log1p(df['occ'])

plt.figure(figsize=(12,5))
plt.hist(df['occ_log'], bins=100)
plt.title("Distribusi Occ after log")
plt.show()

# ## Deteksi Outlier pada 'flow' menggunakan IQR


Q1 = df['flow'].quantile(0.25)
Q3 = df['flow'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_flow = df[(df['flow'] < lower_bound) | (df['flow'] > upper_bound)]

print(f"Q1 (flow): {Q1}")
print(f"Q3 (flow): {Q3}")
print(f"IQR (flow): {IQR}")
print(f"Lower Bound (flow): {lower_bound}")
print(f"Upper Bound (flow): {upper_bound}")
print(f"Number of outliers in 'flow': {len(outliers_flow)}")

Q1_occ = df['occ'].quantile(0.25)
Q3_occ = df['occ'].quantile(0.75)
IQR_occ = Q3_occ - Q1_occ

lower_bound_occ = Q1_occ - 1.5 * IQR_occ
upper_bound_occ = Q3_occ + 1.5 * IQR_occ

outliers_occ = df[(df['occ'] < lower_bound_occ) | (df['occ'] > upper_bound_occ)]

print(f"Q1 (occ): {Q1_occ}")
print(f"Q3 (occ): {Q3_occ}")
print(f"IQR (occ): {IQR_occ}")
print(f"Lower Bound (occ): {lower_bound_occ}")
print(f"Upper Bound (occ): {upper_bound_occ}")
print(f"Number of outliers in 'occ': {len(outliers_occ)}")

plt.figure(figsize=(10, 6))
plt.boxplot(df['flow'])
plt.title('Box Plot of Flow with Outliers')
plt.ylabel('Flow')
plt.show()

# **Reasoning**:
# To visualize the identified outliers and the distribution of the 'occ' column, I will create a box plot. This will graphically represent the quartiles, IQR, and the data points that extend beyond the whiskers, which are considered outliers.
# 
# 


plt.figure(figsize=(10, 6))
plt.boxplot(df['occ'])
plt.title('Box Plot of Occ with Outliers')
plt.ylabel('Occ')
plt.show()

# ## Outlier Treatment (Capping) untuk flow, occ


df['flow_capped'] = df['flow'].clip(lower=lower_bound, upper=upper_bound)
print(f"Original 'flow' min: {df['flow'].min()}, max: {df['flow'].max()}")
print(f"Capped 'flow_capped' min: {df['flow_capped'].min()}, max: {df['flow_capped'].max()}")

df['occ_capped'] = df['occ'].clip(lower=lower_bound_occ, upper=upper_bound_occ)
print(f"Original 'occ' min: {df['occ'].min()}, max: {df['occ'].max()}")
print(f"Capped 'occ_capped' min: {df['occ_capped'].min()}, max: {df['occ_capped'].max()}")

import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))
plt.hist(df['flow_capped'], bins=100)
plt.title("Distribusi Flow (Capped)")
plt.xlabel("Flow (Capped)")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(12,5))
plt.hist(df['occ_capped'], bins=100)
plt.title("Distribusi Occ (Capped)")
plt.xlabel("Occ (Capped)")
plt.ylabel("Frequency")
plt.show()

# ## Verifikasi Hasil Capping


print("Statistik Deskriptif untuk 'flow_capped':")
print(df['flow_capped'].describe())

print("Statistik Deskriptif untuk 'occ_capped':")
print(df['occ_capped'].describe())

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.boxplot(df['flow_capped'])
plt.title('Box Plot of Flow (Capped) with Outliers')
plt.ylabel('Flow Capped')
plt.show()

plt.figure(figsize=(10, 6))
plt.boxplot(df['occ_capped'])
plt.title('Box Plot of Occ (Capped) with Outliers')
plt.ylabel('Occ Capped')
plt.show()

# # Klasifikasi Macet/Lancar


# Mendefinisikan target variabel
# 


df['traffic_status'] = np.where(df['occ_capped'] > 0.1, 'macet', 'lancar')
print(df[['occ_capped', 'traffic_status']].head(500))

#INI HANYA UNTUK TESTING(TIDAK JADI DIGUNAKAN PADA X TRAIN/TEST)

# OCC Tidak jadi digunakan karena Target Leakage
# 
#  Definisi: Mendefinisikan "macet" sebagai $occ \ge 0.11$.Kebocoran: kemudian memberikan kolom $occ$ (atau $occ\_capped$) sebagai fitur ke model.Hasil: Model tidak belajar memprediksi macet; ia hanya melihat fitur $occ$ dan langsung tahu jawabannya dengan kepastian $100\%$ karena jawabannya sudah terkandung dalam fitur.


# # Training (RandomForest)


X = df[['flow_capped']]
y = df['traffic_status']

print("Features (X) head after revision:")
print(X.head())
print("\nTarget (y) head:")
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train shape after revision: {X_train.shape}")
print(f"X_test shape after revision: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

print("RandomForestClassifier model retrained successfully with revised features.")

# # Testing & Evaluasi




y_pred = model.predict(X_test)

accuracy_revised = accuracy_score(y_test, y_pred)
print(f"Accuracy after feature revision: {accuracy_revised:.4f}")

print("\nClassification Report after feature revision:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix after feature revision:")
print(confusion_matrix(y_test, y_pred))

# Jadi, terlihat bahwa kelas macet memiliki akurasi lebih rendah karena datanya hanya 183064, berbanding terbalik dengan kelas lancar


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['lancar', 'macet'], yticklabels=['lancar', 'macet'])
plt.title('Confusion Matrix after Feature Revision')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

report = classification_report(y_test, y_pred, output_dict=True)

df_report = pd.DataFrame(report).transpose()

df_report = df_report.loc[['lancar', 'macet']]

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

sns.barplot(x=df_report.index, y='precision', data=df_report, ax=axes[0], palette='viridis', hue=df_report.index, legend=False)
axes[0].set_title('Precision by Class')
axes[0].set_ylabel('Score')

sns.barplot(x=df_report.index, y='recall', data=df_report, ax=axes[1], palette='viridis', hue=df_report.index, legend=False)
axes[1].set_title('Recall by Class')
axes[1].set_ylabel('Score')

sns.barplot(x=df_report.index, y='f1-score', data=df_report, ax=axes[2], palette='viridis', hue=df_report.index, legend=False)
axes[2].set_title('F1-Score by Class')
axes[2].set_ylabel('Score')

plt.suptitle('Model Performance Metrics by Traffic Status', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ## Training (K-Nearest Neighbors)


from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train, y_train)

print("K-Nearest Neighbors model trained successfully.")

# ## Testing & Evaluasi (K-Nearest Neighbors)


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred_knn = knn_model.predict(X_test)

accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"Accuracy for KNN model: {accuracy_knn:.4f}")

print("\nClassification Report for KNN model:")
print(classification_report(y_test, y_pred_knn))

print("\nConfusion Matrix for KNN model:")
print(confusion_matrix(y_test, y_pred_knn))

import matplotlib.pyplot as plt
import seaborn as sns

cm_knn = confusion_matrix(y_test, y_pred_knn)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Greens', xticklabels=['lancar', 'macet'], yticklabels=['lancar', 'macet'])
plt.title('Confusion Matrix for KNN Model')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

report_knn = classification_report(y_test, y_pred_knn, output_dict=True)

df_report_knn = pd.DataFrame(report_knn).transpose()
df_report_knn = df_report_knn.loc[['lancar', 'macet']]

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

sns.barplot(x=df_report_knn.index, y='precision', data=df_report_knn, ax=axes[0], palette='viridis', hue=df_report_knn.index, legend=False)
axes[0].set_title('Precision by Class (KNN)')
axes[0].set_ylabel('Score')

sns.barplot(x=df_report_knn.index, y='recall', data=df_report_knn, ax=axes[1], palette='viridis', hue=df_report_knn.index, legend=False)
axes[1].set_title('Recall by Class (KNN)')
axes[1].set_ylabel('Score')

sns.barplot(x=df_report_knn.index, y='f1-score', data=df_report_knn, ax=axes[2], palette='viridis', hue=df_report_knn.index, legend=False)
axes[2].set_title('F1-Score by Class (KNN)')
axes[2].set_ylabel('Score')

plt.suptitle('KNN Model Performance Metrics by Traffic Status', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ## Model Comparison


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

report_rf = classification_report(y_test, y_pred, output_dict=True)
report_knn = classification_report(y_test, y_pred_knn, output_dict=True)


accuracy_rf = report_rf['accuracy']
accuracy_knn = report_knn['accuracy']

f1_macet_rf = report_rf['macet']['f1-score']
f1_lancar_rf = report_rf['lancar']['f1-score']
f1_macet_knn = report_knn['macet']['f1-score']
f1_lancar_knn = report_knn['lancar']['f1-score']


comparison_df = pd.DataFrame({
    'Metric': ['Accuracy', 'F1-Score (Macet)',' F1-Score (Lancar)']
}).set_index('Metric')

comparison_df['RandomForest'] = [accuracy_rf, f1_macet_rf,f1_lancar_rf]
comparison_df['K-Nearest Neighbors'] = [accuracy_knn, f1_macet_knn,f1_lancar_knn]

print("\nModel Performance Comparison:")
print(comparison_df)

comparison_df.plot(kind='bar', figsize=(10, 6))
plt.title('Comparison of RandomForest vs. KNN Model Performance')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.legend(title='Model')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()