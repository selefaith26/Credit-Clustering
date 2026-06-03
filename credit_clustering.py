"""
Credit Card Customer Segmentation
K-Means Clustering and Association Rules
Data Mining Assignment - Grand Canyon University
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('outputs', exist_ok=True)


# LOAD AND PREPROCESS DATA

print("\n  CREDIT CARD CUSTOMER SEGMENTATION")
print("  K-Means Clustering + Association Rules")
print("  ----------\n")

print("  [1] Loading and Preprocessing Data")

df = pd.read_csv('CC GENERAL.csv')
df = df.drop(columns=['CUST_ID'])

print(f"  Dataset shape    : {df.shape}")
print(f"  Missing values   : {df.isnull().sum().sum()}")

# Fill missing values with median
df['CREDIT_LIMIT']    = df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT'].median())
df['MINIMUM_PAYMENTS']= df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].median())

print(f"  After cleaning   : {df.isnull().sum().sum()} missing values")

# Select key features for clustering
features = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE',
            'CREDIT_LIMIT', 'PAYMENTS', 'PRC_FULL_PAYMENT']

X = df[features].copy()

# Standardize
scaler = StandardScaler()
X_std  = scaler.fit_transform(X)

print(f"  Features used    : {features}")


# K-MEANS — FIND OPTIMAL K USING ELBOW METHOD

print("\n  [2] Finding Optimal Number of Clusters (Elbow Method)")

inertia    = []
sil_scores = []
K_range    = range(2, 10)

for k in K_range:
    km  = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_std)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(X_std, km.labels_))
    print(f"  k={k}  inertia={km.inertia_:.0f}  silhouette={sil_scores[-1]:.3f}")

# Plot 1 — Elbow curve
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(K_range, inertia, marker='o', color='steelblue')
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method')

axes[1].plot(K_range, sil_scores, marker='o', color='tomato')
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Scores')

plt.tight_layout()
plt.savefig('outputs/plot1_elbow_silhouette.png')
plt.close()
print("\n  Plot saved: plot1_elbow_silhouette.png")


# K-MEANS — BUILD FINAL MODEL

print("\n  [3] Building K-Means Model (k=4)")

km_final = KMeans(n_clusters=4, random_state=42, n_init=10)
km_final.fit(X_std)
df['Cluster'] = km_final.labels_

print(f"  Cluster distribution:")
for i in range(4):
    count = (df['Cluster'] == i).sum()
    print(f"    Cluster {i}: {count} customers ({count/len(df)*100:.1f}%)")

# Cluster profiles
print(f"\n  Cluster Profiles (mean values):")
profile = df.groupby('Cluster')[features].mean().round(2)
print(profile.to_string())


# PLOT 2 — CLUSTER VISUALIZATION USING PCA

pca   = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

plt.figure(figsize=(8, 6))
colors = ['steelblue', 'tomato', 'green', 'orange']
labels = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3']
for i in range(4):
    mask = df['Cluster'] == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=colors[i], label=labels[i], alpha=0.4, s=10)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('K-Means Customer Segments (PCA View)')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/plot2_kmeans_clusters.png')
plt.close()
print("\n  Plot saved: plot2_kmeans_clusters.png")


# PLOT 3 — CLUSTER PROFILE BAR CHART

profile_norm = (profile - profile.min()) / (profile.max() - profile.min())
fig, ax = plt.subplots(figsize=(10, 5))
profile_norm.T.plot(kind='bar', ax=ax, color=colors, edgecolor='white')
ax.set_xlabel('Feature')
ax.set_ylabel('Normalized Mean Value')
ax.set_title('Cluster Profiles — Normalized Feature Comparison')
ax.legend([f'Cluster {i}' for i in range(4)])
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('outputs/plot3_cluster_profiles.png')
plt.close()
print("  Plot saved: plot3_cluster_profiles.png")


# ASSOCIATION RULES

print("\n  [4] Building Association Rules Model")

# Bin continuous features into categories for association analysis
df_assoc = pd.DataFrame()
df_assoc['High_Balance']    = (df['BALANCE']           > df['BALANCE'].median()).astype(int)
df_assoc['High_Purchases']  = (df['PURCHASES']         > df['PURCHASES'].median()).astype(int)
df_assoc['High_CashAdv']    = (df['CASH_ADVANCE']       > df['CASH_ADVANCE'].median()).astype(int)
df_assoc['High_CreditLim']  = (df['CREDIT_LIMIT']       > df['CREDIT_LIMIT'].median()).astype(int)
df_assoc['High_Payments']   = (df['PAYMENTS']           > df['PAYMENTS'].median()).astype(int)
df_assoc['Full_Payer']      = (df['PRC_FULL_PAYMENT']   > 0.5).astype(int)

# Convert to boolean for apriori
df_bool = df_assoc.astype(bool)

# Mine frequent itemsets
frequent_items = apriori(df_bool, min_support=0.3, use_colnames=True)
rules          = association_rules(frequent_items, metric='lift', min_threshold=1.2)
rules          = rules.sort_values('lift', ascending=False)

print(f"  Frequent itemsets found : {len(frequent_items)}")
print(f"  Association rules found : {len(rules)}")
print(f"\n  Top 5 rules by lift:")
print(f"  {'Antecedent':<25} {'Consequent':<20} {'Support':>9} {'Confidence':>11} {'Lift':>7}")
print(f"  {'─'*76}")
for _, row in rules.head(5).iterrows():
    ant = ', '.join(list(row['antecedents']))
    con = ', '.join(list(row['consequents']))
    print(f"  {ant:<25} {con:<20} {row['support']:>9.3f} {row['confidence']:>11.3f} {row['lift']:>7.3f}")


# PLOT 4 — ASSOCIATION RULES SCATTER

plt.figure(figsize=(8, 5))
plt.scatter(rules['support'], rules['confidence'],
            c=rules['lift'], cmap='RdYlGn', alpha=0.7, s=50)
plt.colorbar(label='Lift')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Association Rules — Support vs Confidence (color = Lift)')
plt.tight_layout()
plt.savefig('outputs/plot4_association_rules.png')
plt.close()
print("\n  Plot saved: plot4_association_rules.png")


# FINAL SUMMARY

sil_final = silhouette_score(X_std, km_final.labels_)

print(f"\n  ----------")
print(f"  FINAL SUMMARY")
print(f"  ----------")
print(f"  K-Means clusters         : 4")
print(f"  Silhouette score         : {sil_final:.3f}")
print(f"  Association rules found  : {len(rules)}")
print(f"  Top lift score           : {rules['lift'].max():.3f}")
print(f"  ----------\n")
print("  Done. Plots saved to outputs/ folder.\n")
