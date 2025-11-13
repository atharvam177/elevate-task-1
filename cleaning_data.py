import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = '#f8f9fa'
plt.rcParams['axes.facecolor'] = '#ffffff'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

COLORS = {
    'primary': '#3498db',
    'secondary': '#e74c3c',
    'success': '#2ecc71',
    'warning': '#f39c12',
    'info': '#9b59b6',
    'gradient': ['#667eea', '#764ba2', '#f093fb', '#4facfe']
}

print("=" * 80)
print("🚢 STEP 1: LOADING AND EXPLORING THE TITANIC DATASET")
print("=" * 80)

import kagglehub
import os

print("\n📥 Downloading dataset from Kaggle...")
path = kagglehub.dataset_download("yasserh/titanic-dataset")
print(f"✓ Dataset downloaded to: {path}")

csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
if csv_files:
    csv_file = os.path.join(path, csv_files[0])
    print(f"✓ Found CSV file: {csv_files[0]}")
else:
    raise FileNotFoundError("No CSV file found in the downloaded dataset")

df = pd.read_csv(csv_file)

print("\n📊 First 5 rows of the dataset:")
print(df.head())

print("\n📋 Dataset Information:")
print(df.info())

print("\n📈 Dataset Shape:", df.shape)
print(f"   - Rows: {df.shape[0]}")
print(f"   - Columns: {df.shape[1]}")

print("\n📊 Statistical Summary:")
print(df.describe())

fig = plt.figure(figsize=(16, 10))
fig.suptitle('🚢 Titanic Dataset Overview', fontsize=20, fontweight='bold', y=0.98)

ax1 = plt.subplot(2, 3, 1)
survival_counts = df['Survived'].value_counts()
colors_survival = [COLORS['secondary'], COLORS['success']]
wedges, texts, autotexts = ax1.pie(survival_counts, labels=['Died', 'Survived'], 
                                     autopct='%1.1f%%', startangle=90,
                                     colors=colors_survival, explode=(0.05, 0.05),
                                     shadow=True, textprops={'fontsize': 12, 'weight': 'bold'})
ax1.set_title('Survival Rate', fontsize=14, fontweight='bold', pad=10)

ax2 = plt.subplot(2, 3, 2)
gender_counts = df['Sex'].value_counts()
bars = ax2.bar(gender_counts.index, gender_counts.values, 
               color=[COLORS['primary'], COLORS['info']], alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_title('Gender Distribution', fontsize=14, fontweight='bold', pad=10)
ax2.set_ylabel('Count', fontweight='bold')
ax2.grid(axis='y', alpha=0.3, linestyle='--')
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom', fontweight='bold', fontsize=11)

ax3 = plt.subplot(2, 3, 3)
class_counts = df['Pclass'].value_counts().sort_index()
bars = ax3.bar(['1st Class', '2nd Class', '3rd Class'], class_counts.values,
               color=COLORS['gradient'][:3], alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_title('Passenger Class Distribution', fontsize=14, fontweight='bold', pad=10)
ax3.set_ylabel('Count', fontweight='bold')
ax3.grid(axis='y', alpha=0.3, linestyle='--')
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom', fontweight='bold', fontsize=11)

ax4 = plt.subplot(2, 3, 4)
ax4.hist(df['Age'].dropna(), bins=30, color=COLORS['primary'], 
         alpha=0.7, edgecolor='black', linewidth=1.2)
ax4.axvline(df['Age'].median(), color=COLORS['secondary'], linestyle='--', 
            linewidth=2, label=f'Median: {df["Age"].median():.1f}')
ax4.set_title('Age Distribution', fontsize=14, fontweight='bold', pad=10)
ax4.set_xlabel('Age', fontweight='bold')
ax4.set_ylabel('Frequency', fontweight='bold')
ax4.legend(frameon=True, shadow=True)
ax4.grid(alpha=0.3, linestyle='--')

ax5 = plt.subplot(2, 3, 5)
ax5.hist(df['Fare'].dropna(), bins=30, color=COLORS['success'], 
         alpha=0.7, edgecolor='black', linewidth=1.2)
ax5.axvline(df['Fare'].median(), color=COLORS['secondary'], linestyle='--', 
            linewidth=2, label=f'Median: ${df["Fare"].median():.2f}')
ax5.set_title('Fare Distribution', fontsize=14, fontweight='bold', pad=10)
ax5.set_xlabel('Fare ($)', fontweight='bold')
ax5.set_ylabel('Frequency', fontweight='bold')
ax5.legend(frameon=True, shadow=True)
ax5.grid(alpha=0.3, linestyle='--')

ax6 = plt.subplot(2, 3, 6)
embarked_counts = df['Embarked'].value_counts()
colors_embarked = [COLORS['warning'], COLORS['info'], COLORS['success']]
bars = ax6.bar(['Southampton', 'Cherbourg', 'Queenstown'], 
               [embarked_counts.get('S', 0), embarked_counts.get('C', 0), embarked_counts.get('Q', 0)],
               color=colors_embarked, alpha=0.8, edgecolor='black', linewidth=1.5)
ax6.set_title('Port of Embarkation', fontsize=14, fontweight='bold', pad=10)
ax6.set_ylabel('Count', fontweight='bold')
ax6.grid(axis='y', alpha=0.3, linestyle='--')
plt.setp(ax6.xaxis.get_majorticklabels(), rotation=15, ha='right')
for bar in bars:
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.show()

print("\n" + "=" * 80)
print("🔍 STEP 2: DETECTING AND HANDLING MISSING VALUES")
print("=" * 80)

print("\n❌ Missing Values Count:")
missing_values = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({
    'Column': missing_values.index,
    'Missing Count': missing_values.values,
    'Percentage': missing_percent.values
})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Percentage', ascending=False)
print(missing_df)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('📊 Missing Values Analysis', fontsize=8, fontweight='bold', y=1.02)

if not missing_df.empty:
    bars = ax1.barh(missing_df['Column'], missing_df['Percentage'], 
                    color=COLORS['gradient'], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Missing Percentage (%)', fontweight='bold', fontsize=12)
    ax1.set_title('Missing Values by Column', fontsize=14, fontweight='bold', pad=10)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    for i, (bar, pct) in enumerate(zip(bars, missing_df['Percentage'])):
        ax1.text(pct + 1, bar.get_y() + bar.get_height()/2, 
                f'{pct:.1f}%', va='center', fontweight='bold', fontsize=11)
    
    ax1.axvline(x=50, color=COLORS['warning'], linestyle='--', 
                linewidth=2, alpha=0.5, label='50% threshold')
    ax1.legend(frameon=True, shadow=True)

ax2_data = df.isnull().astype(int)
sns.heatmap(ax2_data.T, cmap=['#2ecc71', '#e74c3c'], cbar=False, 
            ax=ax2, yticklabels=True, xticklabels=False)
ax2.set_title('Missing Data Heatmap\n(Green=Present, Red=Missing)', 
              fontsize=14, fontweight='bold', pad=10)
ax2.set_xlabel('Passenger Index', fontweight='bold')
ax2.set_ylabel('Features', fontweight='bold')

plt.tight_layout()
plt.show()

print("\n🔧 Handling Missing Values...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('🔄 Missing Values: Before & After Treatment', fontsize=18, fontweight='bold', y=0.98)

before_missing = df.isnull().sum()
before_missing = before_missing[before_missing > 0]

if 'Age' in df.columns and df['Age'].isnull().any():
    axes[0, 0].hist(df['Age'].dropna(), bins=30, color=COLORS['primary'], 
                    alpha=0.7, edgecolor='black', label='Original Data')
    median_age = df['Age'].median()
    df['Age'].fillna(median_age, inplace=True)
    axes[0, 0].axvline(median_age, color=COLORS['secondary'], linestyle='--', 
                      linewidth=2, label=f'Filled with Median: {median_age:.1f}')
    axes[0, 0].set_title('Age - Median Imputation', fontsize=13, fontweight='bold')
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend(frameon=True, shadow=True)
    axes[0, 0].grid(alpha=0.3)
    print(f"   ✓ Age: Filled {before_missing['Age']} missing values with median ({median_age:.2f})")

if 'Embarked' in df.columns and df['Embarked'].isnull().any():
    embarked_before = df['Embarked'].value_counts()
    mode_embarked = df['Embarked'].mode()[0]
    df['Embarked'].fillna(mode_embarked, inplace=True)
    
    embarked_after = df['Embarked'].value_counts()
    x = np.arange(len(embarked_after))
    width = 0.35
    
    axes[0, 1].bar(x - width/2, [embarked_before.get(i, 0) for i in embarked_after.index], 
                  width, label='Before', color=COLORS['warning'], alpha=0.8, edgecolor='black')
    axes[0, 1].bar(x + width/2, embarked_after.values, width, 
                  label='After', color=COLORS['success'], alpha=0.8, edgecolor='black')
    axes[0, 1].set_title(f'Embarked - Mode Imputation (Mode: {mode_embarked})', 
                        fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel('Port')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(embarked_after.index)
    axes[0, 1].legend(frameon=True, shadow=True)
    axes[0, 1].grid(axis='y', alpha=0.3)
    print(f"   ✓ Embarked: Filled {before_missing.get('Embarked', 0)} missing values with mode ('{mode_embarked}')")

if 'Cabin' in df.columns:
    cabin_stats = pd.DataFrame({
        'Category': ['Has Cabin', 'No Cabin'],
        'Count': [df['Cabin'].notna().sum(), df['Cabin'].isna().sum()]
    })
    
    df['Has_Cabin'] = df['Cabin'].notna().astype(int)
    
    colors_cabin = [COLORS['success'], COLORS['secondary']]
    wedges, texts, autotexts = axes[1, 0].pie(cabin_stats['Count'], labels=cabin_stats['Category'],
                                               autopct='%1.1f%%', colors=colors_cabin,
                                               explode=(0.05, 0.05), shadow=True, startangle=90,
                                               textprops={'fontsize': 11, 'weight': 'bold'})
    axes[1, 0].set_title('Cabin - Binary Feature Creation', fontsize=13, fontweight='bold')
    df.drop('Cabin', axis=1, inplace=True)
    print(f"   ✓ Cabin: Created 'Has_Cabin' binary feature and dropped original column")

if 'Fare' in df.columns and df['Fare'].isnull().any():
    axes[1, 1].hist(df['Fare'].dropna(), bins=30, color=COLORS['success'], 
                    alpha=0.7, edgecolor='black', label='Original Data')
    median_fare = df['Fare'].median()
    df['Fare'].fillna(median_fare, inplace=True)
    axes[1, 1].axvline(median_fare, color=COLORS['secondary'], linestyle='--', 
                      linewidth=2, label=f'Filled with Median: ${median_fare:.2f}')
    axes[1, 1].set_title('Fare - Median Imputation', fontsize=13, fontweight='bold')
    axes[1, 1].set_xlabel('Fare ($)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend(frameon=True, shadow=True)
    axes[1, 1].grid(alpha=0.3)
    print(f"   ✓ Fare: Filled missing values with median (${median_fare:.2f})")
else:
    after_missing = df.isnull().sum().sum()
    summary_text = f"✅ All Missing Values Handled!\n\n"
    summary_text += f"Before: {before_missing.sum()} missing values\n"
    summary_text += f"After: {after_missing} missing values\n\n"
    summary_text += f"Reduction: {before_missing.sum() - after_missing} values"
    
    axes[1, 1].text(0.5, 0.5, summary_text, 
                   ha='center', va='center', fontsize=14, 
                   bbox=dict(boxstyle='round,pad=1', facecolor=COLORS['success'], 
                            alpha=0.3, edgecolor='black', linewidth=2),
                   fontweight='bold')
    axes[1, 1].axis('off')

plt.tight_layout()
plt.show()

print("\n✅ Missing values after handling:")
print(f"   Total: {df.isnull().sum().sum()} missing values")

print("\n" + "=" * 80)
print("🔢 STEP 3: ENCODING CATEGORICAL FEATURES")
print("=" * 80)

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"\n📝 Categorical columns found: {categorical_cols}")

fig = plt.figure(figsize=(16, 10))
fig.suptitle('🔤 Categorical Feature Encoding', fontsize=18, fontweight='bold', y=0.98)

if 'Sex' in df.columns:
    ax1 = plt.subplot(2, 2, 1)
    sex_counts = df['Sex'].value_counts()
    bars = ax1.bar(sex_counts.index, sex_counts.values, 
                   color=[COLORS['primary'], COLORS['info']], alpha=0.8, 
                   edgecolor='black', linewidth=1.5)
    ax1.set_title('Sex - Original Categories', fontsize=13, fontweight='bold', pad=10)
    ax1.set_ylabel('Count', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    le_sex = LabelEncoder()
    df['Sex_Encoded'] = le_sex.fit_transform(df['Sex'])
    
    ax2 = plt.subplot(2, 2, 2)
    encoded_counts = df['Sex_Encoded'].value_counts().sort_index()
    bars = ax2.bar([f'{i}\n({le_sex.classes_[i]})' for i in encoded_counts.index], 
                   encoded_counts.values,
                   color=[COLORS['primary'], COLORS['info']], alpha=0.8, 
                   edgecolor='black', linewidth=1.5)
    ax2.set_title('Sex - After Label Encoding', fontsize=13, fontweight='bold', pad=10)
    ax2.set_ylabel('Count', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    print(f"\n   ✓ Sex: Label Encoded")
    print(f"      Mapping: {dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_)))}")

if 'Embarked' in df.columns:
    ax3 = plt.subplot(2, 2, 3)
    embarked_counts = df['Embarked'].value_counts()
    bars = ax3.bar(['Southampton (S)', 'Cherbourg (C)', 'Queenstown (Q)'],
                   [embarked_counts.get('S', 0), embarked_counts.get('C', 0), 
                    embarked_counts.get('Q', 0)],
                   color=[COLORS['warning'], COLORS['info'], COLORS['success']], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_title('Embarked - Original Categories', fontsize=13, fontweight='bold', pad=10)
    ax3.set_ylabel('Count', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=15, ha='right')
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked', drop_first=True)
    df = pd.concat([df, embarked_dummies], axis=1)
    
    ax4 = plt.subplot(2, 2, 4)
    dummy_data = pd.DataFrame({
        'Embarked_Q': embarked_dummies['Embarked_Q'].sum() if 'Embarked_Q' in embarked_dummies else 0,
        'Embarked_S': embarked_dummies['Embarked_S'].sum() if 'Embarked_S' in embarked_dummies else 0
    }, index=[0])
    
    dummy_data.T.plot(kind='barh', ax=ax4, legend=False, 
                     color=[COLORS['success'], COLORS['warning']], 
                     alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_title('Embarked - After One-Hot Encoding', fontsize=13, fontweight='bold', pad=10)
    ax4.set_xlabel('Count (1s in binary columns)', fontweight='bold')
    ax4.set_ylabel('New Binary Features', fontweight='bold')
    ax4.grid(axis='x', alpha=0.3, linestyle='--')
    
    print(f"\n   ✓ Embarked: One-Hot Encoded")
    print(f"      New columns: {embarked_dummies.columns.tolist()}")
    print(f"      Note: 'Embarked_C' is baseline (all 0s when both are 0)")

plt.tight_layout()
plt.show()

columns_to_drop = ['Sex', 'Embarked', 'Name', 'Ticket']
existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
if existing_cols_to_drop:
    df.drop(existing_cols_to_drop, axis=1, inplace=True)
    print(f"\n   ✓ Dropped original categorical columns: {existing_cols_to_drop}")

print(f"\n📊 Dataset shape after encoding: {df.shape}")

print("\n" + "=" * 80)
print("📉 STEP 4: DETECTING AND VISUALIZING OUTLIERS")
print("=" * 80)

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
outlier_cols = [col for col in numerical_cols if col not in ['Survived', 'Pclass', 'Sex_Encoded', 
                                                               'Has_Cabin', 'Embarked_Q', 'Embarked_S']]

print(f"\n🔍 Analyzing outliers in: {outlier_cols}")

fig, axes = plt.subplots(2, len(outlier_cols), figsize=(16, 10))
fig.suptitle('📊 Outlier Detection Analysis', fontsize=18, fontweight='bold', y=0.98)

if len(outlier_cols) == 1:
    axes = axes.reshape(-1, 1)

def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound, Q1, Q3

print("\n📊 Outlier Summary (IQR Method):")

for idx, col in enumerate(outlier_cols):
    outliers, lower, upper, Q1, Q3 = detect_outliers_iqr(df, col)
    outlier_count = len(outliers)
    outlier_percent = (outlier_count / len(df)) * 100
    
    bp = axes[0, idx].boxplot(df[col].dropna(), patch_artist=True, widths=0.6,
                               boxprops=dict(facecolor=COLORS['primary'], alpha=0.7, 
                                           edgecolor='black', linewidth=1.5),
                               whiskerprops=dict(color='black', linewidth=1.5),
                               capprops=dict(color='black', linewidth=1.5),
                               medianprops=dict(color=COLORS['secondary'], linewidth=2),
                               flierprops=dict(marker='o', markerfacecolor=COLORS['secondary'], 
                                             markersize=6, alpha=0.7, markeredgecolor='black'))
    axes[0, idx].set_title(f'{col}\nBoxplot View', fontweight='bold', fontsize=12)
    axes[0, idx].set_ylabel('Value', fontweight='bold')
    axes[0, idx].grid(axis='y', alpha=0.3, linestyle='--')
    
    axes[0, idx].text(1.15, axes[0, idx].get_ylim()[1] * 0.95, 
                     f'Outliers: {outlier_count}\n({outlier_percent:.1f}%)',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['warning'], 
                              alpha=0.3, edgecolor='black'),
                     fontsize=10, fontweight='bold')
    
    axes[1, idx].hist(df[col].dropna(), bins=30, color=COLORS['primary'], 
                     alpha=0.7, edgecolor='black', linewidth=1.2)
    axes[1, idx].axvline(lower, color=COLORS['secondary'], linestyle='--', 
                        linewidth=2, label=f'Lower: {lower:.1f}', alpha=0.8)
    axes[1, idx].axvline(upper, color=COLORS['secondary'], linestyle='--', 
                        linewidth=2, label=f'Upper: {upper:.1f}', alpha=0.8)
    axes[1, idx].axvline(Q1, color=COLORS['success'], linestyle=':', 
                        linewidth=1.5, label=f'Q1: {Q1:.1f}', alpha=0.6)
    axes[1, idx].axvline(Q3, color=COLORS['success'], linestyle=':', 
                        linewidth=1.5, label=f'Q3: {Q3:.1f}', alpha=0.6)
    axes[1, idx].set_title(f'{col}\nDistribution with IQR', fontweight='bold', fontsize=12)
    axes[1, idx].set_xlabel('Value', fontweight='bold')
    axes[1, idx].set_ylabel('Frequency', fontweight='bold')
    axes[1, idx].legend(loc='upper right', fontsize=8, frameon=True, shadow=True)
    axes[1, idx].grid(alpha=0.3, linestyle='--')
    
    print(f"\n   {col}:")
    print(f"      - Total values: {len(df[col].dropna())}")
    print(f"      - Outliers: {outlier_count} ({outlier_percent:.2f}%)")
    print(f"      - IQR Range: [{lower:.2f}, {upper:.2f}]")
    print(f"      - Quartiles: Q1={Q1:.2f}, Q3={Q3:.2f}")

plt.tight_layout()
plt.show()

print("\n" + "=" * 80)
print("🗑️  STEP 5: REMOVING OUTLIERS (STRATEGIC APPROACH)")
print("=" * 80)

print("\n⚠️  Note: Removing outliers can improve model performance but may lose information.")
print("    We'll remove extreme outliers from 'Fare' only for demonstration.")

original_size = len(df)

if 'Fare' in df.columns:
    outliers, lower, upper, _, _ = detect_outliers_iqr(df, 'Fare')
    df_cleaned = df[(df['Fare'] >= lower) & (df['Fare'] <= upper)]
    removed_count = len(df) - len(df_cleaned)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('🎯 Outlier Removal Impact on Fare Distribution', 
                fontsize=16, fontweight='bold', y=1.02)
    
    ax1.hist(df['Fare'], bins=40, color=COLORS['warning'], alpha=0.7, 
            edgecolor='black', linewidth=1.2)
    ax1.axvline(lower, color=COLORS['secondary'], linestyle='--', linewidth=2, 
               label=f'Lower Bound: ${lower:.2f}')
    ax1.axvline(upper, color=COLORS['secondary'], linestyle='--', linewidth=2,
               label=f'Upper Bound: ${upper:.2f}')
    ax1.set_title(f'Before Removal\n({original_size} passengers)', 
                 fontweight='bold', fontsize=13)
    ax1.set_xlabel('Fare ($)', fontweight='bold')
    ax1.set_ylabel('Frequency', fontweight='bold')
    ax1.legend(frameon=True, shadow=True)
    ax1.grid(alpha=0.3, linestyle='--')
    
    ax2.hist(df_cleaned['Fare'], bins=40, color=COLORS['success'], alpha=0.7,
            edgecolor='black', linewidth=1.2)
    ax2.set_title(f'After Removal\n({len(df_cleaned)} passengers)', 
                 fontweight='bold', fontsize=13)
    ax2.set_xlabel('Fare ($)', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.grid(alpha=0.3, linestyle='--')
    
    summary_text = f"📊 OUTLIER REMOVAL SUMMARY\n\n"
    summary_text += f"Original Dataset: {original_size} rows\n"
    summary_text += f"After Removal: {len(df_cleaned)} rows\n"
    summary_text += f"Removed: {removed_count} rows ({(removed_count/original_size)*100:.2f}%)\n\n"
    summary_text += f"Fare Range Kept:\n"
    summary_text += f"${lower:.2f} - ${upper:.2f}"
    
    ax3.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle='round,pad=1', facecolor=COLORS['info'], 
                     alpha=0.2, edgecolor='black', linewidth=2),
            fontweight='bold', family='monospace')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    df = df_cleaned.copy()
    print(f"\n   ✓ Removed {removed_count} outlier rows ({(removed_count/original_size)*100:.2f}%)")
    print(f"      Original size: {original_size}")
    print(f"      New size: {len(df)}")

print("\n" + "=" * 80)
print("⚖️  STEP 6: FEATURE SCALING AND NORMALIZATION")
print("=" * 80)

scale_cols = [col for col in outlier_cols if col in df.columns]
print(f"\n📏 Scaling features: {scale_cols}")

fig = plt.figure(figsize=(18, 12))
fig.suptitle('📐 Feature Scaling Transformation', fontsize=20, fontweight='bold', y=0.98)

n_features = len(scale_cols)
n_rows = n_features
n_cols = 3

for idx, col in enumerate(scale_cols):
    ax1 = plt.subplot(n_rows, n_cols, idx * n_cols + 1)
    ax1.hist(df[col], bins=30, color=COLORS['primary'], alpha=0.7, 
            edgecolor='black', linewidth=1.2)
    mean_val = df[col].mean()
    std_val = df[col].std()
    ax1.axvline(mean_val, color=COLORS['secondary'], linestyle='--', 
               linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax1.set_title(f'{col} - Original\n(μ={mean_val:.2f}, σ={std_val:.2f})', 
                 fontweight='bold', fontsize=11)
    ax1.set_ylabel('Frequency', fontweight='bold')
    ax1.legend(loc='best', fontsize=9, frameon=True, shadow=True)
    ax1.grid(alpha=0.3, linestyle='--')

scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[scale_cols] = scaler.fit_transform(df[scale_cols])

print("\n✅ Standardization Applied (mean=0, std=1)")
print("\n📊 Scaled Statistics:")
print(df_scaled[scale_cols].describe().round(3))

for idx, col in enumerate(scale_cols):
    ax2 = plt.subplot(n_rows, n_cols, idx * n_cols + 2)
    ax2.hist(df_scaled[col], bins=30, color=COLORS['success'], alpha=0.7,
            edgecolor='black', linewidth=1.2)
    mean_scaled = df_scaled[col].mean()
    std_scaled = df_scaled[col].std()
    ax2.axvline(0, color=COLORS['secondary'], linestyle='--', 
               linewidth=2, label=f'Mean: {mean_scaled:.2f}')
    ax2.set_title(f'{col} - Scaled\n(μ≈0, σ≈1)', 
                 fontweight='bold', fontsize=11)
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.legend(loc='best', fontsize=9, frameon=True, shadow=True)
    ax2.grid(alpha=0.3, linestyle='--')
    
    ax3 = plt.subplot(n_rows, n_cols, idx * n_cols + 3)
    box_data = [df[col], df_scaled[col]]
    bp = ax3.boxplot(box_data, labels=['Original', 'Scaled'], patch_artist=True,
                    widths=0.6,
                    boxprops=dict(alpha=0.7, linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    medianprops=dict(color='red', linewidth=2),
                    flierprops=dict(marker='o', markersize=5, alpha=0.5))
    bp['boxes'][0].set_facecolor(COLORS['primary'])
    bp['boxes'][1].set_facecolor(COLORS['success'])
    ax3.set_title(f'{col} - Comparison', fontweight='bold', fontsize=11)
    ax3.set_ylabel('Value', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)

plt.tight_layout()
plt.show()

print("\n" + "=" * 80)
print("🎉 STEP 7: FINAL CLEANED DATASET SUMMARY")
print("=" * 80)

print("\n✅ Preprocessing Complete!")
print(f"\n📊 Final Dataset Shape: {df_scaled.shape}")
print(f"   - Rows: {df_scaled.shape[0]}")
print(f"   - Features: {df_scaled.shape[1]}")

print("\n📋 Final Columns:")
for i, col in enumerate(df_scaled.columns.tolist(), 1):
    print(f"   {i:2d}. {col}")

fig = plt.figure(figsize=(20, 12))
fig.suptitle('🎊 Final Preprocessed Dataset - Complete Analysis', 
            fontsize=20, fontweight='bold', y=0.98)

ax1 = plt.subplot(2, 2, (1, 3))
correlation_matrix = df_scaled.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
           cmap='coolwarm', center=0, square=True, linewidths=1.5,
           cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
           ax=ax1, vmin=-1, vmax=1)
ax1.set_title('Feature Correlation Matrix', fontsize=15, fontweight='bold', pad=15)

if 'Survived' in df_scaled.columns:
    ax2 = plt.subplot(2, 2, 2)
    feature_corr = correlation_matrix['Survived'].drop('Survived').sort_values(key=abs, ascending=True)
    colors = [COLORS['success'] if x > 0 else COLORS['secondary'] for x in feature_corr]
    bars = ax2.barh(feature_corr.index, feature_corr.values, color=colors, 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.axvline(0, color='black', linewidth=1.5)
    ax2.set_xlabel('Correlation with Survival', fontweight='bold', fontsize=12)
    ax2.set_title('Feature Importance\n(Correlation with Survival)', 
                 fontsize=14, fontweight='bold', pad=10)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    for bar, val in zip(bars, feature_corr.values):
        x_pos = val + (0.02 if val > 0 else -0.02)
        ax2.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                va='center', ha='left' if val > 0 else 'right', 
                fontweight='bold', fontsize=9)

ax3 = plt.subplot(2, 2, 4)
summary_stats = {
    'Total Samples': [original_size, len(df_scaled)],
    'Features': [df.shape[1], df_scaled.shape[1]],
    'Missing Values': [missing_values.sum(), df_scaled.isnull().sum().sum()],
    'Duplicates': [df.duplicated().sum(), df_scaled.duplicated().sum()]
}

summary_df = pd.DataFrame(summary_stats, index=['Before', 'After'])
x = np.arange(len(summary_stats))
width = 0.35

bars1 = ax3.bar(x - width/2, summary_df.iloc[0], width, label='Before',
               color=COLORS['warning'], alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax3.bar(x + width/2, summary_df.iloc[1], width, label='After',
               color=COLORS['success'], alpha=0.8, edgecolor='black', linewidth=1.5)

ax3.set_ylabel('Count', fontweight='bold', fontsize=12)
ax3.set_title('Data Quality: Before vs After', fontsize=14, fontweight='bold', pad=10)
ax3.set_xticks(x)
ax3.set_xticklabels(summary_stats.keys(), rotation=15, ha='right')
ax3.legend(frameon=True, shadow=True, fontsize=11)
ax3.grid(axis='y', alpha=0.3, linestyle='--')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', 
                fontweight='bold', fontsize=10)

plt.tight_layout()
plt.show()

output_file = 'titanic_cleaned.csv'
df_scaled.to_csv(output_file, index=False)
print(f"\n💾 Cleaned dataset saved as '{output_file}'")

print("\n📊 Preview of Final Cleaned Dataset:")
print("="*100)
print(df_scaled.head(10).to_string())
print("="*100)

print("\n📈 Final Dataset Statistics Summary:")
print("="*100)
print(df_scaled.describe().round(3).to_string())
print("="*100)


print("\n" + "=" * 80)
print("✨ DATA PREPROCESSING COMPLETE! ✨")
print("=" * 80)
