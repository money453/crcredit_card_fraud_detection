import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('data/PS_20174392719_1491204439457_log.csv')

# Show basic info
print("Shape of data:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nTarget Distribution (isFraud):\n", df['isFraud'].value_counts(normalize=True))

# Save plot of class distribution
sns.countplot(x='isFraud', data=df)
plt.title("Fraud (1) vs Non-Fraud (0)")
plt.savefig('notebooks/class_distribution.png')
plt.close()

# Save transaction type vs fraud plot
sns.countplot(x='type', data=df, hue='isFraud')
plt.title("Transaction Type vs Fraud")
plt.xticks(rotation=45)
plt.savefig('notebooks/fraud_by_type.png')
plt.close()

print("EDA plots saved to 'notebooks/' folder.")
