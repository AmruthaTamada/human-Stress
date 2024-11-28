import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('generated_data1.csv')
selected_columns = ['Snoring Rate', 'Heart Rate', 'Blood Oxygen', 'Sleep Hours', 'Limb Movements']

correlation = df[selected_columns].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=1, linecolor='white', cbar_kws={"shrink": 0.8})

plt.title('Correlation Heatmap of Selected Parameters', fontsize=16)
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.show()



