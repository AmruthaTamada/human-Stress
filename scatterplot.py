import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('generated_data1.csv')
sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Heart Rate', y='Stress Level', hue='Stress Level', palette='coolwarm', s=100, alpha=0.8)
plt.title('Heart Rate vs Stress Level')
plt.xlabel('Heart Rate')
plt.ylabel('Stress Level')

plt.legend(title='Stress Level')
plt.show()
