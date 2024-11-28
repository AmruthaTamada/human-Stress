import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('generated_data1.csv')

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
plt.plot(df['Snoring Rate'], df['Stress Level'], marker='o', linestyle='-', color='r')

plt.title('Snoring Rate vs Stress Level')
plt.xlabel('Snoring Rate')
plt.ylabel('Stress Level')

plt.grid(True)
plt.show()
