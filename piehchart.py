import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('generated_data1.csv')

stress = df['Stress Level'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(stress, labels=stress.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set3'))

plt.title('Distribution of Stress Levels')
plt.axis('equal')
plt.show()

