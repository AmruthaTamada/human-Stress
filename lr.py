import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

# Load the dataset

data = pd.read_csv('generated_data1.csv')

# preprocessing

X = data.drop('Stress Level', axis=1)  

y = data['Stress Level']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Normalize the features
    
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# Create the Logistic Regression model

model = LogisticRegression()

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-Score: {f1:.2f}')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

plt.title('Confusion Matrix')

plt.colorbar()

tick_marks = np.arange(2)

plt.xticks(tick_marks, ['Not Stressed', 'Stressed'])

plt.yticks(tick_marks, ['Not Stressed', 'Stressed'])



thresh=cm.max()/2

for i,j in np.ndindex(cm.shape):

    plt.text(j, i, cm[i,j],horziontalalignment="center",color="white" if cm[i,j]>thresh else "black")



plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.tight_layout()

plt.show()



joblib.dump(model, 'stress_detection_model.pkl')