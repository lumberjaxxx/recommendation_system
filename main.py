import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

#---------data loading------
data = pd.read_csv("college_data.csv")

#categorical features and target label
le_gender = LabelEncoder()
le_strand = LabelEncoder()
le_status = LabelEncoder()
le_program = LabelEncoder()

data['gender'] = le_gender.fit_transform(data['gender'])
data['strand'] = le_strand.fit_transform(data['strand'])
data['socio_status'] = le_status.fit_transform(data['socio_status'])
data['college_program'] = le_program.fit_transform(data['college_program'])

# Split features (X) and target (y)
X = data[['admission_test', 'gwa', 'age', 'gender', 'strand', 'socio_status']]
y = data['college_program']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Scale numeric features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: One-hot encode target variable
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Step 8: Build the FFNN model
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),   # Hidden layer 1
    Dropout(0.3),
    Dense(32, activation='relu'),                              # Hidden layer 2
    Dropout(0.3),
    Dense(y_train.shape[1], activation='softmax')               # Output layer
])


# Step 9: Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Step 10: Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=1)


# Step 11: Evaluate performance
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")


# Step 12: Predict and get classification report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred_classes, target_names=le_program.classes_))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))