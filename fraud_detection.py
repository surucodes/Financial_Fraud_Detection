import sqlite3
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

conn = sqlite3.connect('fraud_data.db')
query = "SELECT * FROM transaction_data"
df = pd.read_sql(query, conn)
conn.close()
print(df.head())

label_encoders = {}
for column in ['transaction_action', 'user_name', 'recipient_name']:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Extract features from the date column
df['transaction_time'] = pd.to_datetime(df['transaction_time'])
df['transaction_day'] = df['transaction_time'].dt.day
df['transaction_month'] = df['transaction_time'].dt.month
df['transaction_dayofweek'] = df['transaction_time'].dt.dayofweek

df['transaction_amount'] = (df['transaction_amount'] -
                            df['transaction_amount'].mean()) / df['transaction_amount'].std()

# Random Forest
X = df.drop(['social_security_number', 'transaction_time',
            'transaction_nature'], axis=1)
y = df['social_security_number']
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)


def predict_fraudulence(date, user_name, recipient_name, transaction_action, transaction_amount):
    input_data = pd.DataFrame({
        'user_name': [user_name],
        'transaction_amount': [transaction_amount],
        'recipient_name': [recipient_name],
        'transaction_action': [transaction_action],
        'transaction_day': [date.day],
        'transaction_month': [date.month],
        'transaction_dayofweek': [date.dayofweek]
    })
    input_data = input_data[X.columns]
    for column in ['transaction_action', 'user_name', 'recipient_name']:
        input_data[column] = label_encoders[column].transform(
            input_data[column])
    prediction = clf.predict(input_data)

    if prediction[0] == 1:
        return "Fraudulent Transaction"
    else:
        return "Legitimate Transaction"


# User input
date = pd.to_datetime(input("Enter transaction date (YYYY-MM-DD): "))
user_name = input("Enter user name: ")
recipient_name = input("Enter recipient name: ")
transaction_action = input(
    "Enter transaction action (e.g., Deposit, Transfer): ")
transaction_amount = float(input("Enter transaction amount: "))

# Predict fraudulence
result = predict_fraudulence(
    date, user_name, recipient_name, transaction_action, transaction_amount)
print("Prediction:", result)

# def load_data():
#     conn = sqlite3.connect('fraud_data.db')
#     query = "SELECT * FROM transaction_data"
#     df = pd.read_sql(query, conn)
#     conn.close()
#     return df

# def preprocess_data(df):
#     print("Original DataFrame:")
#     print(df.head())
#     print("Data types before preprocessing:")
#     print(df.dtypes)
#     df['transaction_time'] = pd.to_datetime(df['transaction_time'])
#     if 'user_name' in df.columns:
#         df = pd.get_dummies(df, columns=['user_name'], drop_first=True)
#     numeric_cols = df.select_dtypes(include=['number']).columns
#     imputer = SimpleImputer(strategy='mean')
#     df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
#     df = pd.concat(
#         [df.drop(columns=['social_security_number']), encoded_ss_df], axis=1)
#     # df['social_security_number'] = df['social_security_number'].astype(
#     #     str).fillna('Unknown')

#     encoder = OneHotEncoder(sparse=False)
#     encoded_ss_numbers = encoder.fit_transform(df[['social_security_number']])
#     encoded_ss_df = pd.DataFrame(encoded_ss_numbers, columns=[
#                                  f'social_security_number_{i}' for i in range(encoded_ss_numbers.shape[1])])

#     print("DataFrame after preprocessing:")
#     print(df.head())
#     print("Data types after preprocessing:")
#     print(df.dtypes)

#     return df


# def train_model(df):
#     target_variable = 'transaction_action'
#     if target_variable not in df.columns:
#         raise ValueError(
#             f"Column '{target_variable}' does not exist in DataFrame. Please check your DataFrame columns.")
#     df = preprocess_data(df)

#     X = df.drop(columns=[target_variable, 'transaction_time'])
#     y = df[target_variable]
#     X.columns = X.columns.astype(str)

#     # Handle missing values
#     imputer = SimpleImputer(strategy='mean')
#     X_imputed = imputer.fit_transform(X)
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_imputed, y, test_size=0.2, random_state=42)
#     clf = RandomForestClassifier(n_estimators=100, random_state=42)
#     clf.fit(X_train, y_train)

#     return clf, X_test, y_test


# def evaluate_model(clf, X_test, y_test):
#     y_pred = clf.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred, average='micro')
#     recall = recall_score(y_test, y_pred, average='micro')
#     f1 = f1_score(y_test, y_pred, average='micro')

#     return accuracy, precision, recall, f1


# def load_model():
#     model = RandomForestClassifier()
#     return model


# def preprocess_input(input_data):
#     df = pd.DataFrame(input_data, index=[0])
#     return df


# def predict(model, input_data):
#     input_df = preprocess_input(input_data)
#     conn = sqlite3.connect('fraud_data.db')
#     query = "SELECT * FROM transaction_data"
#     df = pd.read_sql(query, conn)
#     conn.close()

#     df = preprocess_data(df)
#     X = df.drop(columns=['transaction_action'])
#     y = df['transaction_action']
#     model.fit(X, y)
#     prediction = model.predict(input_df)
#     return prediction
