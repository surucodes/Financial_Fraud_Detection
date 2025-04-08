import sqlite3
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title='Financial Fraud Detection System', layout='wide',
                   initial_sidebar_state='auto')


css = """
<style>
:root {
  --bg-color: #22222218;
  --text-color: rgb(45, 49, 45);
}

body {
  color: var(--text-color);
  background-color: var(--bg-color);
}


.stButton>button {
  width: 100%;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background-color: rgb(255, 255, 255);
  color: black;
  padding: 0.25rem 0.75rem;
  position: relative;
  text-decoration: none;
  border-radius: 4px;
  border-width: 1px;
  border-style: solid;
  border-color: aquamarine;
  border-image: initial;
}

.stButton>button:hover {
  border-color: rgba(9, 223, 38, 0.626);
  color: rgba(9, 223, 38, 0.626);
}

.stButton>button:active {
  box-shadow: none;
  background-color: rgba(9, 223, 38, 0.626);
  color: white;
}

.highlight {
  border-radius: 0.4rem;
  color: white;
  padding: 0.5rem;
  margin-bottom: 1rem;
}

.bold {
  padding-left: 1rem;
  font-weight: 700;
}

.blue {
  background-color: rgba(9, 223, 38, 0.626);
}

.red {
  background-color: lightblue;
}
</style>
"""


conn = sqlite3.connect('fraud_data.db')
query = "SELECT * FROM transaction_data"
df = pd.read_sql(query, conn)
conn.close()


def predict_fraudulence(date, user_name, recipient_name, transaction_amount):
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
    input_data = pd.DataFrame({
        'user_name': [user_name],
        'transaction_amount': [transaction_amount],
        'recipient_name': [recipient_name],
        'transaction_action': ["Transfer"],
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


def team_page():
    st.title('Our Team')
    st.header("Aditya Pandey (AI-A) (Registration Number: 225890264)")
    st.write("")
    st.header("Suraj Prasanna (AI-B) (Registration Number: 225890296)")
    st.write("")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


def database_page():
    st.title('Database Used')
    st.header("Transaction Data")
    conn = sqlite3.connect('fraud_data.db')
    query = "SELECT * FROM transaction_data"
    df = pd.read_sql(query, conn)
    st.write(df)
    st.markdown("<br>", unsafe_allow_html=True)
    st.header("User Account Data")
    query = "SELECT * FROM accounts_data"
    df = pd.read_sql(query, conn)
    st.write(df)
    conn.close()
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


def main():
    st.markdown(css, unsafe_allow_html=True)
    if 'app_mode' not in st.session_state:
        st.session_state['app_mode'] = 'main'
    with st.sidebar:
        st.info('**Financial Fraud Detection System**')
        team_button = st.button("Our Team")
        database_button = st.button('View Database')
        st.session_state.log_holder = st.empty()
        if team_button:
            st.session_state.app_mode = 'team'
        if database_button:
            st.session_state.app_mode = 'database'
    st.title('Financial Fraud Detection System')
    if st.session_state.app_mode == 'team':
        team_page()
    if st.session_state.app_mode == 'database':
        database_page()
    # model = load_model()

    st.write("Enter the transaction details:")
    transaction_date = st.text_input("Date of Transaction(YYYY-MM-DD):")
    # transaction_date = st.date_input("Date of Transaction:")
    user_name = st.text_input("User:")
    recipient_name = st.text_input("Recipient:")
    transaction_amount = st.number_input("Transaction Amount:")
    if st.button("Predict"):
        st.write("Evaluating the model...")
        st.write("Model evaluation complete!")
        result = predict_fraudulence(
            transaction_date, user_name, recipient_name, transaction_amount)
        st.write(f"This Transaction is a {result}!")


if __name__ == '__main__':
    main()
