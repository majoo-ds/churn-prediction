import streamlit as st
import pandas as pd
from xgboost import XGBClassifier


st.set_page_config(layout="wide", page_title="Single Predictor", page_icon="📈")
st.markdown("# Single Predictor")
st.markdown("__Make sure to fill all fields correctly__")

### create columns
inp1, inp2, inp3, inp4, inp5 = st.columns(5)
# field = outlet id
id = inp1.text_input("Input Outlet ID")
# field = recency
recency = inp2.text_input("Input Recency")
# field = frequency
frequency = inp3.text_input("Input Frequency")
# field = recency
monetary = inp4.text_input("Input Monetary")
# field = age
age = inp5.text_input("Input Age")


############ session state
# initiate session_state
if "id" not in st.session_state:
    st.session_state["id"] = id
if "recency" not in st.session_state:
    st.session_state["recency"] = recency
if "frequency" not in st.session_state:
    st.session_state["frequency"] = frequency
if "monetary" not in st.session_state:
    st.session_state["monetary"] = monetary
if "age" not in st.session_state:
    st.session_state["age"] = age

# submit button
submit = st.button("Predict")

# if button is pressed
if submit:
    st.session_state["id"] = id
    st.session_state["recency"] = recency
    st.session_state["frequency"] = frequency
    st.session_state["monetary"] = monetary
    st.session_state["age"] = age

    # create dataframe
    df = pd.DataFrame(
        {
            'Outlet ID': [st.session_state["id"]],
            'Recency': [int(st.session_state["recency"])],
            'Frequency': [int(st.session_state["frequency"])],
            'Monetary': [int(st.session_state["monetary"])],
            'Age': [int(st.session_state["age"])]
        }
    )

    # new instance of XGBoost model (classifier)
    model = XGBClassifier()
    # load the model
    model.load_model('classifier.json')

    # split dataframe
    df_single_train = df.drop(['Outlet ID'], axis=1)

    # make prediction (new column named Churn)
    df['Churn'] = model.predict(df_single_train)
    # make probability (new column named Potensi Churn)
    df['Potensi Churn'] = model.predict_proba(df_single_train)[:,1]
    # format Potensi Churn column
    df['Potensi Churn'] = df['Potensi Churn'].apply(lambda x: "{0:.2f}%".format(x*100))

    # show dataframe
    st.dataframe(df)
