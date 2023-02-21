import streamlit as st
import pandas as pd
import numpy as np
import gspread
import gspread_dataframe
from google.oauth2.service_account import Credentials
import datetime
from tableau_api_lib import TableauServerConnection
from tableau_api_lib.utils import querying
from xgboost import XGBClassifier
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
from PIL import Image
import plotly.express as px


# favicon image
im = Image.open("favicon.ico")

st.set_page_config(
    page_title="Auto Churn Predictor",
    page_icon=im,
)

st.write("# Auto Churn Predictor! 🕵️‍♂️")

# authorize access to your Google Sheets account
credentials = Credentials.from_service_account_info(
    st.secrets["gcp_service_account"],
    scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ],
)

# authorize access to Tableau
tableau_server_config = {
        'tableau_prod': {
                'server': 'https://prod-apsoutheast-a.online.tableau.com',
                'api_version': '3.18',
                'personal_access_token_name': 'query-view',
                'personal_access_token_secret': '/33zGi0aQK+KhhmN/7g81g==:ky0CgfT27jiMtwmE2hywmib9xkPW5AxZ',
                'site_name': 'majooinsight',
                'site_url': 'majooinsight'
        }
}
# create connection
conn = TableauServerConnection(tableau_server_config)
conn.sign_in()

# existing sheets (churn processing)
@st.cache_data(ttl=600)
def get_sheets():
    # Create a connection object.
    gc = gspread.authorize(credentials)
    # open the google sheets file
    read = gc.open("Churn Processing 2023")
    # define the sheet name
    worksheet = read.worksheet(title='Sheet1')
    # read as pandas dataframe
    df = gspread_dataframe.get_as_dataframe(worksheet=worksheet, index=False, evaluate_formulas=True)
    # data manipulation
    df['ID Outlet'] = df['ID Outlet'].astype('str')
    
    return df

dataframe = get_sheets()

# existing database from Tableau
@st.cache_data(ttl=600)
def get_tableau():
    site_views_df_all = querying.get_views_dataframe(conn)
    content_url_all = 'DataChurn/sheets/RFMPC'
    relevant_views_df_all = site_views_df_all[site_views_df_all['contentUrl'] == content_url_all]

    # ambil id workbook
    view_id_all = relevant_views_df_all['id'].to_list()[0]


    # query data and read as dataframe
    df_raw_all = querying.get_view_data_dataframe(conn, view_id=view_id_all)
    # data manipulation
    # change into date data type
    df_raw_all['Expired Date'] = pd.to_datetime(df_raw_all['Expired Date'])
    df_raw_all['Expired Date'] = df_raw_all['Expired Date'].dt.normalize()
    # change into str data type
    df_raw_all['ID Outlet'] = df_raw_all['ID Outlet'].astype('str')
    # filter only today above and below 90 days
    df_raw_all = df_raw_all.loc[(df_raw_all['Expired Date'] >= datetime.datetime.today()) & (df_raw_all['Expired Date'] <= datetime.datetime.today() + datetime.timedelta(90))].copy()
    # delete Churn column
    df_raw_all.drop(columns=['Churn'], inplace=True)


    return df_raw_all

tableau_df = get_tableau()
df_new = tableau_df.loc[~tableau_df['ID Outlet'].isin(dataframe['ID Outlet'])].copy()
conn.sign_out()


# new instance of XGBoost model (classifier)
model = XGBClassifier()
# load the model
model.load_model('classifier.json')

# make prediction (new column named Churn)
df_new['Churn'] = model.predict(df_new.loc[:, ['Recency', 'Frequency', 'Monetary', 'Age']])
# probability
df_new['Probability'] = model.predict_proba(df_new.loc[:, ['Recency', 'Frequency', 'Monetary', 'Age']])[:,1]
df_new['Probability'] = df_new['Probability'].apply(lambda x: "{0:.2f}%".format(x*100))

# reorder the columns to meet google sheets format
df_new = df_new.loc[:, ['ID Outlet', 'Recency',	'Frequency', 'Monetary', 'Age', 'Churn', 'Probability', 'Expired Date', 'Total GMV', 'Total Trx']].copy()

# add new columns
df_new['Outlet Name'] = ''
df_new['PIC'] = ''
df_new['Status'] = ''
df_new['Last Update'] = ''
df_new['PIC'] = ''
df_new['Note'] = ''

# only show the churn (=1)
df_new = df_new.loc[df_new["Churn"] == 1].copy()


##### Graphic --> Line Chart
# dataframe
df_line = df_new.groupby(pd.Grouper(key='Expired Date', freq='M'))['ID Outlet'].count().to_frame().reset_index()
df_line.columns = ['month', 'num of churns']
line_chart = px.line(df_line, x='month', y='num of churns', markers=True, text='num of churns')
line_chart.update_traces(textposition="bottom right")
st.plotly_chart(line_chart, use_container_width=True)

# title
st.markdown('__From today to next 90 days__')

############## AG GRID ###################

# update and return mode
return_mode_value = DataReturnMode.__members__["FILTERED"]
update_mode_value = GridUpdateMode.__members__["GRID_CHANGED"]

# ag grid setup
gb = GridOptionsBuilder.from_dataframe(df_new)
gb.configure_selection(selection_mode='multiple', use_checkbox=True, groupSelectsChildren=True, groupSelectsFiltered=True)   
gb.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=20)
gridOptions = gb.build()

# the ag Grid dataframe
grid_response = AgGrid(
    df_new, 
    gridOptions=gridOptions,
    data_return_mode=return_mode_value, 
    update_mode=update_mode_value,
    theme='streamlit')


# title
st.markdown('# Selected Data')

# selected rows
selected = pd.DataFrame(grid_response['selected_rows'])
st.dataframe(selected)

if len(selected) > 0:
    uploaded_values = selected.copy()
    uploaded_values['Expired Date'] = pd.to_datetime(uploaded_values['Expired Date'])
    uploaded_values['Expired Date'] = uploaded_values['Expired Date'].dt.strftime("%Y-%m-%d")
    uploaded_values = uploaded_values.iloc[:,1:]

    # insert into google sheets
    append = st.button("Upload to Sheets")

    # upload function
    def upload_sheets():
        # Create a connection object.
        gc = gspread.authorize(credentials)
        # open the google sheets file
        read = gc.open("Churn Processing 2023")
    
        read.values_append('Sheet1', {'valueInputOption': 'RAW'}, {'values': uploaded_values.values.tolist()})

    # run upload
    if append:
        upload_sheets()