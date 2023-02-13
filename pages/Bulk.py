import io
import streamlit as st
import pandas as pd
import datetime
from xgboost import XGBClassifier
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
import plotly.express as px


# set page to wide by default
st.set_page_config(page_title="Bulk Predictor", page_icon="📱")
st.markdown("# Bulk Predictor")
st.warning("Please make sure to format the files as following file (the column names and order)")
st.write("Check out this [link](https://docs.google.com/spreadsheets/d/1HB_Ujotydcr5Zda-HoFN7RLl_icoBFB0KHcjwxyAkec/edit?usp=sharing)")

upload_file =  st.file_uploader("Upload your Excel/CSV file", help="Please upload your file at first to prevent errors.")

if upload_file is not None:
    file = pd.ExcelFile(upload_file)
    df = pd.concat([pd.read_excel(file, sheet_name=sheet) for sheet in file.sheet_names], axis=0)

    # convert data types
    df["Age"] = df["Age"].astype("int")
    df["Frequency"] = df["Frequency"].astype("int")
    df["Monetary"] = df["Monetary"].astype("int")
    df["Recency"] = df["Recency"].astype("int")

    # new instance of XGBoost model (classifier)
    model = XGBClassifier()
    # load the model
    model.load_model('classifier.json')

    # split dataframe
    df_new_train = df.drop(['ID Outlet'], axis=1)

    # make prediction (new column named Churn)
    df['Churn'] = model.predict(df_new_train)
    # make probability (new column named Potensi Churn)
    df['Potensi Churn'] = model.predict_proba(df_new_train)[:,1]
    # new Potensi Churn column
    df['Potensi Churn (%)'] = df['Potensi Churn'].apply(lambda x: "{0:.2f}%".format(x*100))


    #### Pie Chart
    df_grouped = df.groupby('Churn')['ID Outlet'].count().reset_index()
    fig = px.pie(df_grouped, values='ID Outlet', names='Churn', width=500, height=500, title='Churn or Not from Uploaded Data', 
                            color_discrete_sequence=px.colors.qualitative.Pastel2)
    fig.update_traces(textinfo="label+percent", textfont_size=16)
    st.plotly_chart(fig, use_container_width=True)

    ##### ALL DATA
     # title
    st.markdown('# All Data')
    st.markdown("### The result of all prediction")
    st.dataframe(df)

    # downloadable dataframe button
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # Write excel with single worksheet
        df.to_excel(writer, index=False)
        # Close the Pandas Excel writer and output the Excel file to the buffer
        writer.save()

        # assign file to download button
        st.download_button(
            label="Download All Data in Excel",
            data=buffer,
            file_name=f"churn_bulk_prediction_{datetime.datetime.now().strftime('%Y-%m-%d')}.xlsx",
            mime="application/vnd.ms-excel",
            key="0"
        )


    ##### SELECTED DATA
    # title
    st.markdown('# Selected Data')
    st.markdown("_Filter, Sort, and/or Check to select_")
    # show dataframe using AG GRid
    st.markdown("### The result of selected prediction")
    # selected rows
    # streamlit ag grid setup
    # update and return mode
    return_mode_value = DataReturnMode.__members__["FILTERED"]
    update_mode_value = GridUpdateMode.__members__["GRID_CHANGED"]

    # grid options of AgGrid
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_selection(selection_mode='multiple', use_checkbox=True, groupSelectsChildren=True, groupSelectsFiltered=True)   
    gb.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=20)
    gridOptions = gb.build()    

    grid_response = AgGrid(
        df, 
        gridOptions=gridOptions,
        data_return_mode=return_mode_value, 
        update_mode=update_mode_value,
        theme='streamlit')

    
    selected = pd.DataFrame(grid_response['selected_rows'])
    st.dataframe(selected)

   

    # downloadable dataframe button
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # Write excel with single worksheet
        selected.to_excel(writer, index=False)
        # Close the Pandas Excel writer and output the Excel file to the buffer
        writer.save()

        # assign file to download button
        st.download_button(
            label="Download Selected Data in Excel",
            data=buffer,
            file_name=f"churn_bulk_prediction_{datetime.datetime.now().strftime('%Y-%m-%d')}.xlsx",
            mime="application/vnd.ms-excel",
            key="1"
        )