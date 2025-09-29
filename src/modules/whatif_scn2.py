import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from datetime import datetime,timedelta
from sklearn.preprocessing import StandardScaler

def load_and_prepare_data(file_path):
    df = pd.read_excel(file_path, sheet_name="Sheet1")
    columns_to_convert = ["Q2", "ABN %", "Loaded AHT", "Met", "Missed"]
    for col in columns_to_convert:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.fillna(0)
    return df


def train_regression_model(df):
    features = ["Occ Assumption", "Staffing", "Demand", "Occupancy Rate"]
    target = "Requirement"
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # print(f"Model Trained: MAE = {mae:.2f}, R2 Score = {r2:.4f}")
    # joblib.dump(model, 'model_whatifs')
    # coefficients = dict(zip(features, model.coef_))
    # intercept = model.intercept_
    # equation = " + ".join([f"{coef:.2f}*{feat}" for feat, coef in coefficients.items()])
    # print(f"Regression Equation: Requirement = {equation} + {intercept:.2f}")
    return model

def analyze_scenarios(df, model, occ,occ_assmp):
    test_data = pd.DataFrame({
        "Occ Assumption": [occ_assmp],
        "Staffing": [df["Staffing"].mean()],
        "Demand": [df["Demand"].mean()],
        "Occupancy Rate": [occ]  
    })
    fte_prediction = model.predict(test_data)[0]
    return int(fte_prediction)

def fte_impact_of_demand_change(df, model, occ, occ_assmp, weighted_avg_staff):
    demand_changes = [-25,-20,-15,-10, -5, -2, -1, 0, 1, 2, 5, 10, 15, 20, 25]
    predictions = []
    for change in demand_changes:
        demand_scenario = df["Demand"].mean() * (1 + change / 100)
        fte_pred = model.predict(pd.DataFrame({
            "Occ Assumption": [occ_assmp],
            "Staffing": [weighted_avg_staff], "Demand": [demand_scenario], "Occupancy Rate": [occ]
        }))[0]
        predictions.append((change, demand_scenario, fte_pred))
    return predictions

def impact_of_occ_assumption_change(df, model, occ, occ_assmp, weighted_avg_demand, weighted_avg_staff):
    occ_changes = [-10, -5, -2, -1, 0, 1, 2, 5, 10]
    predictions = []
    for change in occ_changes:
        occ_scenario = occ_assmp * (1 + change / 100)
        fte_pred = model.predict(pd.DataFrame({
            "Occ Assumption": [occ_scenario],
            "Staffing": [weighted_avg_staff], "Demand": [weighted_avg_demand], "Occupancy Rate": [occ],
            "Occ Assumption": [occ_scenario]
        }))[0]
        predictions.append((change, occ_scenario, fte_pred))
    return predictions

st.title("What-If Analysis for FTE Requirements")
st.sidebar.header("User Inputs")

def scn2():

    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.write("Excel file uploaded successfully!")
        df_date = df[(df['USD'] == 'Combined') & (df['Level'] == 'Combined')].copy()
        df_date['startDate'] = pd.to_datetime(df_date['startDate per day'])
        max_date = df_date['startDate'].max()
        formatted_max_date = max_date.strftime('%Y-%m-%d')

        # Find the Sunday of that week
        sunday_date = max_date - timedelta(days=max_date.weekday() + 1) if max_date.weekday() != 6 else max_date
        formatted_sunday = sunday_date.strftime('%Y-%m-%d')

        date_str = st.text_input("Enter the date to search (YYYY-MM-DD):", value=formatted_sunday)
        search_date = datetime.strptime(date_str, "%Y-%m-%d")
        st.write(f"Selected Date: {search_date.date()}")  

        usd_options = df['USD'].unique()
        level_options = df['Level'].unique()
        default_usd_global_index = list(usd_options).index("Combined") if "Combined" in usd_options else 0
        default_level_index = list(level_options).index("Combined") if "Combined" in level_options else 0

        language = st.selectbox("Select Language", df['Language'].unique())
        req_media = st.selectbox("Select Req Media", df['Req Media'].unique())
        usd = st.selectbox("Select USD", df['USD'].unique(), index=default_usd_global_index)
        level = st.selectbox("Select Level", df['Level'].unique(), index=default_level_index)

        filtered_df = df[(df['Language'] == language) &
                         (df['Req Media'] == req_media) &
                         (df['USD'] == usd) &
                         (df['Level'] == level)].copy()

        # Ensure date column is datetime
        filtered_df['Date'] = pd.to_datetime(filtered_df['startDate per day'], errors='coerce')

        filtered_df["Staffing"] = filtered_df["Staffing"]*1.4

        # Get week range
        start_of_week = search_date
        end_of_week = search_date + timedelta(days=6)
        weekly_df = filtered_df[(filtered_df['Date'] >= start_of_week) & (filtered_df['Date'] <= end_of_week)]

        # Calculate weekly averages      

        # avg_q2 = weekly_df['Q2'].mean() if 'Q2' in weekly_df else 20
        # avg_abn = round(weekly_df['ABN %'].mean(),2) if 'ABN %' in weekly_df else 2.00
        avg_occ = weekly_df['Occupancy Rate'].mean()*100 if 'Occupancy Rate' in weekly_df else 70
        avg_occ_assmp = weekly_df['Occ Assumption'].mean()*100 if 'Occ Assumption' in weekly_df else 70

        # Sidebar sliders with dynamic defaults
        # q2 = st.sidebar.slider("Set Q2 Time", min_value=0, max_value=100, value=int(avg_q2), step=1)
        # abn = st.sidebar.slider("Set Abandon Rate (%)", min_value=0.0, max_value=20.0, value=float(avg_abn), step=0.01)
        occ = st.sidebar.slider("Set Occupancy Rate (%)", min_value=0, max_value=100, value=int(avg_occ), step=1)
        occ_assmp = st.sidebar.slider("Set Occ Assumption (%)", min_value=0, max_value=100, value=int(avg_occ_assmp), step=1)
            
        # Rename columns safely
        filtered_df.rename(columns={ 
            'Met': 'Service Level',
            'Loaded AHT': 'AHT'
        }, inplace=True)

        # Select and copy relevant columns
        filtered_df1 = filtered_df[['Date', 'Service Level', 'Q2', 'AHT', 'Demand',
                            'Occ Assumption', 'Requirement', 'Staffing', 'Occupancy Rate',
                            'Staffing Diff', 'ABN %', 'Calls']].copy()
        
        
        # Drop rows with any missing values
        filtered_df1 = filtered_df1.dropna()

        # Convert 'Date' column to datetime and sort
        filtered_df1['Date'] = pd.to_datetime(filtered_df1['Date'], errors='coerce')
        filtered_df1 = filtered_df1.sort_values(by='Date')


        for col in ['Occ Assumption', 'Q2', 'Service Level', 'ABN %']:
            filtered_df1[col] = pd.to_numeric(filtered_df1[col], errors='coerce')
     

        st.header("Output------------------------------>")          
        model = train_regression_model(filtered_df1)

        fte_prediction = analyze_scenarios(weekly_df, model, occ/100, occ_assmp/100)
        st.write(f"Predicted FTE Requirement based on user inputs: {fte_prediction}")
    
        st.header("FTE Impact on Demand Change")
        weighted_avg_demand = weekly_df['Demand'].mean()
        weighted_avg_staff = (weekly_df['Staffing'] * weekly_df['Calls']).sum() / weekly_df['Calls'].sum()
        st.write(f"**Average Demand Weekly = {weighted_avg_demand * 7:.2f} | Average Staffing = {weighted_avg_staff:.2f}**")


        demand_impact = fte_impact_of_demand_change(weekly_df, model, occ/100, occ_assmp/100, weighted_avg_staff)
        for change, demand_scenario, fte_pred in demand_impact:
            st.write(f"Demand Change = {change}% | New Weekly Demand = {demand_scenario * 7:.2f}: | Predicted FTE Requirement= {fte_pred:.2f}")

        st.header("FTE Impact on OCC Assumption Change")
        weighted_avg_demand = weekly_df['Demand'].mean()
        weighted_avg_staff = (weekly_df['Staffing'] * weekly_df['Calls']).sum() / weekly_df['Calls'].sum()
        st.write(f"**Average Demand Weekly = {weighted_avg_demand * 7:.2f} | Average Staffing = {weighted_avg_staff:.2f}**")

        occ_impact = impact_of_occ_assumption_change(weekly_df, model, occ/100,occ_assmp/100, weighted_avg_demand, weighted_avg_staff)
        for change, occ_scenario, fte_pred in occ_impact:
            st.write(f"Occ Assump Change {change}% | New Occupancy Assumption = {occ_scenario * 100:.1f}% | Predicted FTE Requirement = {fte_pred:.2f}")

if __name__ == '__main__':
    scn2()
