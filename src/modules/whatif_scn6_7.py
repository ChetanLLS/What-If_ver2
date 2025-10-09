import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# Load and prepare data
def load_and_prepare_data(file_path):
    df = pd.read_excel(file_path, sheet_name="Sheet1")
    columns_to_convert = ["Q2", "ABN %", "Loaded AHT", "Met", "Missed"]
    for col in columns_to_convert:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Requirement", "Staffing", "Demand"])
    df = df.fillna(0)
    return df

# Train regression model
def train_regression_model(df):
    features = ["Q2", "ABN %", "Occ Assumption", "Staffing", "Occupancy Rate"]
    target = "Requirement"
    X = df[features]
    y = df[target]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    # # Get coefficients and intercept
    # coefficients = model.coef_
    # intercept = model.intercept_

    # # Display results
    # print("Intercept:", intercept)
    # print("Coefficients:")
    
    # feature_names = X_train.columns
    # for name, coef in zip(feature_names, coefficients):
    #     print(f"{name}: {coef}")    
    return model

# KPI impact scenario
def kpi_impact_scenario(df, model, percentage_changes, q2, abn, occ, occ_assmp):
    results = []
    for change in percentage_changes:
        adjusted_calls = df["Calls"].median() * (1 + change / 100)
        adjusted_fte = df["Staffing"].median()
        calls_per_fte = adjusted_calls / adjusted_fte if adjusted_fte != 0 else 0

        test_scenario = pd.DataFrame({
            "Q2": [q2 * (1 + change / 100)],
            "ABN %": [abn * (1 + change / 100)],
            "Occ Assumption": [occ_assmp / 100 * (1 + change / 100)],
            "Staffing": [adjusted_fte],
            "Occupancy Rate": [occ / 100 * (1 + change / 100)]
        })

        fte_pred = model.predict(test_scenario)[0]

        results.append({
            "Change %": change,
            "Calls Per FTE": round(calls_per_fte, 2),
            "Predicted FTE Requirement": round(fte_pred, 2),
            "Adjusted Q2": round(q2 * (1 + change / 100), 2),
            "Adjusted ABN %": round(abn * (1 + change / 100), 2),
            "Adjusted Occupancy Rate %": round(occ * (1 + change / 100), 2),
            "Adjusted Occ Assumption %": round(occ_assmp * (1 + change / 100), 2)
        })
    return pd.DataFrame(results)

# Analyze single scenario
def analyze_scenarios(df, model, q2, abn, occ, occ_assmp):
    test_data = pd.DataFrame({
        "Q2": [q2],
        "ABN %": [abn],
        "Occ Assumption": [occ_assmp / 100],
        "Staffing": [df["Staffing"].median()],
        "Occupancy Rate": [occ / 100]
    })
    return model.predict(test_data)[0]

# Main Streamlit app
def scn6_7():
    st.title("Impact of Calls per FTE on KPI Metrics")
    st.sidebar.header("User Inputs")

    file_path = st.file_uploader("Upload your Excel file", type=["xlsx"])
    if file_path is not None:
        df = pd.read_excel(file_path)
        st.success("Excel file uploaded successfully!")

        # staffing adjustment
        df['Staffing'] = df['Staffing'] *1.4  

        df_date = df[(df['USD'] == 'Combined') & (df['Level'] == 'Combined')].copy()
        df_date['startDate'] = pd.to_datetime(df_date['startDate per day'])
        max_date = df_date['startDate'].max()

        # sunday_date = max_date - timedelta(days=max_date.weekday() + 1) if max_date.weekday() != 6 else max_date
        sunday_date = max_date - timedelta(days=7 if max_date.weekday() == 6 else max_date.weekday() + 1)
        date_str = st.text_input("Enter the date to search (YYYY-MM-DD):", value=sunday_date.strftime('%Y-%m-%d'))
        search_date = datetime.strptime(date_str, "%Y-%m-%d")
        st.write(f"Selected Date: {search_date.date()}")

        df = load_and_prepare_data(file_path)

        language = st.selectbox("Select Language", df['Language'].unique())
        req_media = st.selectbox("Select Req Media", df['Req Media'].unique())
        usd = st.selectbox("Select USD", df['USD'].unique(), index=df['USD'].unique().tolist().index("Combined"))
        level = st.selectbox("Select Level", df['Level'].unique(), index=df['Level'].unique().tolist().index("Combined"))

        filtered_df = df[(df['Language'] == language) &
                         (df['Req Media'] == req_media) &
                         (df['USD'] == usd) &
                         (df['Level'] == level)].copy()

        filtered_df['Date'] = pd.to_datetime(filtered_df['startDate per day'], errors='coerce')
        start_of_week = search_date
        end_of_week = search_date + timedelta(days=6)
        weekly_df = filtered_df[(filtered_df['Date'] >= start_of_week) & (filtered_df['Date'] <= end_of_week)]

        avg_q2 = weekly_df['Q2'].mean() if 'Q2' in weekly_df else 20
        avg_abn = weekly_df['ABN %'].mean() if 'ABN %' in weekly_df else 2
        avg_occ = weekly_df['Occupancy Rate'].mean() * 100 if 'Occupancy Rate' in weekly_df else 70
        avg_occ_assmp = weekly_df['Occ Assumption'].mean() * 100 if 'Occ Assumption' in weekly_df else 70

        q2 = st.sidebar.slider("Set Q2 Time", 0, 100, int(avg_q2), 1)
        abn = st.sidebar.slider("Set Abandon Rate (%)", 0.0, 10.0, float(avg_abn), 0.01)
        occ = st.sidebar.slider("Set Occupancy Rate (%)", 0, 100, int(avg_occ), 1)
        occ_assmp = st.sidebar.slider("Set Occ Assumption (%)", 0, 100, int(avg_occ_assmp), 1)

        st.header("Predicted FTE Requirement")
        model = train_regression_model(filtered_df)
        fte_prediction = analyze_scenarios(weekly_df, model, q2, abn, occ, occ_assmp)
        st.write(f"Predicted FTE based on inputs: **{fte_prediction:.2f}**")

        st.header("KPI Impact from Calls per FTE Changes")
        percentage_changes = np.array([-25, -20, -15, -10, -5, -2, -1, 0, 1, 2, 5, 10, 15, 20, 25])
        kpi_impact_df = kpi_impact_scenario(weekly_df, model, percentage_changes, q2, abn, occ, occ_assmp)
        st.dataframe(kpi_impact_df.round(2))

if __name__ == '__main__':
    scn6_7()
