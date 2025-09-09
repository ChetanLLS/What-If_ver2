# from modules.utils import st,pd,np
# from datetime import datetime, timedelta
import calendar
# from modules.utils import StandardScaler
# from modules.utils import LinearRegression
# from modules.utils import r2_score, mean_squared_error
# from modules.utils import joblib
# import time
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
import joblib
import time

from datetime import datetime,timedelta

# Streamlit application
st.title('Scenario Input and Excel Upload')

def format_number_to_million(number):
    # Convert the number to millions
    number_in_million = number / 1_000_000
    # Format the number with commas and 4 decimal places
    formatted_number = "{:,.4f}".format(number_in_million)
    print(formatted_number)
    return formatted_number
def scn3():
    # File uploader for Excel file
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

    if uploaded_file is not None:
        # Read the Excel file
        df = pd.read_excel(uploaded_file)
        st.write("Excel file uploaded successfully!")

        # Dropdown filters based on data columns
        language_filter = st.selectbox("Select Language", options=df['Language'].unique())
        req_media_filter = st.selectbox("Select Req Media", options=df['Req Media'].unique())
        usd_filter = st.selectbox("Select USD", options=df['USD'].unique())
        level_filter = st.selectbox("Select Level", options=df['Level'].unique())

        # Data filtering based on selected filters
        filtered_df = df[(df['Language'] == language_filter) & 
                        (df['Req Media'] == req_media_filter) & 
                        (df['USD'] == usd_filter) & 
                        (df['Level'] == level_filter)]

        # Data preparation and model training
        try:
            start_time = time.time()  # Start timing

            # Data processing
            filtered_df['Abandon Demand'] = filtered_df['ABN %'] * filtered_df['Demand']
            filtered_df.rename(columns={'startDate per day': 'Date', 'Met': 'Service Level',
                                        'Loaded AHT': 'AHT'}, inplace=True)
            filtered_df1 = filtered_df[['Date', 'Service Level', 'Q2', 'AHT', 'Abandon Demand', 'Demand',
                                        'Occ Assumption', 'Requirement', 'Staffing', 'Occupancy Rate', 'Staffing Diff', 'ABN %']].copy()
            filtered_df1.dropna(inplace=True)
            filtered_df1['Occ Assumption'] = pd.to_numeric(filtered_df1['Occ Assumption'], errors='coerce')
            filtered_df1['Q2'] = pd.to_numeric(filtered_df1['Q2'], errors='coerce')
            filtered_df1['Abandon Demand'] = pd.to_numeric(filtered_df1['Abandon Demand'], errors='coerce')
            filtered_df1['Service Level'] = pd.to_numeric(filtered_df1['Service Level'], errors='coerce')
            filtered_df1['ABN %'] = pd.to_numeric(filtered_df1['ABN %'], errors='coerce')
            filtered_df1['Date'] = pd.to_datetime(filtered_df1['Date'])
            filtered_df1 = filtered_df1.sort_values(by='Date')

            # Data split
            split_index = int(len(filtered_df1) * 0.7)
            split_date = filtered_df1.iloc[split_index]['Date']
            train_df = filtered_df1[filtered_df1['Date'] <= split_date]
            test_df = filtered_df1[filtered_df1['Date'] > split_date]
            X_train = train_df[['Demand']]
            X_test = test_df[['Demand']]
            y_train = train_df['Requirement']
            y_test = test_df['Requirement']

            # Scale data
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Evaluate model
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            r2_train = r2_score(y_train, y_train_pred)
            r2_test = r2_score(y_test, y_test_pred)
            mse_train = mean_squared_error(y_train, y_train_pred)
            mse_test = mean_squared_error(y_test, y_test_pred)

            end_time = time.time()  # End timing
            training_time = end_time - start_time  # Calculate training time
            st.success(f"Model and scaler saved successfully. Model R^2 score on training data: {r2_train:.4f}, testing data: {r2_test:.4f}. Training time: {training_time:.2f} seconds")

            # Save the model and scaler
            model_file_path = 'linear_regression_model.pkl'
            scaler_file_path = 'scaler.pkl'
            joblib.dump(model, model_file_path)
            joblib.dump(scaler, scaler_file_path)

            # Log model accuracy
            def log_model_accuracy(r2_train, r2_test, mse_train, mse_test):
                log_data = {
                    'Timestamp': [datetime.now()],
                    'R2_Train': [r2_train],
                    'R2_Test': [r2_test],
                    'MSE_Train': [mse_train],
                    'MSE_Test': [mse_test]
                }
                log_df = pd.DataFrame(log_data)
                try:
                    existing_log = pd.read_excel('model_accuracy_log.xlsx')
                    log_df = pd.concat([existing_log, log_df], ignore_index=True)
                except FileNotFoundError:
                    pass
                log_df.to_excel('model_accuracy_log.xlsx', index=False)

            log_model_accuracy(r2_train, r2_test, mse_train, mse_test)
            st.write("Model accuracy logged successfully.")

            # Function to predict FTE based on demand increase percentage
            def predict_fte(demand_increase_percent, scenario_type, date_str, df):

                try:
                    search_date = datetime.strptime(date_str, "%Y-%m-%d")

                    if scenario_type == 'weekly':
                        week_year = search_date.strftime("%Y-%U")
                        weekly_df = df[df['Date'].dt.strftime("%Y-%U") == week_year]
                        average_demand = weekly_df['Demand'].mean()

                        start_date = search_date - timedelta(days=search_date.weekday())
                        end_date = start_date + timedelta(days=6)

                        predictions = {}
                        actual_fte_values = []
                        predicted_fte_values = []

                        for date in pd.date_range(start=start_date, end=end_date):
                            date = date.strftime("%Y-%m-%d")
                            daily_demand = df[df['Date'].dt.strftime("%Y-%m-%d") == date]['Demand'].mean()
                            actual_fte = df[df['Date'] == date]['Requirement'].mean()

                            if np.isnan(daily_demand):
                                daily_demand = 0

                            new_daily_demand = daily_demand * (1 + demand_increase_percent / 100)
                            new_daily_demand_std = scaler.transform([[new_daily_demand]])
                            predicted_fte = model.predict(new_daily_demand_std)[0]

                            predictions[date] = predicted_fte
                            actual_fte_values.append(actual_fte)
                            predicted_fte_values.append(predicted_fte)

                        average_actual_fte = np.mean(actual_fte_values)
                        average_predicted_fte = np.mean(predicted_fte_values)
                        fte_percentage_change = ((average_predicted_fte - average_actual_fte) / average_actual_fte) * 100

                    elif scenario_type == 'monthly':
                        month_year = search_date.strftime("%Y-%m")
                        monthly_df = df[df['Date'].dt.strftime("%Y-%m") == month_year]
                        average_demand = monthly_df['Demand'].mean()

                        start_date = search_date.replace(day=1)
                        end_date = search_date.replace(day=calendar.monthrange(search_date.year, search_date.month)[1])

                        predictions = {}
                        actual_fte_values = []
                        predicted_fte_values = []

                        for date in pd.date_range(start=start_date, end=end_date):
                            date = date.strftime("%Y-%m-%d")
                            daily_demand = df[df['Date'].dt.strftime("%Y-%m-%d") == date]['Demand'].mean()
                            actual_fte = df[df['Date'] == date]['Requirement'].mean()

                            if np.isnan(daily_demand):
                                daily_demand = 0

                            new_daily_demand = daily_demand * (1 + demand_increase_percent / 100)
                            new_daily_demand_std = scaler.transform([[new_daily_demand]])
                            predicted_fte = model.predict(new_daily_demand_std)[0]

                            predictions[date] = predicted_fte
                            actual_fte_values.append(actual_fte)
                            predicted_fte_values.append(predicted_fte)

                        average_actual_fte = np.mean(actual_fte_values)
                        average_predicted_fte = np.mean(predicted_fte_values)
                        fte_percentage_change = ((average_predicted_fte - average_actual_fte) / average_actual_fte) * 100

                    elif scenario_type == 'yearly':
                        year = search_date.strftime("%Y")
                        yearly_df = df[df['Date'].dt.strftime("%Y") == year]
                        average_demand = yearly_df['Demand'].mean()

                        start_date = search_date.replace(month=1, day=1)
                        end_date = search_date.replace(month=12, day=31)

                        predictions = {}
                        actual_fte_values = []
                        predicted_fte_values = []

                        for date in pd.date_range(start=start_date, end=end_date):
                            date = date.strftime("%Y-%m-%d")
                            daily_demand = df[df['Date'].dt.strftime("%Y-%m-%d") == date]['Demand'].mean()
                            actual_fte = df[df['Date'] == date]['Requirement'].mean()

                            if np.isnan(daily_demand):
                                daily_demand = 0

                            new_daily_demand = daily_demand * (1 + demand_increase_percent / 100)
                            new_daily_demand_std = scaler.transform([[new_daily_demand]])
                            predicted_fte = model.predict(new_daily_demand_std)[0]

                            predictions[date] = predicted_fte
                            actual_fte_values.append(actual_fte)
                            predicted_fte_values.append(predicted_fte)

                        average_actual_fte = np.mean(actual_fte_values)
                        average_predicted_fte = np.mean(predicted_fte_values)
                        fte_percentage_change = ((average_predicted_fte - average_actual_fte) / average_actual_fte) * 100

                    else:
                        st.error("Invalid scenario type. Please select 'Weekly', 'Monthly', or 'Yearly'.")
                        return None

                except ValueError:
                    st.error("Invalid date format. Please enter the date in YYYY-MM-DD format.")
                    return None

                return predictions, scenario_type, average_demand, average_actual_fte, average_predicted_fte, fte_percentage_change

            demand_increase_percent = st.number_input("Enter the percentage increase in demand (%):", min_value=0, step=1)
            scenario_type = st.selectbox("Select the scenario type:", ["Weekly", "Monthly", "Yearly"]).strip().lower()
            date_str = st.text_input("Enter the date to search (YYYY-MM-DD):")
            
            if st.button('Predict FTE'):
                result = predict_fte(demand_increase_percent, scenario_type, date_str , filtered_df1)
                if result:
                    predictions, scenario_type, average_demand, average_actual_fte, average_predicted_fte, fte_percentage_change = result

                    new_demand = average_demand * (1 + demand_increase_percent / 100)

                    st.write(f"Scenario Type: {scenario_type}")
    #                 st.write(f"Average Demand: {format_number_to_million(average_demand)}M")
                    st.write(f"Average Demand: {average_demand}")
                    st.write(f"Demand Increase %: {demand_increase_percent:.2f}")
                    st.write(f"Average Demand: {new_demand}")
    #                 st.write(f"New Demand: {format_number_to_million(new_demand)}M")
                    st.write(f"Actual FTEs Required (Avg): {average_actual_fte:.2f}")
                    st.write(f"Predicted FTEs Required (Avg): {average_predicted_fte:.2f}")
                    st.write(f"FTE % Change: {fte_percentage_change:.2f}")

                    # Save results to Excel
                    results_df = pd.DataFrame({
                        'Scenario Type': [scenario_type],
                        'Demand Increase %': [demand_increase_percent],
                        'Average Demand': [average_demand],
                        'New Demand': [new_demand],
                        'Actual FTEs Required (Avg)': [average_actual_fte],
                        'Predicted FTEs Required (Avg)': [average_predicted_fte],
                        'FTE % Change': [fte_percentage_change]
                    })
                    # results_file = 'FTE_predictions.xlsx'
                    # results_df.to_excel(results_file, index=False, engine='xlsxwriter')

                    # st.success(f"Results saved to {results_file}")
                    # with open(results_file, "rb") as f:
                    #     st.download_button(label="Download Excel", data=f, file_name=results_file, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as e:
            st.error(f"Error processing data: {e}")


               
