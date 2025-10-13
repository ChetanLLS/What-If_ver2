import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import time
from datetime import datetime, timedelta

st.title('Scenario Input and Excel Upload')

def scn3():
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

        # Get week range
        start_of_week = search_date
        end_of_week = search_date + timedelta(days=6)
        weekly_df = filtered_df[(filtered_df['Date'] >= start_of_week) & (filtered_df['Date'] <= end_of_week)]

        # Calculate weekly averages      

        avg_q2 = weekly_df['Q2'].mean() if 'Q2' in weekly_df else 20
        # avg_abn = round(weekly_df['ABN %'].mean(),2) if 'ABN %' in weekly_df else 2.00
        avg_occ = weekly_df['Occupancy Rate'].mean()*100 if 'Occupancy Rate' in weekly_df else 70
        avg_occ_assmp = weekly_df['Occ Assumption'].mean()*100 if 'Occ Assumption' in weekly_df else 70

        # Sidebar sliders with dynamic defaults
        q2 = st.sidebar.slider("Set Q2 Time", min_value=0, max_value=100, value=int(avg_q2), step=1)
        # abn = st.sidebar.slider("Set Abandon Rate (%)", min_value=0.00, max_value=10.00, value=float(avg_abn), step=0.01)
        occ = st.sidebar.slider("Set Occupancy Rate (%)", min_value=0, max_value=100, value=int(avg_occ), step=1)
        occ_assmp = st.sidebar.slider("Set Occ Assumption (%)", min_value=0, max_value=100, value=int(avg_occ_assmp), step=1)

        try:
            start_time = time.time()
            
            # Rename columns safely
            filtered_df.rename(columns={ 
                'Met': 'Service Level',
                'Loaded AHT': 'AHT'
            }, inplace=True)

            # Select and copy relevant columns
            filtered_df1 = filtered_df[['Date', 'Service Level', 'Q2', 'AHT', 'Demand',
                            'Occ Assumption', 'Requirement', 'Staffing', 'Occupancy Rate',
                            'Staffing Diff', 'ABN %']].copy()

            # Drop rows with any missing values
            filtered_df1 = filtered_df1.dropna()

            # Convert 'Date' column to datetime and sort
            filtered_df1['Date'] = pd.to_datetime(filtered_df1['Date'], errors='coerce')
            filtered_df1 = filtered_df1.sort_values(by='Date')


            for col in ['Occ Assumption', 'Q2', 'Service Level', 'ABN %']:
                filtered_df1[col] = pd.to_numeric(filtered_df1[col], errors='coerce')

            split_index = int(len(filtered_df1) * 0.7)
            split_date = filtered_df1.iloc[split_index]['Date']
            train_df = filtered_df1[filtered_df1['Date'] <= split_date]
            test_df = filtered_df1[filtered_df1['Date'] > split_date]

            X_train = train_df[['Demand', 'Q2', 'Occupancy Rate', 'Occ Assumption']]
            X_test = test_df[['Demand', 'Q2', 'Occupancy Rate', 'Occ Assumption']]
            y_train = train_df['Requirement']
            y_test = test_df['Requirement']

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model = LinearRegression()
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            r2_train = r2_score(y_train, y_train_pred)
            r2_test = r2_score(y_test, y_test_pred)
            mse_train = mean_squared_error(y_train, y_train_pred)
            mse_test = mean_squared_error(y_test, y_test_pred)

            # training_time = time.time() - start_time
            # st.success(f"Model trained. R² Train: {r2_train:.4f}, R² Test: {r2_test:.4f}, Time: {training_time:.2f}s")

            joblib.dump(model, 'linear_regression_model.pkl')
            joblib.dump(scaler, 'scaler.pkl')

            log_df = pd.DataFrame({
                'Timestamp': [datetime.now()],
                'R2_Train': [r2_train],
                'R2_Test': [r2_test],
                'MSE_Train': [mse_train],
                'MSE_Test': [mse_test]
            })

            # try:
            #     existing_log = pd.read_excel('model_accuracy_log.xlsx')
            #     log_df = pd.concat([existing_log, log_df], ignore_index=True)
            # except FileNotFoundError:
            #     pass

            # log_df.to_excel('model_accuracy_log.xlsx', index=False)
            # st.write("Model accuracy logged successfully.")

            def predict_fte(demand_change_percent, scenario_type, date_str, df, q2, occ, occ_assmp):
                try:

                    if scenario_type == 'weekly':
                        week_year = search_date.strftime("%Y-%U")
                        weekly_df = df[df['Date'].dt.strftime("%Y-%U") == week_year]
                        average_demand = weekly_df['Demand'].mean()

                        start_date = search_date
                        end_date = search_date + timedelta(days=6)
                        st.write(f"Week Range: {start_date.date()} to {end_date.date()}")

                        predictions = {}
                        actual_fte_values = []
                        predicted_fte_values = []
                        daily_demand_value = []
                        st.write('**Daywise Predictions FTE**')
                        for date in pd.date_range(start=start_date, end=end_date):
                            date_str_day = date.strftime("%Y-%m-%d")
                            daily_demand = df[df['Date'].dt.strftime("%Y-%m-%d") == date_str_day]['Demand'].mean()
                            actual_fte = df[df['Date'].dt.strftime("%Y-%m-%d") == date_str_day]['Requirement'].mean()

                            if np.isnan(daily_demand):
                                daily_demand = 0

                            new_daily_demand = daily_demand * (1 + demand_change_percent / 100)

                            input_features = np.array([[new_daily_demand, q2, occ,occ_assmp]])
                            input_scaled = scaler.transform(input_features)
                            predicted_fte = model.predict(input_scaled)[0]
                            st.write(f"{date_str_day} | Original FTE Requirement: {actual_fte:.2f} |  New Predicted FTE Requirement: {predicted_fte:.2f}")

                            daily_demand_value.append(daily_demand)
                            predictions[date_str_day] = predicted_fte
                            actual_fte_values.append(actual_fte)
                            predicted_fte_values.append(predicted_fte)

                        average_demand = np.mean(daily_demand_value)
                        average_actual_fte = np.mean(actual_fte_values)
                        # st.write(np.average(actual_fte_values, weights=daily_demand_value))
                        # st.write(np.average(predicted_fte_values, weights=daily_demand_value))

                        average_predicted_fte = np.mean(predicted_fte_values)
                        fte_percentage_change = ((average_predicted_fte - average_actual_fte) / average_actual_fte) * 100

                        return predictions, scenario_type, average_demand, average_actual_fte, average_predicted_fte, fte_percentage_change

                    else:
                        st.error("Invalid scenario type. Only 'Weekly' is supported.")
                        return None

                except ValueError:
                    st.error("Invalid date format. Please enter the date in YYYY-MM-DD format.")
                    return None

            # UI Inputs
            demand_change_percent = st.number_input("Enter demand change (%) [Increase +ve| Decrease -ve]:", min_value=-100, max_value=100, value=0, step=1)
            scenario_type = st.selectbox("Select the scenario type:", ["Weekly"]).strip().lower()

            if st.button('Predict FTE'):
                result = predict_fte(demand_change_percent, scenario_type, date_str, filtered_df1, q2, occ/100, occ_assmp/100)
                if result:
                    predictions, scenario_type, average_demand, average_actual_fte, average_predicted_fte, fte_percentage_change = result

                    new_demand = average_demand * (1 + demand_change_percent / 100)
                    actual_demand = average_demand * 7
                    new_demand_scn_type = new_demand * 7

                    st.write('**Weekwise Predictions FTE**')
                    st.write(f"Scenario Type: {scenario_type}")
                    st.write(f"Total Demand - {scenario_type}: {int(actual_demand)}")
                    st.write(f"Demand Change %: {demand_change_percent:.2f}")
                    st.write(f"New Total Demand - {scenario_type}: {int(new_demand_scn_type)}")
                    st.write(f"Original FTEs Requirement: {average_actual_fte:.2f}")
                    st.write(f"New FTEs Requirement: {average_predicted_fte:.2f}")
                    st.write(f"FTE Requirement % Change: {fte_percentage_change:.2f}")

        except Exception as e:
            st.error(f"Error processing data: {e}")

if __name__ == "__main__":
    scn3()
