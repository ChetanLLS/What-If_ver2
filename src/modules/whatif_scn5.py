import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def remove_outliers_iqr(df, column):
    lower_bound = df[column].quantile(0.05)
    upper_bound = df[column].quantile(0.95)
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def load_data(uploaded_file):
    return pd.read_excel(uploaded_file)

def filter_data(df, language, req_media, usd, level):
    return df[(df['Language'] == language) &
              (df['Req Media'] == req_media) &
              (df['USD'] == usd) &
              (df['Level'] == level)]

def preprocess_data(df):
    df['Abandon Demand'] = df['ABN %'] * df['Demand']
    df.rename(columns={'startDate per day': 'Date', 'Met': 'Service Level', 'Loaded AHT': 'AHT'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])

    df['Calls_per_FTE'] = np.where((~df['Staffing'].isna()) & (df['Staffing'] != 0), df['Calls'] / df['Staffing'], 0)
    df['Demand_per_FTE'] = np.where((~df['Demand'].isna()) & (df['Staffing'] != 0), df['Demand'] / df['Staffing'], 0)

    # Bucket ABN %
    df['ABN_rate_bucket'] = pd.cut(df['ABN %'], bins=[0, 0.02, 0.04, 0.06, 0.08, float('inf')],
                                   labels=['0-2%', '2-4%', '4-6%', '6-8%', '8%+'], right=False)


    # Bucket Q2 (0 to 50+ in steps of 5)
    df['Q2_bucket'] = pd.cut(df['Q2'], 
                            bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, float('inf')],
                            labels=['0-5', '5-10', '10-15', '15-20', '20-25', '25-30',
                                     '30-35', '35-40', '40-45', '45-50', '50+'],
                            right=False)


    # One-hot encode buckets
    df = pd.get_dummies(df, columns=['ABN_rate_bucket', 'Q2_bucket'], drop_first=True)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)

    return df

def train_model(train_df, test_df, importance_threshold=0.05):
    initial_features = [col for col in train_df.columns if col != 'Staffing']

    train_df[initial_features] = train_df[initial_features].apply(pd.to_numeric, errors='coerce')
    test_df[initial_features] = test_df[initial_features].apply(pd.to_numeric, errors='coerce')

    train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    train_df.fillna(train_df.mean(numeric_only=True), inplace=True)
    test_df.fillna(test_df.mean(numeric_only=True), inplace=True)

    X_train = train_df[initial_features]
    X_test = test_df[initial_features]
    y_train = train_df['Staffing']
    y_test = test_df['Staffing']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    importances = pd.Series(model.feature_importances_, index=initial_features)
    print(importances)
    important_features = importances[importances >= importance_threshold].index.tolist()

    # Retrain with selected features
    X_train_imp = train_df[important_features]
    X_test_imp = test_df[important_features]
    X_train_imp_scaled = scaler.fit_transform(X_train_imp)
    X_test_imp_scaled = scaler.transform(X_test_imp)

    model.fit(X_train_imp_scaled, y_train)

    training_score = model.score(X_train_imp_scaled, y_train)
    testing_score = model.score(X_test_imp_scaled, y_test)

    return model, scaler, training_score, testing_score, important_features

def predict_changes(model, scaler, feature_columns, adjusted_data):
    adjusted_data = adjusted_data.reshape(1, -1)
    scaled_data = scaler.transform(adjusted_data)
    return model.predict(scaled_data)[0]

def scn_5():
    st.title("Staffing Requirement Prediction with Variable Analysis")

    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.write("Data loaded successfully!")

        df['Staffing'] = df['Staffing'] * 1.4

        usd_options = df['USD'].unique()
        level_options = df['Level'].unique()
        default_usd_global_index = list(usd_options).index("Combined") if "Combined" in usd_options else 0
        default_level_index = list(level_options).index("Combined") if "Combined" in level_options else 0

        language = st.selectbox("Select Language", df['Language'].unique())
        req_media = st.selectbox("Select Req Media", df['Req Media'].unique())
        usd = st.selectbox("Select USD", df['USD'].unique(), index=default_usd_global_index)
        level = st.selectbox("Select Level", df['Level'].unique(), index=default_level_index)

        filtered_df = filter_data(df, language, req_media, usd, level)
        filtered_df = filtered_df[filtered_df["Staffing"] > 0]
        filtered_df = remove_outliers_iqr(filtered_df, "Staffing")

        preprocessed_df = preprocess_data(filtered_df)

        split_index = int(len(preprocessed_df) * 0.7)
        split_date = pd.to_datetime(preprocessed_df.iloc[split_index]['Date'])
        train_df = preprocessed_df[preprocessed_df['Date'] <= split_date].drop(columns=['Date'])
        test_df = preprocessed_df[preprocessed_df['Date'] > split_date].drop(columns=['Date'])
        print(preprocessed_df.columns)

        model, scaler, training_score, testing_score, columns_list = train_model(train_df, test_df)

        st.write(f"Training R² Score: {training_score:.4f}")
        st.write(f"Testing R² Score: {testing_score:.4f}")

        train_df = train_df.drop(columns=['Staffing'])
        test_df = test_df.drop(columns=['Staffing'])

        st.write("### Select Week for Analysis")
        start_date = st.date_input("Select Week Starting Date (Sunday)", value=pd.to_datetime("2025-08-17"))
        end_date = start_date + pd.Timedelta(days=6)

        week_data = preprocessed_df[(preprocessed_df['Date'] >= pd.to_datetime(start_date)) &
                                    (preprocessed_df['Date'] <= pd.to_datetime(end_date))]

        week_data_staffing = week_data[['Staffing', 'Demand']]
        week_data = week_data[columns_list]

        if week_data.empty:
            st.write("No data available for the selected week.")
            return

        week_data_avg = week_data.mean()

        st.write("### Weekly Average Values")
        st.write(pd.DataFrame(week_data_avg).T.round(2))

        st.write("### Staffing Change Analysis")
        variable_to_change = st.selectbox("Select variable component to Change:", columns_list)
        percentage_change = st.slider("Percentage Change (%)", -100, 100, step=1)

        adjusted_weights = week_data_avg.copy()
        adjusted_weights[variable_to_change] *= (1 + percentage_change / 100)

        st.write("### Adjusted Weights")
        st.write(pd.DataFrame(adjusted_weights).T.round(2))

        zero_predicted_staffing = predict_changes(model, scaler, columns_list, week_data_avg.values)
        new_predicted_staffing = predict_changes(model, scaler, columns_list, adjusted_weights.values)

        change_in_staffing = new_predicted_staffing - zero_predicted_staffing
        change_in_staffing_percent = (change_in_staffing / zero_predicted_staffing) * 100

        st.write(f"Changed {variable_to_change} Value: {adjusted_weights[variable_to_change]:.2f}")
        st.write(f"Average Staffing Required: {zero_predicted_staffing:.2f}")
        st.write(f"New Predicted Staffing: {new_predicted_staffing:.2f}")
        st.write(f"Change in Staffing: {change_in_staffing:.2f}")
        st.write(f"Change in Staffing (%): {change_in_staffing_percent:.2f}%")

if __name__ == '__main__':
    scn_5()