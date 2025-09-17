from modules.utils import st, pd, np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def remove_outliers_iqr(df, column):
    lower_bound = df[column].quantile(0.05)
    upper_bound = df[column].quantile(0.95)
    df_no_outliers = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_no_outliers

def load_data(uploaded_file):
    df = pd.read_excel(uploaded_file)
    return df

def filter_data(df, language, req_media, usd, level):
    filtered_df = df[(df['Language'] == language) &
                     (df['Req Media'] == req_media) &
                     (df['USD'] == usd) &
                     (df['Level'] == level)]
    return filtered_df

def preprocess_data(df):
    df['Abandon Demand'] = df['ABN %'] * df['Demand']
    df.rename(columns={'startDate per day': 'Date', 'Met': 'Service Level', 'Loaded AHT': 'AHT'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])

    df['Calls_per_FTE'] = np.where((~df['Staffing'].isna()) & (df['Staffing'] != 0), df['Calls'] / df['Staffing'], 0)
    df['Demand_per_FTE'] = np.where((~df['Demand'].isna()) & (df['Staffing'] != 0), df['Demand'] / df['Staffing'], 0)

    df['Weekday'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['FTE_per_Demand'] = df['Staffing'] / df['Demand']

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)

    filtered_df = df[['Date', 'Abandon Demand', 'Calls_per_FTE', 'ABN %', 'AHT', 'Q2', 'Service Level',
                      'Staffing Diff', 'Demand', 'Occupancy Rate', 'Calls', 'Demand_per_FTE', 'Staffing']]
    return filtered_df

def train_model(train_df, test_df, importance_threshold=0.01):
    initial_features = ['Abandon Demand', 'Calls_per_FTE', 'ABN %', 'AHT', 'Q2', 'Service Level',
                        'Staffing Diff', 'Demand', 'Occupancy Rate', 'Calls', 'Demand_per_FTE']

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
    predicted_staffing = model.predict(scaled_data)
    return predicted_staffing[0]

def scn_5():
    st.title("Staffing Requirement Prediction with Variable Analysis")

    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.write("Data loaded successfully!")

        usd_options = data['USD'].unique()
        level_options = data['Level'].unique()        
        default_usd_global_index = list(usd_options).index("Combined") if "Combined" in usd_options else 0
        default_level_index = list(level_options).index("Combined") if "Combined" in usd_options else 0
        language = st.selectbox("Select Language", df['Language'].unique())
        req_media = st.selectbox("Select Req Media", df['Req Media'].unique())  
        usd = st.selectbox("Select USD", df['USD'].unique(), index=default_usd_global_index)  # Fixed column name
        level = st.selectbox("Select Level", df['Level'].unique(), index=default_level_index)

        filtered_df = filter_data(df, language, req_media, usd, level)

        remove_outlier_df = filtered_df[filtered_df["Staffing"] > 0]
        remove_outlier_df = remove_outliers_iqr(remove_outlier_df, "Staffing")

        preprocessed_df = preprocess_data(remove_outlier_df)

        split_index = int(len(preprocessed_df) * 0.7)
        split_date = pd.to_datetime(preprocessed_df.iloc[split_index]['Date'], format='%m-%d-%Y')
        train_df = preprocessed_df[pd.to_datetime(preprocessed_df['Date'], format='%m-%d-%Y') <= split_date]
        test_df = preprocessed_df[pd.to_datetime(preprocessed_df['Date'], format='%m-%d-%Y') > split_date]

        train_df = train_df.drop(columns=['Date'])
        test_df = test_df.drop(columns=['Date'])

        model, scaler, training_score, testing_score, columns_list = train_model(train_df, test_df)

        st.write(f"Training R^2 Score: {training_score:.4f}")
        st.write(f"Testing R^2 Score: {testing_score:.4f}")
        # st.write("Selected Features Based on Importance:")
        # st.write(columns_list)

        train_df = train_df.drop(columns=['Staffing'])
        test_df = test_df.drop(columns=['Staffing'])

        st.write("### Select Week for Analysis")
        start_date = st.date_input("Select Week Starting Date (Sunday)", value=pd.to_datetime("2025-08-17"))
        end_date = start_date + pd.Timedelta(days=6)

        week_data = preprocessed_df[(pd.to_datetime(preprocessed_df['Date'], format='%m-%d-%Y') >= pd.to_datetime(start_date)) &
                                    (pd.to_datetime(preprocessed_df['Date'], format='%m-%d-%Y') <= pd.to_datetime(end_date))]

        week_data_staffing = week_data[['Staffing', 'Demand']]
        week_data = week_data[columns_list]

        if week_data.empty:
            st.write("No data available for the selected week.")
            return

        week_data_avg = week_data.mean()

        st.write("### Weekly Average Values")
        formatted_weights = pd.DataFrame(week_data_avg).T.round(2)
        formatted_weights.index = ['']  # Remove index label
        st.write(formatted_weights)

        st.write("### Staffing Change Analysis")
        variable_to_change = st.selectbox("Select variable component to Change:", columns_list)
        percentage_change = st.slider("Percentage Change (%)", -100, 100, step=1)

        adjusted_weights = week_data_avg.copy()
        if variable_to_change in columns_list:
            adjusted_weights[variable_to_change] *= (1 + percentage_change / 100)

        st.write("### Adjusted Weights")

        formatted_weights = pd.DataFrame(adjusted_weights).T.round(2)
        formatted_weights.index = ['']  # Remove index label
        st.write(formatted_weights)


        # Predict staffing at 0% change
        zero_change_weights = week_data_avg.copy()
        zero_predicted_staffing = predict_changes(model, scaler, columns_list, zero_change_weights.values)

        # Apply percentage change to selected variable
        adjusted_weights = week_data_avg.copy()
        adjusted_weights[variable_to_change] *= (1 + percentage_change / 100)

        # Predict staffing at new % change
        input_data = adjusted_weights.values.reshape(1, -1)
        new_predicted_staffing = predict_changes(model, scaler, columns_list, input_data)

        # Calculate change relative to zero-change prediction
        change_in_staffing = new_predicted_staffing - zero_predicted_staffing

        # Display results
        st.write(f"Changed {variable_to_change} Value: {adjusted_weights[variable_to_change]:.2f}")
        st.write(f"Average Staffing Required: {zero_predicted_staffing:.2f}")
        st.write(f"New Predicted Staffing: {new_predicted_staffing:.2f}")
        st.write(f"Change in Staffing: {change_in_staffing:.2f}")

        change_in_staffing_percent = (change_in_staffing / zero_predicted_staffing) * 100
        st.write(f"Change in Staffing (%): {change_in_staffing_percent:.2f}%")


if __name__ == '__main__':
    scn_5()
