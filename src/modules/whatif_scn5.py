from modules.utils import st,pd,np
from modules.utils import StandardScaler
from sklearn.decomposition import PCA
from modules.utils import LinearRegression
# from sklearn.preprocessing import StandardScaler

def remove_outliers_iqr(df, column):
    
    # Define the lower and upper bound
    lower_bound = df[column].quantile(0.05)
    upper_bound = df[column].quantile(0.95)
    
    # Remove outliers
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
    df['Date'] = pd.to_datetime(df['Date'])  # Ensure the Date column is parsed as datetime

    # Creating derived variables
    df['ServiceLevel_OccupancyRate'] = df['Service Level'] * df['Occupancy Rate']
    df['Calls_Lag1'] = df['Calls'].shift(1)
    df['Calls_Rolling7'] = df['Calls'].rolling(window=7).mean()
    df['Demand_AHT'] = df['Demand'] * df['AHT']
    df['Abandon_Demand_AHT'] = df['Abandon Demand'] * df['AHT']
     
    # Efficiency-related variables with checks for blank and NaN values
    df['Calls_per_FTE'] = np.where(~df['Staffing'].isna() | df['Staffing'] != 0, df['Calls'] / df['Staffing'], 0)
    df['Demand_per_FTE'] = np.where(~df['Demand'].isna() | df['Staffing'] != 0, df['Demand'] / df['Staffing'], 0)
    
    # Additional features
    df['Weekday'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['Previous_Demand'] = df['Demand'].shift(1)
    df['Previous_AHT'] = df['AHT'].shift(1)
    df['Previous_Occupancy_Rate'] = df['Occupancy Rate'].shift(1)
    df['Weekday_AHT'] = df.groupby(df['Date'].dt.dayofweek)['AHT'].transform('mean')
    df['Weekday_Calls'] = df.groupby(df['Date'].dt.dayofweek)['Calls'].transform('mean')

    # 1. **Service Level Optimization Features**
    df['Min_Service_Level_Target'] = df['Service Level'] * (df['Q2'] / df['Q2'].median())

    # 2. **FTE Requirement Adjustment Features**
    df['FTE_per_Requirement'] = df['Staffing'] / df['Requirement']  # FTE per required staffing
    df['FTE_Change_per_Service_Level'] = df['FTE_per_Requirement'] * (df['Service Level'] /
                                                                                      df['Service Level'].median())

    # 3. **Demand Impact on Staffing**
    df['FTE_per_Demand'] = df['Staffing'] / df['Demand']
    df['Projected_FTE_for_10%_Demand_Increase'] = df['FTE_per_Demand'] * (df['Demand'] * 1.10)

    # 4. **OCC Assumption Impact Analysis**
    df['Service_Level_per_OCC'] = df['Service Level'] / df['Occ Assumption']  # SL per occupancy
    df['Q2_per_OCC'] = df['Q2'] / df['Occ Assumption']  # Q2 per occupancy
    df['ABN_per_OCC'] = df['ABN %'] / df['Occ Assumption']  # ABN% per occupancy

    # 5. **Staffing Requirement Sensitivity Analysis**
    df['Q2_per_Staffing'] = np.where(~df['Staffing'].isna() & (df['Staffing'] != 0), df['Q2'] / df['Staffing'], 0)
    df['ABN_per_Staffing'] = np.where(~df['Staffing'].isna() & df['Staffing'] != 0, df['ABN %'] / df['Staffing'], 0)
    df['Service_Level_per_Staffing'] = np.where(~df['Staffing'].isna() & df['Staffing'] != 0, df['Service Level'] / df['Staffing'], 0)

    # 6. **Calls per FTE Metrics**
    df['Service_Level_per_Calls_per_FTE'] = df['Service Level'] / df['Calls_per_FTE']
    df['Q2_per_Calls_per_FTE'] = df['Q2'] / df['Calls_per_FTE']
    df['ABN_per_Calls_per_FTE'] = df['ABNs'] / df['Calls_per_FTE']    

    # Handling NaN and infinite values for derived variables
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    
    filtered_df = df[['Occupancy Rate','Date', 'Calls', 'Service Level', 'Q2', 'AHT', 'Abandon Demand','Staffing','Staffing Diff','ABN %',
                   'Calls_Lag1', 'Calls_Rolling7','Demand_AHT', 'Abandon_Demand_AHT', 'Weekday', 'Month',
                   'Quarter','Previous_Demand', 'Previous_AHT', 'Previous_Occupancy_Rate', 'Weekday_AHT',
                   'Weekday_Calls','Calls_per_FTE','Demand_per_FTE' , 'Min_Service_Level_Target',
                   'FTE_per_Requirement','FTE_Change_per_Service_Level','FTE_per_Demand','Projected_FTE_for_10%_Demand_Increase',
                   'Service_Level_per_OCC','Q2_per_OCC', 'ABN_per_OCC', 'Q2_per_Staffing', 'ABN_per_Staffing',
                   'Service_Level_per_Staffing','Service_Level_per_Calls_per_FTE','Q2_per_Calls_per_FTE', 'ABN_per_Calls_per_FTE']]
    return filtered_df


def train_model_with_pca(train_df, test_df, n_components=0.95):
    columns_to_convert = ['Occupancy Rate','Calls', 'Service Level', 'Q2', 'AHT', 'Abandon Demand','Staffing Diff','ABN %',
                   'Calls_Lag1', 'Calls_Rolling7','Demand_AHT', 'Abandon_Demand_AHT', 'Weekday', 'Month',
                   'Quarter','Previous_Demand', 'Previous_AHT', 'Previous_Occupancy_Rate', 'Weekday_AHT',
                   'Weekday_Calls','Calls_per_FTE','Demand_per_FTE', 'Min_Service_Level_Target',
                   'FTE_per_Requirement','FTE_Change_per_Service_Level','FTE_per_Demand','Projected_FTE_for_10%_Demand_Increase',
                   'Service_Level_per_OCC','Q2_per_OCC', 'ABN_per_OCC', 'Q2_per_Staffing', 'ABN_per_Staffing',
                   'Service_Level_per_Staffing','Service_Level_per_Calls_per_FTE','Q2_per_Calls_per_FTE', 'ABN_per_Calls_per_FTE']
    
    train_df[columns_to_convert] = train_df[columns_to_convert].apply(pd.to_numeric, errors='coerce')
    test_df[columns_to_convert] = test_df[columns_to_convert].apply(pd.to_numeric, errors='coerce')

    train_df.fillna(train_df.mean(numeric_only=True), inplace=True)
    test_df.fillna(test_df.mean(numeric_only=True), inplace=True)
    train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    train_df.fillna(train_df.mean(numeric_only=True), inplace=True)
    test_df.fillna(test_df.mean(numeric_only=True), inplace=True)
    
    X_train = train_df[columns_to_convert]
    X_test = test_df[columns_to_convert]
    y_train = train_df['Staffing']
    y_test = test_df['Staffing']  

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)  

    # Initialize and fit the Linear Regression model
    model = LinearRegression()
    model.fit(X_train_pca, y_train)

    training_score = model.score(X_train_pca, y_train)
    testing_score = model.score(X_test_pca, y_test)
    
    return model, scaler, training_score, testing_score, pca, columns_to_convert

def predict_changes(model, scaler, pca, feature_columns, adjusted_pca_data):
    # Ensure the adjusted PCA data is in the correct format
    adjusted_pca_data = adjusted_pca_data.reshape(1, -1)
    
    # Perform the inverse transformation to get the original feature space
    sample_data_scaled = pca.inverse_transform(adjusted_pca_data)
    
    # Since scaling was applied previously, reverse the scaling
    sample_data = scaler.inverse_transform(sample_data_scaled)
    
    # Convert to DataFrame and reindex to match feature columns
    sample_data_df = pd.DataFrame(sample_data, columns=feature_columns)
    sample_data_df = sample_data_df.reindex(columns=feature_columns, fill_value=0)
    
    # Apply scaling and PCA transformation again for the model prediction
    sample_data_scaled_again = scaler.transform(sample_data_df)
    sample_data_pca = pca.transform(sample_data_scaled_again)
    
    # Predict staffing based on adjusted PCA values
    predicted_staffing = model.predict(sample_data_pca)
    
    return predicted_staffing[0]


def scn_5():
    st.title("FTE Requirement Prediction with Variable Analysis")

    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.write("Data loaded successfully!")

        language = st.selectbox("Select Language:", df['Language'].unique())
        req_media = st.selectbox("Select Req Media:", df['Req Media'].unique())
        usd = st.selectbox("Select USD:", df['USD'].unique())
        level = st.selectbox("Select Level:", df['Level'].unique())

        filtered_df = filter_data(df, language, req_media, usd, level)
        
        # Remove outlier from staffing
        remove_outlier_df = filtered_df[filtered_df["Staffing"] > 0]        
        remove_outlier_df = remove_outliers_iqr(remove_outlier_df, "Staffing")
        
        preprocessed_df = preprocess_data(remove_outlier_df)

        split_index = int(len(preprocessed_df) * 0.7)
        
        split_date = pd.to_datetime(preprocessed_df.iloc[split_index]['Date'], format='%m-%d-%Y')
        train_df = preprocessed_df[pd.to_datetime(preprocessed_df['Date'], format='%m-%d-%Y') <= split_date]
        test_df = preprocessed_df[pd.to_datetime(preprocessed_df['Date'], format='%m-%d-%Y') > split_date]
        
        # Exclude 'Date' columns from PCA
        train_df = train_df.drop(columns=['Date'])
        test_df = test_df.drop(columns=['Date'])

        model, scaler, training_score, testing_score, pca, columns_list = train_model_with_pca(train_df, test_df)

        st.write(f"Training R^2 Score: {training_score:.4f}")
        st.write(f"Testing R^2 Score: {testing_score:.4f}")
        
        train_df = train_df.drop(columns=['Staffing'])
        test_df = test_df.drop(columns=['Staffing'])        

        st.write("### PCA Components")
        pca_components = [f'PC{i+1}' for i in range(pca.n_components_)]
        components_df = pd.DataFrame(pca.components_, columns=train_df.columns[:len(pca.components_[0])], index=pca_components)        

        st.write("### Select Week for Analysis")
        start_date = st.date_input("Select Week Starting Date (Sunday)", value=pd.to_datetime("2024-01-01"))
        end_date = start_date + pd.Timedelta(days=6)
        
        week_data = preprocessed_df[(pd.to_datetime(preprocessed_df['Date'], format='%m-%d-%Y') >= pd.to_datetime(start_date)) & 
                                    (pd.to_datetime(preprocessed_df['Date'], format='%m-%d-%Y') <= pd.to_datetime(end_date))]
        
        week_data_staffing=week_data[['Staffing']]
        
        week_data=week_data[columns_list]
        
        if week_data.empty:
            st.write("No data available for the selected week.")
            return

        week_data_avg = week_data.mean()

        st.write("### Weekly Average Values")
        st.write(pd.DataFrame(week_data_avg).T)

        st.write("### Staffing Change Analysis")
        variable_to_change = st.selectbox("Select variable component to Change:", train_df.columns[:len(pca.components_[0])])
        percentage_change = st.slider("Percentage Change (%)", -100, 100, step=1)

        # Adjust the PCA components based on the percentage change
        adjusted_components = pca.components_.copy()
        if variable_to_change in train_df.columns:
            column_index = train_df.columns.tolist().index(variable_to_change)
            adjusted_components[:, column_index] *= (1 + percentage_change / 100)

        pca_components_adj = [f'PC{i+1}' for i in range(pca.n_components_)]
        components_df_adj = pd.DataFrame(adjusted_components, columns=train_df.columns[:len(adjusted_components[0])], index=pca_components_adj)

        st.write("### Staffing requirement vs variable_to_change")
        
        # Adjust the weights based on the percentage change
        adjusted_weights = week_data_avg.copy()
        if variable_to_change in train_df.columns:
            adjusted_weights[variable_to_change] *= (1 + percentage_change / 100)

        st.write("### Adjusted Weights")
        st.write(pd.DataFrame(adjusted_weights).T)

        # Update input data and apply PCA transformation with adjusted weights
        input_data = adjusted_weights[components_df.columns].values.reshape(1, -1)
        input_data_scaled = scaler.transform(input_data)
        input_data_pca = pca.transform(input_data_scaled)
        
        new_predicted_staffing = predict_changes(model, scaler, pca, components_df.columns, input_data_pca)

        change_in_staffing_rate = (new_predicted_staffing - week_data_staffing["Staffing"].mean()) / week_data_staffing["Staffing"].mean()       
        st.write(f"Changed {variable_to_change} Value: {week_data_avg[variable_to_change]*(1 + percentage_change / 100):.2f}")
        st.write(f"Average Staffing for the week: {week_data_staffing['Staffing'].mean():.2f}")
        st.write(f"New Predicted Staffing: {new_predicted_staffing:.2f}")
        st.write(f"Change in Staffing: {change_in_staffing_rate * 100:.2f}%")

if __name__ == '__main__':
    scn_5()




