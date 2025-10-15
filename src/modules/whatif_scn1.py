from modules.utils import st,pd,np
from modules.utils import train_test_split
from modules.utils import LinearRegression
from modules.utils import mean_absolute_error, r2_score
from modules.utils import joblib

# Function to load and clean data
def load_and_prepare_data(file_path):
    df = pd.read_excel(file_path, sheet_name="Sheet1")
    
    # Ensure proper column naming
    df.rename(columns={'startDate': 'Date', 'Met': 'Service Level', 'Loaded AHT': 'AHT'}, inplace=True)
    
    # Convert necessary columns to numeric
    numeric_columns = ["Q2", "ABN %", "AHT", "Service Level", "Missed"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df = df.fillna(0)
    return df

# Train Linear Regression Model
def train_regression_model(df):
    features = ["Q2", "ABN %"]
    target = "Service Level"
    
    X = df[features]
    y = df[target]
    
    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model training
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Model evaluation
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Save the model
    joblib.dump(model, 'model_whatifs.pkl')

    # Get regression coefficients
    beta_0 = model.intercept_
    beta_1, beta_2 = model.coef_
    
    return model, mae, r2, beta_0, beta_1, beta_2

# Predict Service Level for given Q2 and ABN %
def analyze_scenarios(model, q2, abn):
    test_data = pd.DataFrame({
        "Q2": [q2],
        "ABN %": [abn / 100]
    })
    sl_prediction = model.predict(test_data)[0]
    return round(sl_prediction, 4)

# Analyze the impact of Q2 changes on Service Level
def sl_impact_of_q2_increase(df, model, q2, abn):
    q2_changes = [-2, -1, 0, 1, 2]
    predictions = []
    
    for change in q2_changes:
        modified_q2 = q2 + change
        sl_pred = model.predict(pd.DataFrame({
            "Q2": [modified_q2], "ABN %": [abn / 100]
        }))[0]
        predictions.append((change, round(sl_pred, 4)))
    
    return predictions

# Streamlit App UI
st.title("What-If Analysis: Service Level vs Q2 time")

def scn1():
    file_path = st.file_uploader("Upload your Excel file", type=["xlsx"])

    if file_path is not None:
        df = load_and_prepare_data(file_path)
        
        st.header("Filter Data")
        
        # Dropdown filters (Fixing 'USD Req' KeyError by replacing with 'USD')
        usd_options = df['USD'].unique()
        level_options = df['Level'].unique()        
        default_usd_global_index = list(usd_options).index("Combined") if "Combined" in usd_options else 0
        default_level_index = list(level_options).index("Combined") if "Combined" in usd_options else 0
        cols = st.columns(2)
        language = cols[0].selectbox("Select Language", df['Language'].unique())
        req_media = cols[1].selectbox("Select Req Media", df['Req Media'].unique())

        cols = st.columns(2)  
        usd = cols[0].selectbox("Select USD", df['USD'].unique(), index=default_usd_global_index)  # Fixed column name
        level = cols[1].selectbox("Select Level", df['Level'].unique(), index=default_level_index)

        # Apply filters
        df_filtered = df[(df['Language'] == language) & 
                         (df['Req Media'] == req_media) &  
                         (df['USD'] == usd) &  # Fixed column name
                         (df['Level'] == level)]
        
        # Train Model
        if not df_filtered.empty:
            model, mae, r2, beta_0, beta_1, beta_2 = train_regression_model(df_filtered)
            # # Display model metrics
            # st.write(f"**Model Performance**")
            # st.write(f"Mean Absolute Error: **{mae:.2f}**")
            # st.write(f"R² Score: **{r2:.4f}**")

            # # Show Regression Equation
            # st.write(f"**Regression Equation:**")
            # st.latex(f"Service\\ Level = {beta_0:.4f} + ({beta_1:.4f} \\times Q2) + ({beta_2:.4f} \\times ABN\\%)")

            # User inputs for predictions
            cols = st.columns(2)
            q2_val = cols[0].number_input("Set Q2 Time:", min_value=0, max_value=100, value=20, step=1)
            st.write("Abandonment rate is just a parameter used in the equation- impct is very minimal")
            abn_val = cols[1].number_input("Set Abandon Rate (%)", min_value=0, max_value=10, value=2, step=1)
            
            st.header("Prediction Results")
            
            sl_prediction = analyze_scenarios(model, q2_val, abn_val)
            st.write(f"Predicted Service Level: **{sl_prediction:.4f}**")
        
            st.header("Impact of Q2 Changes on Service Level")
            sl_impact = sl_impact_of_q2_increase(df_filtered, model, q2_val, abn_val)
            
            for change, sl_pred in sl_impact:
                st.write(f"Q2 Value {q2_val + change}: Predicted SL = **{sl_pred:.4f}**")
        
        else:
            st.warning("No data available for the selected filters. Try different values.")

if __name__ == '__main__':
    scn1()
