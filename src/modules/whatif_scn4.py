# Reload necessary libraries after reset
#Scenario 4
from modules.utils import st,pd,np
from datetime import datetime, timedelta
import calendar
from modules.utils import train_test_split
from modules.utils import StandardScaler
from modules.utils import LinearRegression
from modules.utils import r2_score, mean_squared_error,mean_absolute_error
from modules.utils import joblib
import openpyxl


# # Load the uploaded file again
# file_path = "Data/data_to_analyze.xlsx"
# df = pd.read_excel(file_path, sheet_name="Sheet1")

# # Rename columns for consistency
# df.rename(columns={'startDate': 'Date', 'Met': 'Service Level', 'Loaded AHT': 'AHT'}, inplace=True)

# # Convert necessary columns to numeric
# numeric_columns = ["Q2", "ABN %", "AHT", "Service Level", "Missed", "Occ Assumption"]
# for col in numeric_columns:
#     df[col] = pd.to_numeric(df[col], errors="coerce")

# # Fill missing values
# df = df.fillna(0)

# # Selecting the best predictor "Missed" along with other relevant features
# selected_features = ["Missed","Calls", "Demand", "AHT"]
# targets = ["Service Level", "Q2", "ABN %"]

# # Dictionary to store model results
# model_results = {}

# for target in targets:
#     X = df[selected_features]
#     y = df[target]

#     # Splitting data into training and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Training a Linear Regression model
#     model = LinearRegression()
#     model.fit(X_train, y_train)

#     # Predictions
#     y_pred = model.predict(X_test)

#     # Evaluation metrics
#     mae = mean_absolute_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)

#     # Save model
#     model_filename = f"scenario_models/model_{target}_4.pkl"
#     joblib.dump(model, model_filename)

#     # Store results
#     model_results[target] = {
#         "Mean Absolute Error": mae,
#         "R² Score": r2,
#         "Regression Equation": f"{target} = {model.intercept_:.4f} + " +
#                                " + ".join([f"({coef:.4f} * {feature})" for coef, feature in zip(model.coef_, selected_features)])
#     }

# # Display model results
# print(model_results)

# # Load trained models
# model_service_level = joblib.load(r"scenario_models/model_Service Level_4.pkl")
# model_q2 = joblib.load(r"scenario_models/model_Q2_4.pkl")
# model_abn = joblib.load(r"scenario_models/model_ABN %_4.pkl")

# # Define feature inputs
# selected_features = ["Missed", "Calls", "Demand", "AHT"]

# def predict_service_level(missed, calls, demand, aht):
#     input_data = pd.DataFrame([[missed, calls, demand, aht]], columns=selected_features)
#     return model_service_level.predict(input_data)[0]

# def predict_q2(missed, calls, demand, aht):
#     input_data = pd.DataFrame([[missed, calls, demand, aht]], columns=selected_features)
#     return model_q2.predict(input_data)[0]

# def predict_abn(missed, calls, demand, aht):
#     input_data = pd.DataFrame([[missed, calls, demand, aht]], columns=selected_features)
#     return model_abn.predict(input_data)[0]

def scn_4():
    st.title("What-If Analysis: Impact of change in Occ assumption on service level, Q2 and abandonment rates")

    st.header("We are in progress")
    
    # st.header("Enter Inputs")
    # missed = st.number_input("Missed Calls", min_value=0, max_value=1000, value=10, step=1)
    # calls = st.number_input("Total Calls", min_value=0, max_value=100000, value=30000, step=100)
    # demand = st.number_input("Demand", min_value=0.0, max_value=500000.0, value=300000.0, step=1000.0)
    # aht = st.number_input("Average Handle Time (AHT)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
    
    # if st.button("Predict KPIs"):
    #     service_level = predict_service_level(missed, calls, demand, aht)
    #     q2_time = predict_q2(missed, calls, demand, aht)
    #     abn_rate = predict_abn(missed, calls, demand, aht)
        
    #     st.header("Prediction Results")
    #     st.write(f"**Predicted Service Level:** {service_level:.4f}")
    #     st.write(f"**Predicted Q2 Time:** {q2_time:.4f}")
    #     st.write(f"**Predicted Abandon Rate:** {abn_rate:.4f}")

if __name__ == "__main__":
    scn_4()
