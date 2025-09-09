import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, r2_score

def load_and_prepare_data(file_path):
    df = pd.read_excel(file_path, sheet_name="Sheet1")
    columns_to_convert = ["Q2", "ABN %", "Loaded AHT", "Met", "Missed"]
    for col in columns_to_convert:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Requirement", "Staffing", "Demand"])
    df = df.fillna(0)
    return df

def train_regression_model(df):
    features = ["Q2", "ABN %", "Occ Assumption", "Staffing", "Calls", "Occupancy Rate"]
    target = "Requirement"
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Lasso(alpha=0.1)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    # Print coefficients and equation
    coefficients = model.coef_
    intercept = model.intercept_
    equation = "Requirement = " + " + ".join([f"{coef:.4f} * {feature}" for coef, feature in zip(coefficients, features)]) + f" + {intercept:.4f}"
    
    return model, mae, r2_train, r2_test, cv_mean, cv_std, coefficients, equation

def analyze_scenarios(df, model):
    q2_goal = df["Q2"].median()
    abn_goal = df["ABN %"].median()
    service_levels = np.linspace(0.80, 0.99, 20)
    test_data = pd.DataFrame({
        "Q2": [q2_goal] * len(service_levels),
        "ABN %": [abn_goal] * len(service_levels),
        "Occ Assumption": [df["Occ Assumption"].median()] * len(service_levels),
        "Staffing": [df["Staffing"].median()] * len(service_levels),
        "Calls": [df["Calls"].median()] * len(service_levels),
        "Occupancy Rate": [df["Occupancy Rate"].median()] * len(service_levels),
    })
    fte_predictions = model.predict(test_data)
    lowest_service_level = service_levels[np.argmin(np.abs(fte_predictions - df["Requirement"].median()))]
    
    # Additional Scenarios
    demand_changes = [5, 10, 15, 20, 25]
    demand_results = {change: model.predict(pd.DataFrame({
        "Q2": [q2_goal], "ABN %": [abn_goal], "Occ Assumption": [df["Occ Assumption"].median()],
        "Staffing": [df["Staffing"].median()], "Calls": [df["Calls"].median() * (1 + change / 100)], "Occupancy Rate": [df["Occupancy Rate"].median()]
    }))[0] for change in demand_changes}
    
    return lowest_service_level, demand_results

def main():
    st.title("WFM Regression Analysis Dashboard")
    file_path = "Data/data_to_analyze.xlsx"
    df = load_and_prepare_data(file_path)
    model, mae, r2_train, r2_test, cv_mean, cv_std, coefficients, equation = train_regression_model(df)
    st.write(f"Model Performance: MAE = {mae:.2f}, R2 Train Score = {r2_train:.4f}, R2 Test Score = {r2_test:.4f}")
    st.write(f"Cross-Validation R2: Mean = {cv_mean:.4f}, Std = {cv_std:.4f}")
    st.write("Lasso Regression Equation:")
    st.write(equation)
    lowest_service_level, demand_results = analyze_scenarios(df, model)
    st.write(f"Lowest Service Level Required: {lowest_service_level:.2f}")
    st.write("Predicted FTE Requirements for Demand Changes:")
    for change, fte in demand_results.items():
        st.write(f"Demand Increase {change}%: Predicted FTE = {fte:.2f}")

if __name__ == "__main__":
    main()
