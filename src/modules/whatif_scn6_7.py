from modules.utils import st,pd,np
from modules.utils import StandardScaler
from modules.utils import LinearRegression
from modules.utils import r2_score, mean_squared_error
from modules.utils import train_test_split
# from sklearn.preprocessing import StandardScaler


# Function to load and prepare data
def load_and_prepare_data(file_path):
    df = pd.read_excel(file_path, sheet_name="Sheet1")
    columns_to_convert = ["Q2", "ABN %", "Loaded AHT", "Met", "Missed"]
    for col in columns_to_convert:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Requirement", "Staffing", "Demand"])
    df = df.fillna(0)
    return df

# Function to train regression model
def train_regression_model(df):
    features = ["Q2", "ABN %", "Occ Assumption", "Staffing", "Calls", "Occupancy Rate"]
    target = "Requirement"
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Function for Scenario 1: How does calls per FTE change in different scenarios?
# def calls_per_fte_scenario(df, percentage_changes):
#     results = []
#     for change in percentage_changes:
#         adjusted_calls = df["Calls"].median()
#         adjusted_fte = df["Staffing"].median() * (1 + change / 100)
#         calls_per_fte = adjusted_calls / adjusted_fte if adjusted_fte != 0 else 0
#         results.append({
#             "FTE Change %": change,
#             "Predicted Calls per FTE": calls_per_fte
#         })
#     return pd.DataFrame(results)

# Function for Scenario 2: How do KPIs change if we adjust calls per FTE?
def kpi_impact_scenario(df, model, percentage_changes, q2, abn, occ):
    results = []
    for change in percentage_changes:
        adjusted_calls = df["Calls"].median() * (1 + change / 100)
        adjusted_fte = df["Staffing"].median()
        calls_per_fte = adjusted_calls / adjusted_fte if adjusted_fte != 0 else 0

        test_scenario = pd.DataFrame({
            "Q2": [q2 * (1 + change / 100)],
            "ABN %": [abn/100 * (1 + change / 100)],
            "Occ Assumption": [df["Occ Assumption"].median()],
            "Staffing": [adjusted_fte],
            "Calls": [adjusted_calls],
            "Occupancy Rate": [occ/100 * (1 + change / 100)]
        })

        fte_pred = model.predict(test_scenario)[0]

        results.append({
            "Change - Input Variables %": change,
            "Calls Per FTE (Staffing)": calls_per_fte,
            "Predicted FTE Requirement": fte_pred,
            "Adjusted Q2 Time": q2 * (1 + change / 100),
            "Adjusted Abandonment Rate": abn * (1 + change / 100),
            "Adjusted Occupancy Rate": occ * (1 + change / 100)
        })
    return pd.DataFrame(results)

def analyze_scenarios(df, model, q2, abn, occ):
    test_data = pd.DataFrame({
        "Q2": [q2],
        "ABN %": [abn/100],
        "Occ Assumption": [occ/100],
        "Staffing": [df["Staffing"].median()],
        "Calls": [df["Calls"].median()],
        "Occupancy Rate": [df["Occupancy Rate"].median()]
    })
    fte_prediction = model.predict(test_data)[0]
    return fte_prediction

st.title("Change in calls per FTE (Staffing) on other KPI")
st.sidebar.header("User Inputs")

def scn6_7():

    file_path = st.file_uploader("Upload your Excel file", type=["xlsx"])

    if file_path is not None:
        df = load_and_prepare_data(file_path)
        
        st.header("Filter Data------------------------->")
        # Dropdown filters based on columns
        language = st.selectbox("Select Language", df['Language'].unique())
        req_media = st.selectbox("Select Req Media", df['Req Media'].unique())
        usd = st.selectbox("Select USD", df['USD'].unique())
        level = st.selectbox("Select Level", df['Level'].unique())

        # Filter the DataFrame based on selected values
        df = df[(df['Language'] == language) & (df['Req Media'] == req_media) & 
                     (df['USD'] == usd) & (df['Level'] == level)]
        
        st.header("Output------------------------------>")        
        
        model = train_regression_model(df)
    
        q2 = st.sidebar.slider("Set Q2 Time", min_value=0, max_value=100, value=20, step=1)
        abn = st.sidebar.slider("Set Abandon Rate (%)", min_value=0, max_value=10, value=2, step=1)
        occ = st.sidebar.slider("Set Occupancy Rate (%)", min_value=0, max_value=100, value=80, step=1)
    
        fte_prediction = analyze_scenarios(df, model, q2, abn, occ)
        st.write(f"Predicted FTE based on user inputs: {fte_prediction:.2f}")
    
        st.header("KPI Impact due to Calls per FTE (Staffing) Changes")
        percentage_changes = np.array([-25, -20, -15,-10,-5, -2, -1, 0, 1, 2, 5, 10, 15, 20, 25])  # Variations in FTE assumptions
        kpi_impact_df = kpi_impact_scenario(df, model, percentage_changes, q2, abn, occ)
        st.dataframe(kpi_impact_df.round(2)) 

if __name__ == '__main__':
    scn6_7()
