# from modules.utils import st
# from modules.whatif_scn1 import scn1
# from modules.whatif_scn2 import scn2
# from modules.whatif_scn3 import scn3
# from modules.whatif_scn4 import scn_4
# from modules.whatif_scn5 import scn_5
# from modules.whatif_scn6_7 import scn6_7



# # Dictionary mapping scenarios to functions
# SCENARIOS = {
#     "Scenario 1": scn1,
#     "Scenario 2": scn2,
#     "Scenario 3": scn3,
#     "Scenario 4": scn_4,
#     "Scenario 5": scn_5,
#     "Scenario 6": scn6_7,
# }

# def main():
#     # st.title("What-If Analysis Tool")
    
#     # Dropdown for selecting a scenario
#     selected_scenario = st.selectbox("Select a Scenario", list(SCENARIOS.keys()))
    
#     st.write(f"**You selected:** {selected_scenario}")
    
#     # Run the corresponding function
#     if selected_scenario in SCENARIOS:
#         SCENARIOS[selected_scenario]()
#     else:
#         st.error("Invalid Scenario Selected")
    
# if __name__ == "__main__":
#     main()

from modules.utils import st
from modules.whatif_scn1 import scn1
from modules.whatif_scn2 import scn2
from modules.whatif_scn3 import scn3
from modules.whatif_scn4 import scn_4
from modules.whatif_scn5 import scn_5
from modules.whatif_scn6_7 import scn6_7
# st.set_page_config(page_title="What-if Analysis", layout="wide")

# Dictionary mapping scenarios to functions
SCENARIOS = {
    "Scenario 1 - Average SL% while aiming to Q2 target": scn1,
    "Scenario 2 - Staffing requirements change based on occupancy assumptions, occupancy rate, and interval-based demand changes": scn2,
    "Scenario 3 - Staffing requirement based on demand change, Q2 time, occupancy assumptions, occupancy rate": scn3,
    "Scenario 4 - Work in progress": scn_4,
    "Scenario 5 - Actual staffing based on KPI adjustments (input variables be selected based on importance)": scn_5,
    "Scenario 6 - CallsPerFTE impact on other variables": scn6_7,
}

# Scenario descriptions
SCENARIO_DESCRIPTIONS = {
    "Scenario 1": "Determine the lowest service level % we should run at while still maintaining our Q2 time goal.",
    "Scenario 2": "Determine the change in FTE requirements if we adjust occupancy assumptions, occupancy rate, and interval-based demand changes",
    "Scenario 3": "The model estimates FTE requirements based on input variables including weekly demand increase %, Q2 time, occupancy assumptions, and occupancy rate",
    "Scenario 4": "Estimate service level, Q2 time, or abandonment rate change if we increase/decrease OCC assumption by a given percent.",
    "Scenario 5": "Predict KPI changes (input variables will be automatically in the drop down based on importance)",
    "Scenario 6": "Assess change in calls per FTE for any of the above scenarios (1-5)."
}

def main():
    # Dropdown for selecting a scenario
    selected_scenario = st.selectbox("Select a Scenario", list(SCENARIOS.keys()))
    
    # Info expander below the dropdown
    with st.expander("ℹ️ Scenario Info"):
        st.markdown(f"**{selected_scenario}**: {SCENARIO_DESCRIPTIONS.get(selected_scenario, 'No description available.')}")

    st.write(f"**You selected:** {selected_scenario}")
    
    # Run the corresponding function
    if selected_scenario in SCENARIOS:
        SCENARIOS[selected_scenario]()
    else:
        st.error("Invalid Scenario Selected")

if __name__ == "__main__":
    main()