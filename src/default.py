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
    "Scenario 1 - Average SL% while aiming to Q2 target and ABN % goal": scn1,
    "Scenario 2 - Requirement Change based on Q2 Time, ABN % and Occ adjustments": scn2,
    "Scenario 3 - Needed FTE based on Demand Changes": scn3,
    "Scenario 4 - Work in progress": scn_4,
    "Scenario 5 - Staffing Requirements based on KPI adjustments": scn_5,
    "Scenario 6 - Calls per FTE Change": scn6_7,
}

# Scenario descriptions
SCENARIO_DESCRIPTIONS = {
    "Scenario 1": "Determine the lowest service level % we should run at while still maintaining our Q2 time and Abandonment rate goals.",
    "Scenario 2": "Determine the change in FTE requirements if we adjust service level, Q2 time, abandonment rate, OCC assumption, or weekly fluctuation.",
    "Scenario 3": "Determine how many FTEs we need to add if we increase demand by a certain % per month/week/year.",
    "Scenario 4": "Estimate service level, Q2 time, or abandonment rate change if we increase/decrease OCC assumption by a given percent.",
    "Scenario 5": "Predict KPI changes (Q2 time, abandonment rate, service level) if staffing requirements are adjusted by a certain percentage.",
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