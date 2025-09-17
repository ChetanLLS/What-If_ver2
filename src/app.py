import streamlit as st
from data_extractor import run_data_extractor
# from fte_analysis import run_fte_analysis
# from staffing_Req_Ver3 import run_fte_analysis
from default import main
from Datamart_Daywise_V4 import run_daywise_tool_ver4
from staffing_Req_Ver4 import run_fte_analysis_4


# App title
# st.set_page_config(page_title="What-if Analysis", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Datamart-Daywise", "Staffing Req Analysis", "What-if analysis","Data Extractor"])

# Main content based on selection
if page == "What-if analysis":
    st.title("What-If Analysis Tool")
    main()

elif page == "Data Extractor":
    st.title("Data Extractor")
    run_data_extractor()

elif page == "Staffing Req Analysis":
    st.title("Staffing Req Analysis")
    run_fte_analysis_4()

elif page == "Datamart-Daywise":
    st.title("Datamart Daywise")
    run_daywise_tool_ver4()