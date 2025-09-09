import streamlit as st
import pandas as pd
from datetime import timedelta




def run_fte_analysis():

    st.title('Daywise Distribution Simulator')
    # File uploader
    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

    if uploaded_file is not None:
        sheet_name = 'Sheet1'
        data = pd.read_excel(uploaded_file, sheet_name=sheet_name, engine='openpyxl')
        st.success("File uploaded and data loaded successfully!")

        # Sidebar for user inputs
        st.sidebar.header('Simulator Parameters')
        start_date = st.sidebar.date_input('Start Date')
        end_date = start_date + timedelta(days=6)
        usd_global = st.sidebar.selectbox('USD/Global', data['USD'].unique())
        level = st.sidebar.selectbox('Level', data['Level'].unique())
        demand_increase = st.sidebar.number_input('Demand Increase (%)', min_value=0.0, value=5.0)

        ll_input1 = 0.9
        ul_input1 = 1.1
        ll_input2 = 0.9
        ul_input2 = 1.1

        # Filter data
        filtered_data = data[
            (data['startDate per day'] >= pd.to_datetime(start_date)) &
            (data['startDate per day'] <= pd.to_datetime(end_date)) &
            (data['USD'] == usd_global) &
            (data['Level'] == level)
        ]

        # Calculations
        weekly_demand = filtered_data['Demand'].sum()
        daily_demand = weekly_demand / 7

        if 'Calls' in filtered_data.columns and filtered_data['Calls'].sum() > 0:
            avg_q2_time = (filtered_data['Q2'] * filtered_data['Calls']).sum() / filtered_data['Calls'].sum()
            avg_occ_rate = (filtered_data['Occupancy Rate'] * filtered_data['Calls']).sum() / filtered_data['Calls'].sum()
            avg_abn_rate = (filtered_data['ABN %'] * filtered_data['Calls']).sum() / filtered_data['Calls'].sum()
        else:
            avg_q2_time = filtered_data['Q2'].mean()
            avg_occ_rate = filtered_data['Occupancy Rate'].mean()
            avg_abn_rate = filtered_data['ABN %'].mean()

        staffing_calc_for_ul_ll = filtered_data['Staffing'].mean()
        avg_staffing_max_for_week = filtered_data['Staffing'].max()

        # Staffing adjustment options
        st.sidebar.subheader("Staffing Adjustment Method")
        staffing_method = st.sidebar.radio("Choose method", ["Percentage", "Absolute"])
        if staffing_method == "Percentage":
            staffing_direction = st.sidebar.radio("Adjustment Type", ["Increase", "Decrease"])
            staffing_change_pct = st.sidebar.number_input("Staffing Change (%)", min_value=0.0, value=5.0)
            if staffing_direction == "Increase":
                adjusted_staffing = avg_staffing_max_for_week * (1 + staffing_change_pct / 100)
                adjusted_staffing1 = staffing_calc_for_ul_ll * (1 + staffing_change_pct / 100)
            else:
                adjusted_staffing = avg_staffing_max_for_week * (1 - staffing_change_pct / 100)
                adjusted_staffing1 = staffing_calc_for_ul_ll * (1 - staffing_change_pct / 100)
        else:
            staffing_direction = st.sidebar.radio("Adjustment Type", ["Increase", "Decrease"])
            staffing_change_abs = st.sidebar.number_input("Staffing Change (absolute)", min_value=0.0, value=1.0)
            if staffing_direction == "Increase":
                adjusted_staffing = avg_staffing_max_for_week + staffing_change_abs
                adjusted_staffing1 = staffing_calc_for_ul_ll + staffing_change_abs
            else:
                adjusted_staffing = max(0, avg_staffing_max_for_week - staffing_change_abs)
                adjusted_staffing1 = max(0, staffing_calc_for_ul_ll - staffing_change_abs)

        adjusted_demand = daily_demand * (1 + demand_increase / 100)

        filtered_limits = data[
            (data['Demand'] >= ll_input1 * adjusted_demand) &
            (data['Demand'] <= ul_input1 * adjusted_demand) &
            (data['Staffing'] >= ll_input2 * adjusted_staffing1) &
            (data['Staffing'] <= ul_input2 * adjusted_staffing1)
        ]

        num_rows = len(filtered_limits)

        if num_rows < 5:
            reliability_color = "red"
            reliability_text = "Low Reliability"
        elif 5 <= num_rows <= 10:
            reliability_color = "orange"
            reliability_text = "Moderate Reliability"
        else:
            reliability_color = "green"
            reliability_text = "High Reliability"

        if 'Calls' in filtered_limits.columns and filtered_limits['Calls'].sum() > 0:
            new_avg_q2_time = (filtered_limits['Q2'] * filtered_limits['Calls']).sum() / filtered_limits['Calls'].sum()
            new_avg_occ_rate = (filtered_limits['Occupancy Rate'] * filtered_limits['Calls']).sum() / filtered_limits['Calls'].sum()
            new_avg_abn_rate = (filtered_limits['ABN %'] * filtered_limits['Calls']).sum() / filtered_limits['Calls'].sum()
        else:
            new_avg_q2_time = filtered_limits['Q2'].mean()
            new_avg_occ_rate = filtered_limits['Occupancy Rate'].mean()
            new_avg_abn_rate = filtered_limits['ABN %'].mean()

            # Helper function
        def safe_int(value, default=0):
            return int(value) if pd.notna(value) else default

        def safe_float(value, default=0.0):
            return float(value) if pd.notna(value) else default

        # Display results
        st.write("### Simulation Results")

        st.metric("Weekly Demand", f"{safe_int(weekly_demand)}")
        st.metric("Daily Demand", f"{safe_int(daily_demand)}")
        st.metric("Adjusted Demand", f"{safe_int(adjusted_demand)}")

        st.metric("Average Q2 Time", f"{safe_float(avg_q2_time):.2f}")
        st.metric("Average Occupancy Rate", f"{safe_float(avg_occ_rate) * 100:.1f}%")
        st.metric("Average Abandon Rate", f"{safe_float(avg_abn_rate):.2f}%")

        st.metric("Average Staffing", f"{safe_int(avg_staffing_max_for_week)}")
        st.metric("Adjusted Staffing", f"{safe_int(adjusted_staffing)}")

        fte_base = 2250 * safe_float(filtered_data['Occ Assumption'].mean())
        fte_value = weekly_demand / fte_base if fte_base else 0
        st.metric("FTE Requirement", f"{safe_int(fte_value)}")

        st.markdown(
            f"<div style='padding:10px; background-color:{reliability_color}; color:white; border-radius:5px;'>"
            f"<strong>Reliability Indicator:</strong> {reliability_text}</div>",
            unsafe_allow_html=True
        )
        st.metric("New Average Q2 Time", f"{safe_float(new_avg_q2_time):.2f}")
        st.metric("New Average Occupancy Rate", f"{safe_float(new_avg_occ_rate) * 100:.1f}%")
        st.metric("New Average Abandon Rate", f"{safe_float(new_avg_abn_rate):.2f}%")

    else:
        st.warning("Please upload an Excel file to proceed.")



