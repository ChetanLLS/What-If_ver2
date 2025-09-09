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

        demand_change = st.sidebar.number_input(
            'Demand Change (%)',
            min_value=-99.0,
            max_value=100.0,
            value=0.0,
            step=1.0
        )

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
        daily_demand = weekly_demand / 7 if pd.notna(weekly_demand) else float('nan')

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

        try:
            adjusted_demand = daily_demand * (1 + demand_change / 100)
            if adjusted_demand <= 0:
                st.error("Adjusted demand must be greater than zero. Please revise the demand change.")
            else:
                st.success(f"Adjusted Demand: {adjusted_demand:.2f}")
        except Exception as e:
            st.error("An error occurred while calculating adjusted demand.")

        filtered_OrigDemand_OrigStaff = data[
            (data['Demand'] >= ll_input1 * daily_demand) &
            (data['Demand'] <= ul_input1 * daily_demand) &
            (data['Staffing'] >= ll_input2 * staffing_calc_for_ul_ll) &
            (data['Staffing'] <= ul_input2 * staffing_calc_for_ul_ll)
        ]

        filtered_limits = data[
            (data['Demand'] >= ll_input1 * adjusted_demand) &
            (data['Demand'] <= ul_input1 * adjusted_demand) &
            (data['Staffing'] >= ll_input2 * adjusted_staffing1) &
            (data['Staffing'] <= ul_input2 * adjusted_staffing1)
        ]

        # Delta changes with safety
        def safe_delta(new, old):
            if pd.notna(old) and old != 0:
                return (new - old) / old
            return 0.0

        delta_chg_q2 = safe_delta(filtered_limits['Q2'].mean(), filtered_OrigDemand_OrigStaff['Q2'].mean())
        delta_chg_or = safe_delta(filtered_limits['Occupancy Rate'].mean(), filtered_OrigDemand_OrigStaff['Occupancy Rate'].mean())
        delta_chg_abn = safe_delta(filtered_limits['ABN %'].mean(), filtered_OrigDemand_OrigStaff['ABN %'].mean())

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

        new_avg_q2_time = avg_q2_time * (1 + delta_chg_q2)
        new_avg_occ_rate = avg_occ_rate * (1 + delta_chg_or)
        new_avg_abn_rate = avg_abn_rate * (1 + delta_chg_abn)

        # Display results
        st.write("### Simulation Results")

        st.metric("Weekly Demand", f"{int(weekly_demand)}" if pd.notna(weekly_demand) else "N/A")
        st.metric("Daily Demand", f"{int(daily_demand)}" if pd.notna(daily_demand) else "N/A")
        st.metric("Adjusted Demand", f"{int(adjusted_demand)}" if pd.notna(adjusted_demand) else "N/A")
        st.metric("Average Q2 Time", f"{avg_q2_time:.2f}" if pd.notna(avg_q2_time) else "N/A")
        st.metric("Average Occupancy Rate", f"{avg_occ_rate * 100:.1f}%" if pd.notna(avg_occ_rate) else "N/A")
        st.metric("Average Abandon Rate", f"{avg_abn_rate:.2f}%" if pd.notna(avg_abn_rate) else "N/A")
        st.metric("Average Staffing", f"{int(avg_staffing_max_for_week)}" if pd.notna(avg_staffing_max_for_week) else "N/A")
        st.metric("Adjusted Staffing", f"{int(adjusted_staffing)}" if pd.notna(adjusted_staffing) else "N/A")

        fte_req = weekly_demand / (2250 * filtered_data['Occ Assumption'].mean()) if 'Occ Assumption' in filtered_data.columns else float('nan')
        st.metric("FTE Requirement", f"{int(fte_req)}" if pd.notna(fte_req) else "N/A")

        st.markdown(
            f"<div style='padding:10px; background-color:{reliability_color}; color:white; border-radius:5px;'>"
            f"<strong>Reliability Indicator:</strong> {reliability_text}</div>",
            unsafe_allow_html=True
        )

        st.metric("New Average Q2 Time", f"{new_avg_q2_time:.2f}" if pd.notna(new_avg_q2_time) else "N/A")
        st.metric("New Average Occupancy Rate", f"{new_avg_occ_rate * 100:.1f}%" if pd.notna(new_avg_occ_rate) else "N/A")
        st.metric("New Average Abandon Rate", f"{new_avg_abn_rate:.2f}%" if pd.notna(new_avg_abn_rate) else "N/A")

    else:
        st.warning("Please upload an Excel file to proceed.")

if __name__ == "__main__":
    run_fte_analysis()