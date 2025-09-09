import streamlit as st
import pandas as pd
from datetime import timedelta
def run_fte_analysis_4():
    # File uploader
    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

    if uploaded_file is not None:
        FileName = uploaded_file.name[:3]
    else:
        st.warning("Please upload a file to extract its name.")


    st.title('Daywise Distribution Simulator')

    if uploaded_file is not None:
        sheet_name = 'Sheet1'
        data = pd.read_excel(uploaded_file, sheet_name=sheet_name, engine='openpyxl')
        st.success("File uploaded and data loaded successfully!")

        # Check and convert the date column
        if 'startDate per day' in data.columns:
            data['startDate per day'] = pd.to_datetime(data['startDate per day'], errors='coerce')

            # Filter rows where the date is a Sunday (weekday == 6)
            sundays = data[data['startDate per day'].dt.weekday == 6]

            # Get the latest Sunday
            latest_sunday = sundays['startDate per day'].max().date() if not sundays.empty else None
        else:
            latest_sunday = None
    

        # Sidebar for user inputs
        st.sidebar.header('Simulator Parameters')
        # Date input with default value set to latest Sunday
        start_date = st.sidebar.date_input('Start Date', value=latest_sunday if latest_sunday else None)    
        end_date = start_date + timedelta(days=6)
        
        
        usd_options = data['USD'].unique()
        level_options = data['Level'].unique()
        
        default_usd_global_index = list(usd_options).index("Combined") if "Combined" in usd_options else 0
        default_level_index = list(level_options).index("Combined") if "Combined" in usd_options else 0
        
        usd_global = st.sidebar.selectbox('USD/Global', data['USD'].unique(), index=default_usd_global_index)
        level = st.sidebar.selectbox('Level', data['Level'].unique(), index=default_level_index)
        
        
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

        # Staffing adjustment options- New Changes
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
            st.error(f"An error occurred while calculating adjusted demand")

            
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
        
        delta_chg_q2 = (filtered_limits['Q2'].mean()-filtered_OrigDemand_OrigStaff['Q2'].mean())/filtered_OrigDemand_OrigStaff['Q2'].mean()
        delta_chg_or = (filtered_limits['Occupancy Rate'].mean()-filtered_OrigDemand_OrigStaff['Occupancy Rate'].mean())/filtered_OrigDemand_OrigStaff['Occupancy Rate'].mean()
        delta_chg_abn = (filtered_limits['ABN %'].mean()-filtered_OrigDemand_OrigStaff['ABN %'].mean())/filtered_OrigDemand_OrigStaff['ABN %'].mean() 

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
        

    #     if 'Calls' in filtered_limits.columns and filtered_limits['Calls'].sum() > 0:
    #         new_avg_q2_time = (filtered_limits['Q2'] * filtered_limits['Calls']).sum() / filtered_limits['Calls'].sum()
    #         new_avg_occ_rate = (filtered_limits['Occupancy Rate'] * filtered_limits['Calls']).sum() / filtered_limits['Calls'].sum()
    #         new_avg_abn_rate = (filtered_limits['ABN %'] * filtered_limits['Calls']).sum() / filtered_limits['Calls'].sum()
    #     else:
    #         new_avg_q2_time = filtered_limits['Q2'].mean()
    #         new_avg_occ_rate = filtered_limits['Occupancy Rate'].mean()
    #         new_avg_abn_rate = filtered_limits['ABN %'].mean()

        # Display results
        st.write("### Simulation Results")
        st.metric("Weekly Demand", f"{int(weekly_demand)}")
        st.metric("Daily Demand", f"{int(daily_demand)}")
        st.metric("Adjusted Demand", f"{int(adjusted_demand)}")
        st.metric("Average Q2 Time", f"{avg_q2_time:.2f}")
        st.metric("Average Occupancy Rate", f"{avg_occ_rate * 100:.1f}%")
        st.metric("Average Abandon Rate", f"{avg_abn_rate:.2f}%")
        st.metric("Average Staffing", f"{int(avg_staffing_max_for_week)}")
        st.metric("Adjusted Staffing", f"{int(adjusted_staffing)}")
        st.metric("FTE Requirement", f"{int(weekly_demand/(2250 * filtered_data['Occ Assumption'].mean()))}")

        st.markdown(
            f"<div style='padding:10px; background-color:{reliability_color}; color:white; border-radius:5px;'>"
            f"<strong>Reliability Indicator:</strong> {reliability_text}</div>",
            unsafe_allow_html=True
        )
        st.metric("New Average Q2 Time", f"{new_avg_q2_time:.2f}")
        st.metric("New Average Occupancy Rate", f"{new_avg_occ_rate:.1f}%")
        st.metric("New Average Abandon Rate", f"{new_avg_abn_rate:.2f}%")
        
        
        #Report section
        
        
        # Divider line
        st.markdown("---")
        
        # Main title
        st.title("Report Generation- Section")

        # Sidebar inputs
        st.sidebar.markdown("---")
        st.sidebar.header("Simulation Settings- Report Generation Section")
        simulation_mode = st.sidebar.radio("Select Simulation Mode", ["Single Variable", "Double Variable"])

        # Define change percentages
        change_values = [1, 2, 5, 10, 15, 20, 25]

        # Prepare results list
        results = []

        if simulation_mode == "Single Variable":
            variable_to_change = st.sidebar.radio("Variable to Change", ["Demand", "Staffing"])
            if variable_to_change == "Demand":
                inc_dec_demand = st.sidebar.radio("Select Increase or Decrease (Demand)", ["Increase", "Decrease"])
                for dc in change_values:
                    if inc_dec_demand == "Increase":
                        new_demand = daily_demand * (1 + dc / 100)
                    else:
                        new_demand = daily_demand * (1 - dc / 100)
                        
                    new_staffing = avg_staffing_max_for_week
                    
                    filtered_OrigDemand_OrigStaff = data[
                        (data['Demand'] >= ll_input1 * daily_demand) &
                        (data['Demand'] <= ul_input1 * daily_demand) &
                        (data['Staffing'] >= ll_input2 * staffing_calc_for_ul_ll) &
                        (data['Staffing'] <= ul_input2 * staffing_calc_for_ul_ll)
                    ]
            
            
                    filtered_limits = data[
                        (data['Demand'] >= ll_input1 * new_demand) &
                        (data['Demand'] <= ul_input1 * new_demand) &
                        (data['Staffing'] >= ll_input2 * staffing_calc_for_ul_ll) &
                        (data['Staffing'] <= ul_input2 * staffing_calc_for_ul_ll)
                    ]
        
                    delta_chg_q2 = (filtered_limits['Q2'].mean() - filtered_OrigDemand_OrigStaff['Q2'].mean())/filtered_OrigDemand_OrigStaff['Q2'].mean()
                    delta_chg_or = (filtered_limits['Occupancy Rate'].mean()-filtered_OrigDemand_OrigStaff['Occupancy Rate'].mean())/filtered_OrigDemand_OrigStaff['Occupancy Rate'].mean()
                    delta_chg_abn = (filtered_limits['ABN %'].mean()-filtered_OrigDemand_OrigStaff['ABN %'].mean())/filtered_OrigDemand_OrigStaff['ABN %'].mean()
                    
                    new_q2 = avg_q2_time * (1 + delta_chg_q2)
                    new_occ = avg_occ_rate * (1 + delta_chg_or)
                    new_abn = avg_abn_rate * (1 + delta_chg_abn) 
                    
                    results.append({
                        "Scenario 1": f"Demand {inc_dec_demand} By {abs(dc)}%",
                        "Scenario 2": "No Change",
                        "Language": FileName,
                        "USD/ GLOBAL": usd_global,
                        "Level": level,
                        "FTE Requirement":  f"{int(weekly_demand/(2250 * filtered_data['Occ Assumption'].mean()))}",                    
                        "Demand Daily": f"{int(daily_demand)}",
                        "New Demand (Daily)": round(new_demand),
                        "Staffing": f"{int(avg_staffing_max_for_week)}",
                        "Adjusted/New Staffing": round(new_staffing),
                        "Average Q2 Time": f"{avg_q2_time:.2f}",
                        "New Q2 Time": round(new_q2, 2),
                        "Average Occupancy Rate": f"{avg_occ_rate * 100:.1f}%",
                        "New Occupancy Rate": f"{round(new_occ*100, 2)}%",
                        "Average Abandon Rate": f"{avg_abn_rate:.2f}%",
                        "New Abandon Rate": f"{round(new_abn, 2)}%"

                    })
            else:
                inc_dec_staffing = st.sidebar.radio("Select Increase or Decrease (Staffing) ", ["Increase", "Decrease"])
                for sc in change_values:
                    if inc_dec_staffing == "Increase":
                        new_staffing = avg_staffing_max_for_week * (1 + sc / 100)
                        new_staffing1 = staffing_calc_for_ul_ll * (1 + sc / 100)  
                    else:
                        new_staffing = avg_staffing_max_for_week * (1 - sc / 100)
                        new_staffing1 = staffing_calc_for_ul_ll * (1 - sc / 100)
                        
                    new_demand = daily_demand
                    
                    filtered_OrigDemand_OrigStaff = data[
                        (data['Demand'] >= ll_input1 * daily_demand) &
                        (data['Demand'] <= ul_input1 * daily_demand) &
                        (data['Staffing'] >= ll_input2 * staffing_calc_for_ul_ll) &
                        (data['Staffing'] <= ul_input2 * staffing_calc_for_ul_ll)
                    ]
            
            
                    filtered_limits = data[
                        (data['Demand'] >= ll_input1 * new_demand) &
                        (data['Demand'] <= ul_input1 * new_demand) &
                        (data['Staffing'] >= ll_input2 * new_staffing1) &
                        (data['Staffing'] <= ul_input2 * new_staffing1)
                    ]
        
                    delta_chg_q2 = (filtered_limits['Q2'].mean() - filtered_OrigDemand_OrigStaff['Q2'].mean())/filtered_OrigDemand_OrigStaff['Q2'].mean()
                    delta_chg_or = (filtered_limits['Occupancy Rate'].mean()-filtered_OrigDemand_OrigStaff['Occupancy Rate'].mean())/filtered_OrigDemand_OrigStaff['Occupancy Rate'].mean()
                    delta_chg_abn = (filtered_limits['ABN %'].mean()-filtered_OrigDemand_OrigStaff['ABN %'].mean())/filtered_OrigDemand_OrigStaff['ABN %'].mean()
                    
                    new_q2 = avg_q2_time * (1 + delta_chg_q2)
                    new_occ = avg_occ_rate * (1 + delta_chg_or)
                    new_abn = avg_abn_rate * (1 + delta_chg_abn) 
                    results.append({
                        "Scenario 1": "No Change",
                        "Scenario 2": f"Staffing {inc_dec_staffing} By {abs(sc)}%",
                        "Language": FileName,
                        "USD/ GLOBAL": usd_global,
                        "Level": level,
                        "FTE Requirement":  f"{int(weekly_demand/(2250 * filtered_data['Occ Assumption'].mean()))}",                    
                        "Demand Daily": f"{int(daily_demand)}",
                        "New Demand (Daily)": round(new_demand),
                        "Staffing": f"{int(avg_staffing_max_for_week)}",
                        "Adjusted/New Staffing": round(new_staffing),
                        "Average Q2 Time": f"{avg_q2_time:.2f}",
                        "New Q2 Time": round(new_q2, 2),
                        "Average Occupancy Rate": f"{avg_occ_rate * 100:.1f}%",
                        "New Occupancy Rate": f"{round(new_occ*100, 2)}%",
                        "Average Abandon Rate": f"{avg_abn_rate:.2f}%",
                        "New Abandon Rate": f"{round(new_abn, 2)}%"
                    })

        else:  # Double Variable
            inc_dec_staffing = st.sidebar.radio("Select Increase or Decrease [Staffing]", ["Increase", "Decrease"])
            inc_dec_demand = st.sidebar.radio("Select Increase or Decrease [Demand]", ["Increase", "Decrease"])
            for dc in change_values:
                for sc in change_values:
                    if inc_dec_demand == "Increase":
                        new_demand = daily_demand * (1 + dc / 100)
                    else:
                        new_demand = daily_demand * (1 - dc / 100)
                        
                    if inc_dec_staffing == "Increase":
                        new_staffing = avg_staffing_max_for_week * (1 + sc / 100)
                        new_staffing1 = staffing_calc_for_ul_ll * (1 + sc / 100)  
                    else:
                        new_staffing = avg_staffing_max_for_week * (1 - sc / 100)
                        new_staffing1 = staffing_calc_for_ul_ll * (1 - sc / 100)             
                    
                    filtered_OrigDemand_OrigStaff = data[
                        (data['Demand'] >= ll_input1 * daily_demand) &
                        (data['Demand'] <= ul_input1 * daily_demand) &
                        (data['Staffing'] >= ll_input2 * staffing_calc_for_ul_ll) &
                        (data['Staffing'] <= ul_input2 * staffing_calc_for_ul_ll)
                    ]
            
            
                    filtered_limits = data[
                        (data['Demand'] >= ll_input1 * new_demand) &
                        (data['Demand'] <= ul_input1 * new_demand) &
                        (data['Staffing'] >= ll_input2 * new_staffing1) &
                        (data['Staffing'] <= ul_input2 * new_staffing1)
                    ]
        
                    delta_chg_q2 = (filtered_limits['Q2'].mean() - filtered_OrigDemand_OrigStaff['Q2'].mean())/filtered_OrigDemand_OrigStaff['Q2'].mean()
                    delta_chg_or = (filtered_limits['Occupancy Rate'].mean()-filtered_OrigDemand_OrigStaff['Occupancy Rate'].mean())/filtered_OrigDemand_OrigStaff['Occupancy Rate'].mean()
                    delta_chg_abn = (filtered_limits['ABN %'].mean()-filtered_OrigDemand_OrigStaff['ABN %'].mean())/filtered_OrigDemand_OrigStaff['ABN %'].mean()
                    
                    new_q2 = avg_q2_time * (1 + delta_chg_q2)
                    new_occ = avg_occ_rate * (1 + delta_chg_or)
                    new_abn = avg_abn_rate * (1 + delta_chg_abn) 

                    results.append({
                        "Scenario 1": f"Demand {inc_dec_demand} By {abs(dc)}%",
                        "Scenario 2": f"Staffing {inc_dec_staffing} By {abs(sc)}%",
                        "Language": FileName,
                        "USD/ GLOBAL": usd_global,
                        "Level": level,
                        "FTE Requirement":  f"{int(weekly_demand/(2250 * filtered_data['Occ Assumption'].mean()))}",                    
                        "Demand Daily": f"{int(daily_demand)}",
                        "New Demand (Daily)": round(new_demand),
                        "Staffing": f"{int(avg_staffing_max_for_week)}",
                        "Adjusted/New Staffing": round(new_staffing),
                        "Average Q2 Time": f"{avg_q2_time:.2f}",
                        "New Q2 Time": round(new_q2, 2),
                        "Average Occupancy Rate": f"{avg_occ_rate * 100:.1f}%",
                        "New Occupancy Rate": f"{round(new_occ*100, 2)}%",
                        "Average Abandon Rate": f"{avg_abn_rate:.2f}%",
                        "New Abandon Rate": f"{round(new_abn, 2)}%"
                    })

        # Display results
        result_df = pd.DataFrame(results)
        st.write("### Simulation Results")
        st.dataframe(result_df)

        # Download option
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Report as CSV",
            data=csv,
            file_name="scenario_report.csv",
            mime="text/csv"
        )


    else:
        st.warning("Please upload an Excel file to proceed.")
