import streamlit as st
import pandas as pd
from datetime import timedelta
from datetime import datetime

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
        st.sidebar.header('Common Filters - for both Simulator and Report Generation Sections -->')
        # Date input with default value set to latest Sunday
        start_date = st.sidebar.date_input('Start Date', value=latest_sunday if latest_sunday else None)    
        end_date = start_date + timedelta(days=6)
        
        usd_options = data['USD'].unique()
        level_options = data['Level'].unique()
        
        default_usd_global_index = list(usd_options).index("Combined") if "Combined" in usd_options else 0
        default_level_index = list(level_options).index("Combined") if "Combined" in usd_options else 0
        
        usd_global = st.sidebar.selectbox('USD/Global', data['USD'].unique(), index=default_usd_global_index)
        level = st.sidebar.selectbox('Level', data['Level'].unique(), index=default_level_index)

        # New Change
        data["Staffing"] = data["Staffing"]*1.4

        # Rename columns safely
        data.rename(columns={ 
            'Met': 'Service Level',
            'Loaded AHT': 'AHT'
        }, inplace=True)
        
        
        st.subheader("Filters for Simulator-->")
        
        demand_change = st.number_input(
            'Demand Change (%) [Decrease (-ve) | Increase (+ve)]',
            min_value=-99.0,
            max_value=100.0,
            value=0.0,
            step=1.0
        )


        ll_input1 = 0.9
        ul_input1 = 1.1
        ll_input2 = 0.9
        ul_input2 = 1.1

        
        # Extract year from the date column
        data['Year'] = pd.to_datetime(data['startDate per day']).dt.year


        # Filter data
        data = data[
            (data['USD'] == usd_global) &
            (data['Level'] == level)
        ]

        filtered_data = data[
            (data['startDate per day'] >= pd.to_datetime(start_date)) &
            (data['startDate per day'] <= pd.to_datetime(end_date))
        ]

        # Calculations
        weekly_demand = filtered_data['Demand'].sum()
        daily_demand = weekly_demand / 7

        if 'Calls' in filtered_data.columns and filtered_data['Calls'].sum() > 0:
            avg_q2_time = (filtered_data['Q2'] * filtered_data['Calls']).sum() / filtered_data['Calls'].sum()
            avg_occ_rate = (filtered_data['Occupancy Rate'] * filtered_data['Calls']).sum() / filtered_data['Calls'].sum()
            avg_abn_rate = (filtered_data['ABN %'] * filtered_data['Calls']).sum() / filtered_data['Calls'].sum()
            avg_sl = (filtered_data['Service Level'] * filtered_data['Calls']).sum() / filtered_data['Calls'].sum()
        else:
            avg_q2_time = filtered_data['Q2'].mean()
            avg_occ_rate = filtered_data['Occupancy Rate'].mean()
            avg_abn_rate = filtered_data['ABN %'].mean()
            avg_sl = filtered_data['Service Level'].mean()

        staffing_calc_for_ul_ll = filtered_data['Staffing'].mean()

        avg_staffing_max_for_week = filtered_data['Staffing'].mean()

        # Staffing adjustment options- New Changes
        st.write("Staffing Adjustment Method-")
        cols = st.columns(3)
        staffing_method = cols[0].radio("Choose method", ["Percentage", "Absolute"])
        if staffing_method == "Percentage":
            staffing_direction = cols[1].radio("Adjustment Type", ["Increase", "Decrease"])
            staffing_change_pct = cols[2].number_input("Staffing Change (%)", min_value=0.0, value=5.0)
            if staffing_direction == "Increase":
                adjusted_staffing = avg_staffing_max_for_week * (1 + staffing_change_pct / 100)
                adjusted_staffing1 = staffing_calc_for_ul_ll * (1 + staffing_change_pct / 100)
            else:
                adjusted_staffing = avg_staffing_max_for_week * (1 - staffing_change_pct / 100)
                adjusted_staffing1 = staffing_calc_for_ul_ll * (1 - staffing_change_pct / 100)
        else:
            staffing_direction = st.radio("Adjustment Type", ["Increase", "Decrease"])
            staffing_change_abs = st.number_input("Staffing Change (absolute)", min_value=0.0, value=1.0)
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

        # Get the current year
        current_year = datetime.now().year

        # Define weights
        weights = {
            current_year: 0.6,
            current_year - 1: 0.2,
            current_year - 2: 0.2
        }

        # Map weights to the DataFrame
        filtered_limits['Weight'] = filtered_limits['Year'].map(weights)
        filtered_OrigDemand_OrigStaff['Weight'] = filtered_OrigDemand_OrigStaff['Year'].map(weights)


        weighted_q2_orig = (filtered_OrigDemand_OrigStaff['Q2'] * filtered_OrigDemand_OrigStaff['Weight']).sum() / filtered_OrigDemand_OrigStaff['Weight'].sum()
        weighted_occr_orig = (filtered_OrigDemand_OrigStaff['Occupancy Rate'] * filtered_OrigDemand_OrigStaff['Weight']).sum() / filtered_OrigDemand_OrigStaff['Weight'].sum()
        weighted_abn_orig = (filtered_OrigDemand_OrigStaff['ABN %'] * filtered_OrigDemand_OrigStaff['Weight']).sum() / filtered_OrigDemand_OrigStaff['Weight'].sum()
        weighted_sl_orig = (filtered_OrigDemand_OrigStaff['Service Level'] * filtered_OrigDemand_OrigStaff['Weight']).sum() / filtered_OrigDemand_OrigStaff['Weight'].sum()


        weighted_q2 = (filtered_limits['Q2'] * filtered_limits['Weight']).sum() / filtered_limits['Weight'].sum()
        weighted_occr = (filtered_limits['Occupancy Rate'] * filtered_limits['Weight']).sum() / filtered_limits['Weight'].sum()
        weighted_abn = (filtered_limits['ABN %'] * filtered_limits['Weight']).sum() / filtered_limits['Weight'].sum()
        weighted_sl = (filtered_limits['Service Level'] * filtered_limits['Weight']).sum() / filtered_limits['Weight'].sum()

        
        delta_chg_q2 = (weighted_q2 - weighted_q2_orig)/weighted_q2_orig
        delta_chg_or = (weighted_occr - weighted_occr_orig)/weighted_occr_orig
        delta_chg_abn = (weighted_abn - weighted_abn_orig)/weighted_abn_orig
        delta_chg_sl = (weighted_sl - weighted_sl_orig)/weighted_sl_orig


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
        new_avg_occ_rate = min(avg_occ_rate * (1 + delta_chg_or),1)

        new_avg_abn_rate = min(avg_abn_rate * (1 + delta_chg_abn),100)
        new_avg_sl = min(avg_sl * (1 + delta_chg_sl), 1)

        # Display results
        st.write("### Simulation Results")
        # Create columns
        cols = st.columns(3)

        # First row
        cols[0].metric("Weekly Demand", f"{int(weekly_demand)}")
        cols[1].metric("Daily Demand", f"{int(daily_demand)}")
        cols[2].metric("Adjusted Demand", f"{int(adjusted_demand)}")

        # Second row
        cols = st.columns(4)
        cols[0].metric("Average Q2 Time", f"{avg_q2_time:.2f}")
        cols[1].metric("Average Occupancy Rate", f"{avg_occ_rate * 100:.1f}%")
        cols[2].metric("Average Abandon Rate", f"{avg_abn_rate:.2f}%")
        cols[3].metric("Average Service Level", f"{avg_sl*100:.2f}%")

        # Third row
        cols = st.columns(3)
        cols[0].metric("Average Staffing", f"{int(avg_staffing_max_for_week)}")
        cols[1].metric("Adjusted Staffing", f"{int(adjusted_staffing)}")
        cols[2].metric("FTE Requirement", f"{int(weekly_demand/(2250 * filtered_data['Occ Assumption'].mean()))}")         

        # Display results

        st.markdown(
            f"<div style='padding:10px; background-color:{reliability_color}; color:white; border-radius:5px;'>"
            f"<strong>Reliability Indicator:</strong> {reliability_text}</div>",
            unsafe_allow_html=True
        )

        cols = st.columns(4)
        cols[0].metric("New Avg Q2 Time", f"{new_avg_q2_time:.2f}")
        cols[1].metric("New Avg Occupancy Rate", f"{new_avg_occ_rate*100:.1f}%")
        
        # cols = st.columns(2)
        cols[2].metric("New Avg Abandon Rate", f"{new_avg_abn_rate:.2f}%")
        cols[3].metric("New Avg Service Level", f"{new_avg_sl*100:.2f}%")
        
        #Report section
        
        # Divider line
        st.markdown("---")
        # Create a visual boundary using a container
        with st.container():   
            # Main title
            st.title("Report Generation- Section")

            st.subheader("Filters Only for Report Generation-->")
            
            simulation_mode = st.radio("Select Simulation Mode", ["Single Variable", "Double Variable"])

            # Define change percentages
            change_values = [1, 2, 5, 10, 15, 20, 25]

            # Prepare results list
            results = []
            
            if simulation_mode == "Single Variable":
                variable_to_change = st.radio("Variable to Change", ["Demand", "Staffing"])
                if variable_to_change == "Demand":
                    inc_dec_demand = st.radio("Select Increase or Decrease (Demand)", ["Increase", "Decrease"])
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
        
                        # Get the current year
                        current_year = datetime.now().year

                        # Define weights
                        weights = {
                            current_year: 0.6,
                            current_year - 1: 0.2,
                            current_year - 2: 0.2
                        }

                        # Map weights to the DataFrame
                        filtered_limits['Weight'] = filtered_limits['Year'].map(weights)
                        filtered_OrigDemand_OrigStaff['Weight'] = filtered_OrigDemand_OrigStaff['Year'].map(weights)


                        weighted_q2_orig = (filtered_OrigDemand_OrigStaff['Q2'] * filtered_OrigDemand_OrigStaff['Weight']).sum() / filtered_OrigDemand_OrigStaff['Weight'].sum()
                        weighted_occr_orig = (filtered_OrigDemand_OrigStaff['Occupancy Rate'] * filtered_OrigDemand_OrigStaff['Weight']).sum() / filtered_OrigDemand_OrigStaff['Weight'].sum()
                        weighted_abn_orig = (filtered_OrigDemand_OrigStaff['ABN %'] * filtered_OrigDemand_OrigStaff['Weight']).sum() / filtered_OrigDemand_OrigStaff['Weight'].sum()
                        weighted_sl_orig = (filtered_OrigDemand_OrigStaff['Service Level'] * filtered_OrigDemand_OrigStaff['Weight']).sum() / filtered_OrigDemand_OrigStaff['Weight'].sum()

                        weighted_q2 = (filtered_limits['Q2'] * filtered_limits['Weight']).sum() / filtered_limits['Weight'].sum()
                        weighted_occr = (filtered_limits['Occupancy Rate'] * filtered_limits['Weight']).sum() / filtered_limits['Weight'].sum()
                        weighted_abn = (filtered_limits['ABN %'] * filtered_limits['Weight']).sum() / filtered_limits['Weight'].sum()
                        weighted_sl = (filtered_limits['Service Level'] * filtered_limits['Weight']).sum() / filtered_limits['Weight'].sum()
        
                        delta_chg_q2 = (weighted_q2 - weighted_q2_orig)/weighted_q2_orig
                        delta_chg_or = (weighted_occr - weighted_occr_orig)/weighted_occr_orig
                        delta_chg_abn = (weighted_abn - weighted_abn_orig)/weighted_abn_orig
                        delta_chg_sl = (weighted_sl - weighted_sl_orig)/weighted_sl_orig 
                        
                        new_avg_q2_time = avg_q2_time * (1 + delta_chg_q2)
                        new_avg_occ_rate = min(avg_occ_rate * (1 + delta_chg_or),1)
                        new_avg_abn_rate = min(avg_abn_rate * (1 + delta_chg_abn),1)
                        new_avg_sl = min(avg_sl * (1 + delta_chg_sl), 1)
                    
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
                            "New Q2 Time": round(new_avg_q2_time, 2),
                            "Average Occupancy Rate": f"{avg_occ_rate * 100:.1f}%",
                            "New Occupancy Rate": f"{round(new_avg_occ_rate*100, 2)}%",
                            "Average Abandon Rate": f"{avg_abn_rate:.2f}%",
                            "New Abandon Rate": f"{round(new_avg_abn_rate, 2)}%",
                            "Average Service Level": f"{avg_sl*100:.2f}%",
                            "New Service Level": f"{round(new_avg_sl*100, 2)}%"                            

                        })
                else:
                    inc_dec_staffing = st.radio("Select Increase or Decrease (Staffing) ", ["Increase", "Decrease"])
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
        
                        # Get the current year
                        current_year = datetime.now().year

                        # Define weights
                        weights = {
                            current_year: 0.6,
                            current_year - 1: 0.2,
                            current_year - 2: 0.2
                        }

                        # Map weights to the DataFrame
                        filtered_limits['Weight'] = filtered_limits['Year'].map(weights)
                        filtered_OrigDemand_OrigStaff['Weight'] = filtered_OrigDemand_OrigStaff['Year'].map(weights)


                        weighted_q2_orig = (filtered_OrigDemand_OrigStaff['Q2'] * filtered_OrigDemand_OrigStaff['Weight']).sum() / filtered_OrigDemand_OrigStaff['Weight'].sum()
                        weighted_occr_orig = (filtered_OrigDemand_OrigStaff['Occupancy Rate'] * filtered_OrigDemand_OrigStaff['Weight']).sum() / filtered_OrigDemand_OrigStaff['Weight'].sum()
                        weighted_abn_orig = (filtered_OrigDemand_OrigStaff['ABN %'] * filtered_OrigDemand_OrigStaff['Weight']).sum() / filtered_OrigDemand_OrigStaff['Weight'].sum()
                        weighted_sl_orig = (filtered_OrigDemand_OrigStaff['Service Level'] * filtered_OrigDemand_OrigStaff['Weight']).sum() / filtered_OrigDemand_OrigStaff['Weight'].sum()

                        weighted_q2 = (filtered_limits['Q2'] * filtered_limits['Weight']).sum() / filtered_limits['Weight'].sum()
                        weighted_occr = (filtered_limits['Occupancy Rate'] * filtered_limits['Weight']).sum() / filtered_limits['Weight'].sum()
                        weighted_abn = (filtered_limits['ABN %'] * filtered_limits['Weight']).sum() / filtered_limits['Weight'].sum()
                        weighted_sl = (filtered_limits['Service Level'] * filtered_limits['Weight']).sum() / filtered_limits['Weight'].sum()
        
                        delta_chg_q2 = (weighted_q2 - weighted_q2_orig)/weighted_q2_orig
                        delta_chg_or = (weighted_occr - weighted_occr_orig)/weighted_occr_orig
                        delta_chg_abn = (weighted_abn - weighted_abn_orig)/weighted_abn_orig
                        delta_chg_sl = (weighted_sl - weighted_sl_orig)/weighted_sl_orig 
                        
                        new_avg_q2_time = avg_q2_time * (1 + delta_chg_q2)
                        new_avg_occ_rate = min(avg_occ_rate * (1 + delta_chg_or),1)
                        new_avg_abn_rate = min(avg_abn_rate * (1 + delta_chg_abn),1)
                        new_avg_sl = min(avg_sl * (1 + delta_chg_sl), 1)

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
                            "New Q2 Time": round(new_avg_q2_time, 2),
                            "Average Occupancy Rate": f"{avg_occ_rate * 100:.1f}%",
                            "New Occupancy Rate": f"{round(new_avg_occ_rate*100, 2)}%",
                            "Average Abandon Rate": f"{avg_abn_rate:.2f}%",
                            "New Abandon Rate": f"{round(new_avg_abn_rate, 2)}%",
                            "Average Service Level": f"{avg_sl*100:.2f}%",
                            "New Service Level": f"{round(new_avg_sl*100, 2)}%"                              

                        })

            else:  # Double Variable
                cols = st.columns(2)
                inc_dec_demand = cols[0].radio("Select Increase or Decrease [Demand]", ["Increase", "Decrease"])
                inc_dec_staffing = cols[1].radio("Select Increase or Decrease [Staffing]", ["Increase", "Decrease"])
                
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
        
                        # Get the current year
                        current_year = datetime.now().year

                        # Define weights
                        weights = {
                            current_year: 0.6,
                            current_year - 1: 0.2,
                            current_year - 2: 0.2
                        }

                        # Map weights to the DataFrame
                        filtered_limits['Weight'] = filtered_limits['Year'].map(weights)
                        filtered_OrigDemand_OrigStaff['Weight'] = filtered_OrigDemand_OrigStaff['Year'].map(weights)


                        weighted_q2_orig = (filtered_OrigDemand_OrigStaff['Q2'] * filtered_OrigDemand_OrigStaff['Weight']).sum() / filtered_OrigDemand_OrigStaff['Weight'].sum()
                        weighted_occr_orig = (filtered_OrigDemand_OrigStaff['Occupancy Rate'] * filtered_OrigDemand_OrigStaff['Weight']).sum() / filtered_OrigDemand_OrigStaff['Weight'].sum()
                        weighted_abn_orig = (filtered_OrigDemand_OrigStaff['ABN %'] * filtered_OrigDemand_OrigStaff['Weight']).sum() / filtered_OrigDemand_OrigStaff['Weight'].sum()
                        weighted_sl_orig = (filtered_OrigDemand_OrigStaff['Service Level'] * filtered_OrigDemand_OrigStaff['Weight']).sum() / filtered_OrigDemand_OrigStaff['Weight'].sum()

                        weighted_q2 = (filtered_limits['Q2'] * filtered_limits['Weight']).sum() / filtered_limits['Weight'].sum()
                        weighted_occr = (filtered_limits['Occupancy Rate'] * filtered_limits['Weight']).sum() / filtered_limits['Weight'].sum()
                        weighted_abn = (filtered_limits['ABN %'] * filtered_limits['Weight']).sum() / filtered_limits['Weight'].sum()
                        weighted_sl = (filtered_limits['Service Level'] * filtered_limits['Weight']).sum() / filtered_limits['Weight'].sum()
        
                        delta_chg_q2 = (weighted_q2 - weighted_q2_orig)/weighted_q2_orig
                        delta_chg_or = (weighted_occr - weighted_occr_orig)/weighted_occr_orig
                        delta_chg_abn = (weighted_abn - weighted_abn_orig)/weighted_abn_orig
                        delta_chg_sl = (weighted_sl - weighted_sl_orig)/weighted_sl_orig 
                        
                        new_avg_q2_time = avg_q2_time * (1 + delta_chg_q2)
                        new_avg_occ_rate = min(avg_occ_rate * (1 + delta_chg_or),1)
                        new_avg_abn_rate = min(avg_abn_rate * (1 + delta_chg_abn),1)
                        new_avg_sl = min(avg_sl * (1 + delta_chg_sl), 1)
                    
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
                            "New Q2 Time": round(new_avg_q2_time, 2),
                            "Average Occupancy Rate": f"{avg_occ_rate * 100:.1f}%",
                            "New Occupancy Rate": f"{round(new_avg_occ_rate*100, 2)}%",
                            "Average Abandon Rate": f"{avg_abn_rate:.2f}%",
                            "New Abandon Rate": f"{round(new_avg_abn_rate, 2)}%",
                            "Average Service Level": f"{avg_sl*100:.2f}%",
                            "New Service Level": f"{round(new_avg_sl*100, 2)}%"                            

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


if __name__ == "__main__":
    run_fte_analysis_4()
