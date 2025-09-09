import streamlit as st
import pandas as pd
from io import BytesIO

# Function to process the uploaded file
def process_file(file, sheets):
    wp_df_combined = pd.DataFrame()
    for sheet in sheets:
        df = pd.read_excel(file, sheet_name=sheet, header=None)
        df = df.iloc[3:, 3:].reset_index(drop=True)
        df = df.T.reset_index(drop=True)
        
        df.columns = df.iloc[0].astype(str) + " " + df.iloc[1].astype(str) + " " + df.iloc[2].astype(str)
        df.columns = df.columns.str.replace("nan", "").str.strip()
        df = df[3:].reset_index(drop=True)
        df["Week Of:"] = pd.to_datetime(df["Week Of:"], errors="coerce").dt.date

        occ_columns = [col for col in df.columns if "OCC Assumptions" in col]
        volume_columns = [col for col in df.columns if "Volume (ACTUAL)" in col]
        staff_diff_columns = [col for col in df.columns if "STAFFING DIFFERENTIAL" in col]
        aht_columns = [col for col in df.columns if "AHT (ACTUAL)" in col]
        q4_columns = [col for col in df.columns if "Q4 Time" in col]
        q2_columns = [col for col in df.columns if "Q2 Time" in col]
        or_columns = [col for col in df.columns if "Occupancy Rate" in col]
        sr_columns = [col for col in df.columns if "Staffing Requirement (ACTUAL)" in col]
        staff_columns = [col for col in df.columns if "Staffing (ACTUAL/FORECASTED)" in col]
        sr_af_columns = [col for col in df.columns if "Staffing Requirement (ADJUSTED FORECAST)" in col]
        sr_adj_columns = [col for col in df.columns if "Staffing Requirement Adjustment" in col]
        sr_org_columns = [col for col in df.columns if "Staffing Requirement (ORIGINAL FORECAST)" in col]
        sr_var_columns = [col for col in df.columns if "Staffing Requirement (Variance)" in col]         

        combined_columns = occ_columns + volume_columns + staff_diff_columns + aht_columns + q4_columns + or_columns + sr_columns + staff_columns + q2_columns + sr_af_columns + sr_adj_columns + sr_org_columns + sr_var_columns
        wp_df = df[["Week Of:"] + combined_columns]
        wp_df_combined = pd.concat([wp_df_combined, wp_df], axis=1)

    return wp_df_combined, aht_columns, occ_columns

# Function to convert DataFrame to Excel and provide download link

def convert_df_to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Processed Data')
    writer.close()
    processed_data = output.getvalue()
    return processed_data


# Helper function to check if all columns exist
def columns_exist(df, columns):
    return int(any("L3" in col and col in df.columns for col in columns))

def add_derived_variables_not_csa_opi(df):
    # Volume Aggregations
    df["USD Combined Volume (ACTUAL)"] = df["USD L3 Volume (ACTUAL)"] + df["USD L4 Volume (ACTUAL)"] + df["USD L5 Volume (ACTUAL)"]
    df["Global Combined Volume (ACTUAL)"] = df["Global L3 Volume (ACTUAL)"] + df["Global L4 Volume (ACTUAL)"] + df["Global L5 Volume (ACTUAL)"]
    df["Combined Combined Volume (ACTUAL)"] = df["USD Combined Volume (ACTUAL)"] + df["Global Combined Volume (ACTUAL)"]
    
    df["Combined L3 Volume (ACTUAL)"] = df["USD L3 Volume (ACTUAL)"] + df["Global L3 Volume (ACTUAL)"]
    df["Combined L4 Volume (ACTUAL)"] = df["USD L4 Volume (ACTUAL)"] + df["Global L4 Volume (ACTUAL)"]
    df["Combined L5 Volume (ACTUAL)"] = df["USD L5 Volume (ACTUAL)"] + df["Global L5 Volume (ACTUAL)"]
    
    
    # AHT Calculations (Weighted Averages)
    df["Combined L3 AHT (ACTUAL)"] = (
        (df["USD L3 AHT (ACTUAL)"] * df["USD L3 Volume (ACTUAL)"]) +
        (df["Global L3 AHT (ACTUAL)"] * df["Global L3 Volume (ACTUAL)"])
    ) / (df["USD L3 Volume (ACTUAL)"] + df["Global L3 Volume (ACTUAL)"])

    df["Combined L4 AHT (ACTUAL)"] = (
        (df["USD L4 AHT (ACTUAL)"] * df["USD L4 Volume (ACTUAL)"]) +
        (df["Global L4 AHT (ACTUAL)"] * df["Global L4 Volume (ACTUAL)"])
    ) / (df["USD L4 Volume (ACTUAL)"] + df["Global L4 Volume (ACTUAL)"])

    df["Combined L5 AHT (ACTUAL)"] = (
        (df["USD L5 AHT (ACTUAL)"] * df["USD L5 Volume (ACTUAL)"]) +
        (df["Global L5 AHT (ACTUAL)"] * df["Global L5 Volume (ACTUAL)"])
    ) / (df["USD L5 Volume (ACTUAL)"] + df["Global L5 Volume (ACTUAL)"])
    
    df["USD Combined AHT (ACTUAL)"] = (
        (df["USD L3 AHT (ACTUAL)"] * df["USD L3 Volume (ACTUAL)"]) +
        (df["USD L4 AHT (ACTUAL)"] * df["USD L4 Volume (ACTUAL)"]) +
        (df["USD L5 AHT (ACTUAL)"] * df["USD L5 Volume (ACTUAL)"])
    ) / (df["USD L3 Volume (ACTUAL)"] + df["USD L4 Volume (ACTUAL)"] + df["USD L5 Volume (ACTUAL)"])
    
    df["Global Combined AHT (ACTUAL)"] = (
        (df["Global L3 AHT (ACTUAL)"] * df["Global L3 Volume (ACTUAL)"]) +
        (df["Global L4 AHT (ACTUAL)"] * df["Global L4 Volume (ACTUAL)"]) +
        (df["Global L5 AHT (ACTUAL)"] * df["Global L5 Volume (ACTUAL)"])
    ) / (df["Global L3 Volume (ACTUAL)"] + df["Global L4 Volume (ACTUAL)"] + df["Global L5 Volume (ACTUAL)"])    
    
    df["Combined Combined AHT (ACTUAL)"] = (
        (df["USD Combined AHT (ACTUAL)"] * df["USD Combined Volume (ACTUAL)"]) +
        (df["Global Combined AHT (ACTUAL)"] * df["Global Combined Volume (ACTUAL)"])
    ) / (df["USD Combined Volume (ACTUAL)"] + df["Global Combined Volume (ACTUAL)"])
    
    # Occ Assumption Calculations (Weighted Averages)
    df["USD OCC Assumptions (Combined):"] = (
        (df["USD OCC Assumptions (L3):"] * df["USD L3 Volume (ACTUAL)"]) + 
        (df["USD OCC Assumptions (L4):"] * df["USD L4 Volume (ACTUAL)"]) + 
        (df["USD OCC Assumptions (L5):"] * df["USD L5 Volume (ACTUAL)"])
        ) / (df["USD L3 Volume (ACTUAL)"] + df["USD L4 Volume (ACTUAL)"] +df["USD L5 Volume (ACTUAL)"])
            
    df["Global OCC Assumptions (Combined):"] = (
        (df["Global OCC Assumptions (L3):"]* df["Global L3 Volume (ACTUAL)"]) + 
        (df["Global OCC Assumptions (L4):"]* df["Global L4 Volume (ACTUAL)"]) + 
        (df["Global OCC Assumptions (L5):"]* df["Global L5 Volume (ACTUAL)"])
        ) / (df["Global L3 Volume (ACTUAL)"] + df["Global L4 Volume (ACTUAL)"] +df["Global L5 Volume (ACTUAL)"])

    df["Combined OCC Assumptions (Combined):"] = (
        (df["Global OCC Assumptions (L3):"]* df["Global L3 Volume (ACTUAL)"]) + 
        (df["Global OCC Assumptions (L4):"]* df["Global L4 Volume (ACTUAL)"]) + 
        (df["Global OCC Assumptions (L5):"]* df["Global L5 Volume (ACTUAL)"]) +
        (df["USD OCC Assumptions (L3):"]* df["USD L3 Volume (ACTUAL)"]) + 
        (df["USD OCC Assumptions (L4):"]* df["USD L4 Volume (ACTUAL)"]) + 
        (df["USD OCC Assumptions (L5):"]* df["USD L5 Volume (ACTUAL)"])
        ) / (df["Global L3 Volume (ACTUAL)"] + df["Global L4 Volume (ACTUAL)"] +df["Global L5 Volume (ACTUAL)"]+
           df["USD L3 Volume (ACTUAL)"] + df["USD L4 Volume (ACTUAL)"] +df["USD L5 Volume (ACTUAL)"])
            
    df["Combined OCC Assumptions (L3):"] = (
        (df["USD OCC Assumptions (L3):"] * df["USD L3 Volume (ACTUAL)"]) +
        (df["Global OCC Assumptions (L3):"] * df["Global L3 Volume (ACTUAL)"])
    ) / (df["USD L3 Volume (ACTUAL)"] + df["Global L3 Volume (ACTUAL)"])

    df["Combined OCC Assumptions (L4):"] = (
        (df["USD OCC Assumptions (L4):"] * df["USD L4 Volume (ACTUAL)"]) +
        (df["Global OCC Assumptions (L4):"] * df["Global L4 Volume (ACTUAL)"])
    ) / (df["USD L4 Volume (ACTUAL)"] + df["Global L4 Volume (ACTUAL)"])


    df["Combined OCC Assumptions (L5):"] = (
        (df["USD OCC Assumptions (L5):"] * df["USD L5 Volume (ACTUAL)"]) +
        (df["Global OCC Assumptions (L5):"] * df["Global L5 Volume (ACTUAL)"])
    ) / (df["USD L5 Volume (ACTUAL)"] + df["Global L5 Volume (ACTUAL)"])

    # Staffing Aggregations (ADJUSTED, ORIGINAL, ACTUAL, VARIANCE, ACTUAL/FORECASTED)
    for level in ["L3", "L4", "L5"]:
        for metric in ["Staffing Requirement (ADJUSTED FORECAST)", "Staffing Requirement Adjustment",
                       "Staffing Requirement (ORIGINAL FORECAST)", "Staffing Requirement (ACTUAL)",
                       "Staffing Requirement (Variance)", "Staffing (ACTUAL/FORECASTED)"]:
            usd_col = f"USD {level} {metric}"
            global_col = f"Global {level} {metric}"
            combined_col = f"Combined {level} {metric}"
            df[combined_col] = df[usd_col] + df[global_col]

    return df                                                  


# Main function to run the Streamlit app
def run_data_extractor():

    # Streamlit app
    st.title("Excel File Processor")

    # File uploader
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsm", "xlsx"])
    

    if uploaded_file:
        # planner_type selection
        planner_type = st.multiselect("Select sheets to process", ["1. Weekly Planner OPI USD", "2. Weekly Planner OPI Global", 
                                                             "3. Weekly Planner VRI", "1. Weekly Planner OPI", "2. Weekly Planner VRI", 
                                                                   "3. UKD"])
        if planner_type:
            # Process the file
            final_df, aht_columns, occ_columns = process_file(uploaded_file, planner_type)
            
            # Extract column names for dropdown
            file_name = uploaded_file.name
            if file_name.lower().startswith('csa'):
                
                #calculate AHT at combined combined level
                final_df["Combined Combined AHT (ACTUAL)"] = (
                    (final_df["Combined OLY AHT (ACTUAL)"] * final_df["Combined OLY Volume (ACTUAL)"]) +
                    (final_df["Combined BOA AHT (ACTUAL)"] * final_df["Combined BOA Volume (ACTUAL)"])
                ) / (final_df["Combined OLY Volume (ACTUAL)"] + final_df["Combined BOA Volume (ACTUAL)"])

                #calculate Occ Assumption at combined level
                final_df["Combined Combined OCC Assumptions:"] = (
                    (final_df["OCC Assumptions (OLY):"] * final_df["Combined OLY Volume (ACTUAL)"]) +
                    (final_df["OCC Assumptions (BOA):"] * final_df["Combined BOA Volume (ACTUAL)"])
                ) / (final_df["Combined OLY Volume (ACTUAL)"] + final_df["Combined BOA Volume (ACTUAL)"])

                #calculate Occupancy Rate at combined level
                final_df["Combined Combined Occupancy Rate"] = (
                    (final_df["OLY OLY Occupancy Rate"] * final_df["Combined OLY Volume (ACTUAL)"]) +
                    (final_df["BOA BOA Occupancy Rate"] * final_df["Combined BOA Volume (ACTUAL)"])
                ) / (final_df["Combined OLY Volume (ACTUAL)"] + final_df["Combined BOA Volume (ACTUAL)"])
                 #calculate Q4 Time at combined level
                final_df["Combined Combined Q4 Time"] = (
                    (final_df["OLY OLY Q4 Time"] * final_df["Combined OLY Volume (ACTUAL)"]) +
                    (final_df["BOA BOA Q4 Time"] * final_df["Combined BOA Volume (ACTUAL)"])
                ) / (final_df["Combined OLY Volume (ACTUAL)"] + final_df["Combined BOA Volume (ACTUAL)"])   

                volume_columns = [col for col in final_df.columns if "Volume (ACTUAL)" in col] 
                
                dropdown_options = []
                for col in volume_columns:
                    if "Combined" in col and "Volume (ACTUAL)" in col:
                        parts = col.replace("Volume (ACTUAL)", "").strip().split()
#                         st.write(parts)
                        if len(parts) >= 2:
                            dropdown_options.append(parts[1])  # Gets 'Combined', 'OLY', or 'BOA'

            else:
                vri_found = any("VRI" in item for item in planner_type)

                opi_usd_spa = any("1. Weekly Planner OPI USD" in item for item in planner_type)
                opi_global_spa = any("2. Weekly Planner OPI Global" in item for item in planner_type)
                
                if opi_usd_spa or opi_global_spa or vri_found:
                    st.write("")
                else:
                    final_df = add_derived_variables_not_csa_opi(final_df)                 
                    
                sr_columns = [col for col in final_df.columns if "Staffing Requirement (ACTUAL)" in col]                
                dropdown_options = [col.replace("Staffing Requirement (ACTUAL)", "").strip() for col in sr_columns]
            
            # User selection from dropdown
            selected_option = st.selectbox("Select the level", dropdown_options)
            
            if selected_option:

                if file_name.lower().startswith('csa'):
                    # Filter columns based on user selection
                    search_term = f"{selected_option} {selected_option}" if selected_option == "Combined" else selected_option
                    selected_columns = [col for col in final_df.columns if search_term in col]

                    final_df = final_df[["Week Of:"] + selected_columns]
                else:
                    
                    if any('vri' in item.lower() for item in planner_type):
                        selected_option_part = selected_option.split(' ', 1)
                        selected_columns = [col for col in final_df.columns if selected_option_part[1] in col]                    
                        final_df = final_df[["Week Of:"] + selected_columns]
                    else:
                        selected_option_part = selected_option.split(' ', 1)
                        if selected_option_part[0] != selected_option_part[1]:
                            selected_columns = [col for col in final_df.columns if selected_option_part[0] in col 
                                        and selected_option_part[1] in col]
                            final_df = final_df[["Week Of:"] + selected_columns]
                        else:
                            selected_columns = [col for col in final_df.columns if col.count("Combined") >= 2]                         
                            final_df = final_df[["Week Of:"] + selected_columns]
                                
                # Convert DataFrame to Excel
                processed_data = convert_df_to_excel(final_df)            
            
            # Display the final table
            st.write(final_df)
            
            # Convert DataFrame to Excel
            processed_data = convert_df_to_excel(final_df)
            
            planner_type_selected = [item.split("Weekly Planner", 1)[1].strip() for item in planner_type if "Weekly Planner" in item]

            # Provide download link

            st.download_button(
                label="Download data as Excel file",
                data=processed_data,
                file_name="processed_data_" + file_name.lower()[:3] + "_" + "_".join(planner_type_selected) + ".xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
           
             
