# streamlit_app_hourly.py

import streamlit as st
import pandas as pd
import numpy as np
import os
from io import BytesIO
from datetime import timedelta
import warnings
from dateutil import parser

warnings.filterwarnings("ignore")

# -----------------------------
# File Combination Logic
# -----------------------------
# Function to convert DataFrame to Excel

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    output.seek(0)
    return output.read()

def fte_combined_level_calc(df):
    # Rename and convert date column
    df.rename(columns={"startDate per hour": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')

    # Group and pivot the data
    grouped = df.groupby(["Date", "Language", "USD", "Level"], as_index=False)[["Total OPI FTEs", "Total VRI FTEs"]].sum()
    pivot_opi = grouped.pivot_table(index=["Date", "Language"], columns=["USD", "Level"], values="Total OPI FTEs", aggfunc="sum", fill_value=0)
    pivot_vri = grouped.pivot_table(index=["Date", "Language"], columns=["USD", "Level"], values="Total VRI FTEs", aggfunc="sum", fill_value=0)

    # Ensure all expected columns exist in the pivot tables
    for pivot in [pivot_opi, pivot_vri]:
        for usd in ["USD", "Global"]:
            for level in ["L3", "L4", "L5"]:
                if (usd, level) not in pivot.columns:
                    pivot[(usd, level)] = 0

    # Compute combined metrics
    combined_rows = []

    for (date, lang) in pivot_opi.index:
        def safe_get(pivot, usd, level):
            return pivot.get((usd, level), pd.Series(0, index=pivot.index)).get((date, lang), 0)

        # OPI values
        usd_l3_opi = safe_get(pivot_opi, "USD", "L3")
        usd_l4_opi = safe_get(pivot_opi, "USD", "L4")
        usd_l5_opi = safe_get(pivot_opi, "USD", "L5")
        global_l3_opi = safe_get(pivot_opi, "Global", "L3")
        global_l4_opi = safe_get(pivot_opi, "Global", "L4")
        global_l5_opi = safe_get(pivot_opi, "Global", "L5")
        
        # VRI values
        usd_l4_vri = safe_get(pivot_vri, "USD", "L4")
        usd_l5_vri = safe_get(pivot_vri, "USD", "L5")
        global_l4_vri = safe_get(pivot_vri, "Global", "L4")
        global_l5_vri = safe_get(pivot_vri, "Global", "L5")        
        
        # Combined values opi
        combined_l3_opi = usd_l3_opi + global_l3_opi
        combined_l4_opi = usd_l4_opi + global_l4_opi
        combined_l5_opi = usd_l5_opi + global_l5_opi
        usd_combined_opi = usd_l3_opi + usd_l4_opi + usd_l5_opi
        global_combined_opi = global_l3_opi + global_l4_opi + global_l5_opi
        combined_combined_opi = usd_combined_opi + global_combined_opi
        
        # Combined values vri
        combined_l4_vri = usd_l4_vri + global_l4_vri
        combined_l5_vri = usd_l5_vri + global_l5_vri
        usd_combined_vri = usd_l4_vri + usd_l5_vri
        global_combined_vri = global_l4_vri + global_l5_vri
        combined_combined_vri = usd_combined_vri + global_combined_vri        

        combined_rows.extend([
            {"Date": date, "Language": lang, "USD": "Combined", "Level": "L3", "Total OPI FTEs": combined_l3_opi},
            {"Date": date, "Language": lang, "USD": "Combined", "Level": "L4", "Total OPI FTEs": combined_l4_opi, "Total VRI FTEs": combined_l4_vri},
            {"Date": date, "Language": lang, "USD": "Combined", "Level": "L5", "Total OPI FTEs": combined_l5_opi, "Total VRI FTEs": combined_l5_vri},
            {"Date": date, "Language": lang, "USD": "USD", "Level": "Combined", "Total OPI FTEs": usd_combined_opi, "Total VRI FTEs": usd_combined_vri },
            {"Date": date, "Language": lang, "USD": "Global", "Level": "Combined", "Total OPI FTEs": global_combined_opi, "Total OPI FTEs": global_combined_vri},
            {"Date": date, "Language": lang, "USD": "Combined", "Level": "Combined", "Total OPI FTEs": combined_combined_opi, "Total VRI FTEs": combined_combined_vri}
        ])

    # Append new rows to original DataFrame
    combined_df = pd.concat([df, pd.DataFrame(combined_rows)], ignore_index=True)
    
    combined_df.rename(columns={"Date" : "startDate per hour" }, inplace=True)
    combined_df["startDate per hour"] = pd.to_datetime(combined_df["startDate per hour"])

    return combined_df

def extract_after_weekly_planner(text):
    start_pos = text.find('Weekly Planner')
    if start_pos != -1:
        return text[start_pos + len('Weekly Planner'):].strip()
    else:
        return None
    
def assign_hybrid_pct(level):
    if level == 'L4 - MSI':
        return lang_hybrid['L4 Hybrid Minutes %']
    elif level == 'L5 - All Call':
        return lang_hybrid['L5 Hybrid Minutes %']
    else:
        return 0.0

# Function to extract level from a string
def extract_level_and_category(input_string):
    start_pos = input_string.find('(')
    end_pos = input_string.find(')', start_pos)
    
    if start_pos != -1 and end_pos != -1:
        l_value = input_string[start_pos + 1:end_pos]
        
        if any(level in l_value for level in ['L3', 'L4', 'L5', 'Combined']):
            level = l_value
        else:
            level = None

        if any(category in input_string for category in ['USD', 'Global', 'Combined']):
            if 'USD' in input_string:
                category = 'USD'
            elif 'Global' in input_string:
                category = 'Global'
            else:
                category = 'Combined'
        return level, category
    else:
        return None, None
    

def parse_flexible_datetime(df, column='startDate per hour'):
    # Try parsing with common formats
    formats = ['%d-%m-%Y %H:%M', '%Y-%m-%d %H:%M', '%d/%m/%Y %H:%M', '%m/%d/%Y %H:%M']
    
    parsed = pd.Series([pd.NaT] * len(df))
    for fmt in formats:
        try:
            temp = pd.to_datetime(df[column], format=fmt, errors='coerce')
            parsed = parsed.combine_first(temp)
        except Exception:
            continue


    parsed = parsed.reset_index(drop=True)
    fallback = pd.to_datetime(df[column], errors='coerce').reset_index(drop=True)
    parsed = parsed.combine_first(fallback)

    # Round to minute precision
    parsed = parsed.dt.floor('min')

    # Format to yyyy-mm-dd HH:MM
    df[column] = parsed.dt.strftime('%Y-%m-%d %H:%M')

    return df

def combine_files_by_prefix(folder_path, prefixes):
    combined_files = {}
    for prefix in prefixes:
        combined_df = pd.DataFrame()
        for file in os.listdir(folder_path):
            if file.startswith(prefix) and file.endswith(('.csv', '.xlsx', '.xlsm')):
                file_path = os.path.join(folder_path, file)
                if file.endswith('.csv'):
                    df = pd.read_csv(file_path)
                elif file.endswith('.xlsm'):
                    df = pd.read_excel(file_path, engine='openpyxl')
                else:
                    df = pd.read_excel(file_path)
                df = parse_flexible_datetime(df,'startDate per hour')
                
                combined_df = pd.concat([combined_df, df], ignore_index=True)
        if not combined_df.empty:
            combined_files[prefix] = combined_df
    return combined_files

# -----------------------------
# Utility Functions
# -----------------------------
def extract_after_weekly_planner(text):
    start_pos = text.find('Weekly Planner')
    return text[start_pos + len('Weekly Planner'):].strip() if start_pos != -1 else None

def extract_level2(level_str):
    if 'L3' in level_str.upper():
        return 'L3'
    elif 'L4' in level_str.upper():
        return 'L4'
    elif 'L5' in level_str.upper():
        return 'L5'
    else:
        return 'Other'

# def to_excel(df):
#     output = BytesIO()
#     with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
#         df.to_excel(writer, index=False, sheet_name='Sheet1')
#     output.seek(0)
#     return output.read()


def save_csv_to_disk(df, file_path):
    df.to_csv(file_path, index=False)


# -----------------------------
# Occupancy Assumption Processing
# -----------------------------
def process_occupancy_assump(file, sheets):
    occ_assumptions_dataframe = pd.DataFrame()

    for sheet in sheets:
        try:
            df = pd.read_excel(file, sheet_name=sheet, header=None)
        except Exception as e:
            st.error(f"Error reading sheet '{sheet}': {e}")
            st.stop()

        if df.empty:
            continue

        df = df.iloc[3:, 3:].reset_index(drop=True)
        df = df.T.reset_index(drop=True)
        df.columns = df.iloc[0].astype(str) + " " + df.iloc[1].astype(str) + " " + df.iloc[2].astype(str)
        df.columns = df.columns.str.replace("nan", "").str.strip()
        df = df[3:].reset_index(drop=True)
        df["Week Of:"] = pd.to_datetime(df["Week Of:"], format='mixed', dayfirst=True, errors="coerce")

        occ_callvol_columns = [col for col in df.columns if "OCC Assumptions" in col or "Volume (ACTUAL)" in col]
        occ_df = df[["Week Of:"] + occ_callvol_columns]

        if "OCC Assumptions (L4):" in occ_df.columns and "OCC Assumptions (L5):" in occ_df.columns:
            occ_df["OCC Assumptions (Combined):"] = occ_df[["OCC Assumptions (L4):", "OCC Assumptions (L5):"]].mean(axis=1)

        occ_df = occ_df[[col for col in occ_df.columns if "Volume (ACTUAL)" not in col]]
        occ_assumptions_dataframe = pd.concat([occ_assumptions_dataframe, occ_df], axis=0)

    return occ_assumptions_dataframe

# -----------------------------
# Expand Weekly → Daily → Hourly
# -----------------------------
def expand_weekly_occ_to_hourly(df_weekly):
    df_weekly['Week Of:'] = pd.to_datetime(df_weekly['Week Of:'], errors='coerce')
    df_weekly = df_weekly.dropna(subset=['Week Of:'])

    df_daily = df_weekly.loc[df_weekly.index.repeat(7)].copy()
    df_daily['Day Offset'] = df_daily.groupby('Week Of:').cumcount()
    df_daily['startDate per hour'] = df_daily['Week Of:'] + pd.to_timedelta(df_daily['Day Offset'], unit='D')
    df_daily = df_daily.drop(columns=['Week Of:', 'Day Offset'])

    df_long = df_daily.melt(id_vars='startDate per hour', var_name='Level', value_name='OCC Assumption')
    df_long['startDate per hour'] = pd.to_datetime(df_long['startDate per hour'], errors='coerce')

    # Expand to 24 hours
    df_hourly = df_long.loc[df_long.index.repeat(24)].copy()
    df_hourly['Hour'] = df_hourly.groupby(['startDate per hour', 'Level']).cumcount()
    df_hourly['Datetime'] = df_hourly['startDate per hour'] + pd.to_timedelta(df_hourly['Hour'], unit='h')
    df_hourly.drop(columns=['startDate per hour', 'Hour'], inplace=True)
    df_hourly.rename(columns={'Datetime': 'startDate per hour'}, inplace=True)
    return df_hourly

# -----------------------------
# Data Cleaning Functions
# -----------------------------
def convert_df_calls(df):
    df['startDate per hour'] = pd.to_datetime(df['startDate per hour'], errors='coerce')
    float_cols = ['ABNs', 'Calls', 'Q2', 'Loaded AHT', 'ABN %']
    percent_cols = ['Met', 'Missed']
    for col in float_cols:
        df[col] = df[col].astype(str).str.replace('%', '').str.replace(',', '')
        df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in percent_cols:
        df[col] = df[col].astype(str).str.replace('%', '').str.replace(',', '')
        df[col] = pd.to_numeric(df[col], errors='coerce') / 100.0
    return df

def convert_df_fte(df, lang):
    # Let pandas infer the format
    df = parse_flexible_datetime(df,'startDate per hour')
    
    df['Level'] = df['Agent Type'].astype(str).str[:2]
    df.rename(columns={'Product': 'Req Media', 'Location': 'USD', 'Level': 'Level_ix'}, inplace=True)
    df['Weekly FTEs'] = df['Weekly FTEs'].astype(str).str.replace(',', '', regex=False)
    df['Weekly FTEs'] = pd.to_numeric(df['Weekly FTEs'], errors='coerce')
    df['USD'] = df['USD'].replace('Non-USD', 'Global')
    df['Req Media'] = df['Req Media'].replace('Video Dedicated', 'VIDEO')
    return df 

def convert_df_occ(df):
    df['startDate per hour'] = pd.to_datetime(df['startDate per hour'], errors='coerce')
    df.rename(columns={'Req. Media': 'Req Media'}, inplace=True)
    df['OCC'] = pd.to_numeric(df['OCC'].astype(str).str.replace('%', '').str.replace(',', ''), errors='coerce')
    return df


def convert_df_hybrid(df):
    for col in ['L4 Hybrid Minutes %', 'L5 Hybrid Minutes %']:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace('%', '').str.replace(',', ''), errors='coerce') / 100.0
    return df

# -----------------------------
# Hybrid % Assignment
# -----------------------------
# Function to get hybrid percentages by language
def get_hybrid_percentages_by_language(df_hybrid, language):
    language = language.strip().upper()
    matching_languages = df_hybrid[df_hybrid['Language'].str.upper().str.startswith(language.upper(), na=False)]
#     matching_languages = df_hybrid[df_hybrid['Language'].str.upper().str.contains(language, na=False)]

    if matching_languages.empty:
        raise ValueError(f"Language containing '{language}' not found in hybrid data.")
    hybrid_row = matching_languages.iloc[0]
    matched_language = hybrid_row['Language']
    return matched_language, {
        'Language': matched_language,
        'L4 Hybrid Minutes %': float(hybrid_row['L4 Hybrid Minutes %']),
        'L5 Hybrid Minutes %': float(hybrid_row['L5 Hybrid Minutes %'])
    }

# -----------------------------
# Streamlit App Entry Point
# -----------------------------
def run_hourly_datamart_tool():
    global lang_hybrid
    st.title("📊 Hour-Level Datamart Builder")

    folder_path = st.text_input("Enter folder path containing all files:")
    uploaded_hybrid_file = st.file_uploader("Upload Hybrid CSV file", type="csv")
    uploaded_wp_file = st.file_uploader("Upload Weekly Planner XLSM file", type="xlsm")
    planner_sheets = st.multiselect("Select Weekly Planner Sheets", [
        "1. Weekly Planner OPI", "2. Weekly Planner VRI", "3. UKD"
    ])

    if st.button("Process Files"):
        if not folder_path or not planner_sheets:
            st.warning("Please provide folder path and select planner sheets.")
            return

        # Combine files
        combined = combine_files_by_prefix(folder_path, ["HMO-Calls","HMO-FTE","HMO-OCC"])
        if not all(k in combined for k in ["HMO-Calls","HMO-FTE","HMO-OCC"]):
            st.error("Missing one or more required file types (HMO-Calls, HMO-FTE, HMO-OCC).")
            return

        # Process occupancy
        occ_assump_df = process_occupancy_assump(uploaded_wp_file, planner_sheets)
        if any("VRI" in item for item in planner_sheets):
                   
            # Create new columns with Global and USD suffixes
            occ_assump_df['Combined OCC Assumptions (L4):'] = occ_assump_df['OCC Assumptions (L4):']
            occ_assump_df['Combined OCC Assumptions (L5):'] = occ_assump_df['OCC Assumptions (L5):']                
            occ_assump_df['Combined OCC Assumptions (Combined):'] = (
                occ_assump_df[['OCC Assumptions (L4):', 'OCC Assumptions (L5):']]
                .mean(axis=1, skipna=True)
            )
                
            occ_assump_df.drop(columns=['OCC Assumptions (L4):', 'OCC Assumptions (L5):','OCC Assumptions (Combined):']
                                            , inplace=True)

        else:
            occ_assump_df['Combined OCC Assumptions (L4):'] = (
                occ_assump_df[['Global OCC Assumptions (L4):', 'USD OCC Assumptions (L4):']]
                .mean(axis=1, skipna=True)
            )

                
            occ_assump_df['Combined OCC Assumptions (L5):'] = (
                occ_assump_df[['Global OCC Assumptions (L5):', 'USD OCC Assumptions (L5):']]
                .mean(axis=1, skipna=True)
            )               
                
            occ_assump_df['USD OCC Assumptions (Combined):'] = (
                occ_assump_df[['USD OCC Assumptions (L3):', 'USD OCC Assumptions (L4):', 'USD OCC Assumptions (L5):']]
                .mean(axis=1, skipna=True)
            )
                
            occ_assump_df['Global OCC Assumptions (Combined):'] = (
                occ_assump_df[['Global OCC Assumptions (L3):', 'Global OCC Assumptions (L4):', 'Global OCC Assumptions (L5):']]
                .mean(axis=1, skipna=True)
            )
                
            occ_assump_df['Combined OCC Assumptions (Combined):'] = (
                occ_assump_df[['Global OCC Assumptions (Combined):', 'USD OCC Assumptions (Combined):']]
                .mean(axis=1, skipna=True)
            )

        occ_assmp_hourly = expand_weekly_occ_to_hourly(occ_assump_df)
        occ_assmp_hourly[['Level', 'USD']] = occ_assmp_hourly['Level'].apply(lambda x: pd.Series(extract_level_and_category(x)))

        # Process other files
        lang = uploaded_wp_file.name[:3] # crude language detection

        # Convert list to string       
        xlsm_files_str = uploaded_wp_file.name[:3]
        
        lang = xlsm_files_str

        df_calls = convert_df_calls(combined["HMO-Calls"])
        df_fte = convert_df_fte(combined["HMO-FTE"], lang)
        df_occ = convert_df_occ(combined["HMO-OCC"])

        df_hybrid = pd.read_csv(uploaded_hybrid_file)
        df_hybrid = convert_df_hybrid(df_hybrid)

        # df_calls, df_fte, df_hybrid, df_occ = process_csv_files(combined["HMO-FTE"], combined["HMO-Calls"], 
        #                                                             combined["HMO-OCC"], uploaded_hybrid_file, lang)
        
        df_fte['Level'] = df_fte['Level_ix'].apply(extract_level2)

        matched_language, lang_hybrid = get_hybrid_percentages_by_language(df_hybrid, lang)        



        # Step 1: Filter FTE data for the processing language only
        if matched_language=="ASL":
            matched_language="AMERICAN SIGN LANGUAGE"

        df_fte_lang = df_fte[df_fte['Language'] == matched_language].copy()

            
        # Replace NaN values with zero
        df_fte_lang = df_fte_lang.fillna(0)
            
        # Ensure clean 'Product' column
        df_fte_lang = parse_flexible_datetime(df_fte_lang,'startDate per hour')
            
        df_fte_pivoted = df_fte_lang.pivot_table(
            index=['startDate per hour', 'USD','Language','Level_ix'],
            columns='Req Media',
            values='Weekly FTEs',
            aggfunc='sum'
        ).reset_index()
            

        if 'Hybrid' not in df_fte_pivoted.columns:
            df_fte_pivoted['Hybrid'] = None
                
        if 'VIDEO' not in df_fte_pivoted.columns:
            df_fte_pivoted['VIDEO'] = None 
                  
        if 'OPI' not in df_fte_pivoted.columns:
            df_fte_pivoted['OPI'] = None                                
                                
        df_fte_grouped = df_fte_pivoted.copy()


        df_fte_grouped['Hybrid %'] = df_fte_grouped['Level_ix'].apply(assign_hybrid_pct)
        df_fte_grouped.rename(columns={'Level_ix': 'Level'}, inplace=True)
            
        # st.write(df_fte_grouped)
        # Ensure numeric types and fill missing values
        numeric_cols = ['Hybrid', 'Hybrid %', 'OPI', 'VIDEO']
        for col in numeric_cols:
            df_fte_grouped[col] = pd.to_numeric(df_fte_grouped[col], errors='coerce').fillna(0)

        # Calculate Hybrid FTEs
        df_fte_grouped['Hybrid_FTE_VRI'] = df_fte_grouped['Hybrid'] * df_fte_grouped['Hybrid %']
        df_fte_grouped['Hybrid_FTE_OPI'] = df_fte_grouped['Hybrid'] - df_fte_grouped['Hybrid_FTE_VRI']

        # Calculate Total FTEs
        df_fte_grouped['Total OPI FTEs'] = df_fte_grouped[['OPI', 'Hybrid_FTE_OPI']].sum(axis=1)
        df_fte_grouped['Total VRI FTEs'] = df_fte_grouped[['VIDEO', 'Hybrid_FTE_VRI']].sum(axis=1)
            
        # Replace NaN values with zero
        df_fte_grouped = df_fte_grouped.fillna(0)
            
            
        # Select only the necessary columns from df_fte_grouped for merging
        df_opi_vri_fte = df_fte_grouped[['startDate per hour', 'Language','USD','Level', 'Total OPI FTEs', 'Total VRI FTEs']]

        print("df_opi_vri_fte")
        print(df_opi_vri_fte.head(5))             
            
            
        # creating combined level data for FTE data
        df_opi_vri_fte_comb = fte_combined_level_calc(df_opi_vri_fte)
            
            
        planner_type_txt=' '.join(planner_sheets)
        type_data=extract_after_weekly_planner(planner_type_txt)

            
        # Ensure date format consistency in df_calls
        df_calls = parse_flexible_datetime(df_calls,'startDate per hour')
       
        # Ensure date format consistency in df_opi_vri_fte_comb
        df_opi_vri_fte_comb = parse_flexible_datetime(df_opi_vri_fte_comb,'startDate per hour') 

        # Merge only OPI/VRI totals into df_calls using common keys
        df_calls_with_fte = pd.merge(
            df_calls,
            df_opi_vri_fte_comb,
            on=['startDate per hour','USD', 'Language', 'Level'],
            how='left'
        )
 
        df_calls_with_fte['startDate per hour'] = pd.to_datetime(df_calls_with_fte['startDate per hour'])

        occ_assmp_hourly['startDate per hour'] = pd.to_datetime(occ_assmp_hourly['startDate per hour'])

        final_fte_occ_assump = df_calls_with_fte.merge(occ_assmp_hourly, on =['startDate per hour', 'Level', 'USD'], how='left')
        
        df_occ.loc[df_occ['Req Media'] == "Video", 'Req Media'] = "VIDEO"
            
        final_fte_occ_assump_occ_rate = final_fte_occ_assump.merge(
            df_occ,
            on=['startDate per hour','Language','USD', 'Level', 'Req Media'],
            how='inner'
        )

        final_data = final_fte_occ_assump_occ_rate.copy()
            

        final_data['OCC Assumption'] = final_data['OCC Assumption'].fillna(final_data['OCC Assumption'].mean())
        final_data['OCC'].fillna(final_data['OCC'].mean(), inplace=True)           
        final_data["Requirement"] = final_data["Calls"] * final_data['Loaded AHT'] / ((2250 / 7) * final_data["OCC Assumption"])
            
        if 'OPI' in type_data.upper():
            final_data.rename(columns={'Total OPI FTEs':'Staffing'}, inplace=True)
        else:
            final_data.rename(columns={'Total VRI FTEs':'Staffing'}, inplace=True)

        final_data['Demand'] = final_data['Calls'] * final_data['Loaded AHT']
        final_data['Staffing Diff'] = final_data['Staffing'] - final_data['Staffing'].shift(1)
        final_data.rename(columns={"OCC":"Occupancy Rate"}, inplace=True)
        final_data.rename(columns={"OCC Assumption":"Occ Assumption"}, inplace=True)

        final_data = final_data[['startDate per hour', 'Language', 'USD', 'Req Media', 'Level', 'ABNs',
            'Calls', 'Q2', 'Loaded AHT', 'ABN %', 'Met', 'Missed','Demand','Occ Assumption',
                                'Requirement','Staffing','Occupancy Rate','Staffing Diff']]
                  
        # Save the DataFrame to an Excel file          
        if 'OPI' in type_data.upper():

            final_data_opi_or_vri = final_data[final_data['Req Media'] == 'OPI']
        else:
            final_data_opi_or_vri = final_data[final_data['Req Media'] == 'VIDEO']

        # Multiply by 1.4
        final_data_opi_or_vri['Staffing'] *= 1.4
                
        st.write(final_data_opi_or_vri) 
            
        # ✅ Provide download directly
        excel_bytes = to_excel(final_data_opi_or_vri)
        st.download_button(
            label="Download Processed Excel File",
            data=excel_bytes,
            file_name=f'{lang}_{type_data}_output.xlsx',
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        st.success("File processed successfully!")  

if __name__ == "__main__":
    run_hourly_datamart_tool()