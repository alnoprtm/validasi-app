import streamlit as st
import pandas as pd
import io
import numpy as np
from datetime import datetime

# ==========================================
# 1. KONFIGURASI HALAMAN & STATE
# ==========================================
st.set_page_config(
    page_title="Validasi Data Migas",
    page_icon="🛢️",
    layout="wide"
)

# Batas maksimum sel untuk styling otomatis
MAX_CELLS_FOR_STYLING = 15000 

# Inisialisasi Session State
if 'asset_register_df' not in st.session_state:
    st.session_state['asset_register_df'] = None
if 'daily_well_prod_df' not in st.session_state:
    st.session_state['daily_well_prod_df'] = None
if 'daily_asset_prod_df' not in st.session_state:
    st.session_state['daily_asset_prod_df'] = None

# ==========================================
# 2. DEFINISI ATURAN & SKEMA
# ==========================================

# --- A. CONFIG MODUL 1: ASSET REGISTER ---
ASSET_CONFIG = {
    "Zona": {
        "all_columns": ["EntityID", "Region", "Zone", "Latitude", "Longitude"],
        "validate_columns": ["Latitude", "Longitude"],
        "filter_hierarchy": ["Region", "Zone"]
    },
    "Working Area": {
        "all_columns": ["Entity ID", "Entity Type", "Region", "Zone", "Working Area Name", 
                        "Participating Interest", "Latitude", "Longitude", "Profile Summary"],
        "validate_columns": ["Latitude", "Longitude"],
        "filter_hierarchy": ["Region", "Zone", "Working Area Name"]
    },
    "Asset Operation": {
        "all_columns": ["Entity ID", "Region", "Zona", "Working Area", "Working Area Type", 
                        "Asset Operation", "Latitude", "Longitude"],
        "validate_columns": ["Latitude", "Longitude"],
        "filter_hierarchy": ["Region", "Zona", "Working Area", "Asset Operation"]
    },
    "Well": {
        "all_columns": ["Regional", "Zona", "Working Area", "Asset Operation", "Entity ID", 
                        "Well", "Well Status", "Skin Status", "Lifting Method", "Reservoir Name", 
                        "Main Fluid Type", "Latitude", "Longitude", "POP Date Target", 
                        "POP Date Actual", "Profile Summary"],
        "validate_columns": ["Well Status", "Skin Status", "Lifting Method", "Reservoir Name", 
                             "Main Fluid Type", "Latitude", "Longitude", "POP Date Target", 
                             "POP Date Actual", "Profile Summary"],
        "filter_hierarchy": ["Regional", "Zona", "Working Area", "Asset Operation", "Well"]
    }
}

# --- B. CONFIG MODUL 2: PRODUCTION WELL TEST ---
WELL_TEST_COLUMNS = [
    "Regional", "Zona", "Working Area", "Asset Operation", "Well", "Entity ID",
    "Test Date (dd/mm/yyyy)", "Test Duration(Hours)", "Layer Name", "Lifting Method Name",
    "Oil (BOPD)", "Water (BWPD)", "Gas (MMSCFD)", "Condensate (BCPD)", "Fluid (BFPD)",
    "Water Cut (%)", "Gas Lift (MMSCFD)", "Gas Utilization Factor",
    "Productivity Index Oil", "Productivity Index Gas", "Choke Size (/64)",
    "Tubing Head Pressure (Psig)", "Casing Head Pressure (Psig)",
    "Separator Pressure (Psig)", "Separator Temperature (F)",
    "Pump Intake Pressure (Psig)", "API", "Specific Gravity", "Remarks"
]

WELL_TEST_VALIDATE_BASIC = [
    "Test Duration(Hours)", "Layer Name", "Lifting Method Name",
    "Oil (BOPD)", "Water (BWPD)", "Gas (MMSCFD)", "Condensate (BCPD)", "Fluid (BFPD)",
    "Water Cut (%)", "Productivity Index Oil", "Productivity Index Gas",
    "Choke Size (/64)", "Tubing Head Pressure (Psig)", "Casing Head Pressure (Psig)",
    "API", "Specific Gravity", "Remarks"
]

WELL_TEST_HIERARCHY = ["Regional", "Zona", "Working Area", "Asset Operation", "Well"]

# --- C. CONFIG MODUL 3: WELL PARAMETER (WELL ON) ---
WP_ADMIN_COLS = [
    "Regional", "Zona", "Working Area", "Asset Operation", "Entity ID",
    "Well", "Well Status", "Skin Status", "Lifting Method", "Reservoir Name",
    "Main Fluid Type", "Latitude", "Longitude", "POP Date Target",
    "POP Date Actual", "Profile Summary"
]

WELL_PARAM_CONFIG = {
    "ESP": {
        "input_cols": WP_ADMIN_COLS + ["Ownership", "Manufacturer", "Brand", "Supplier / Vendor", "Dynamic Fluid Level (ft-MD)", "Static Fluid Level (ft-MD)", "Pump Efficiency (%)", "Pump Optimal Design Rate / Capacity Design (BFPD)", "Pump Setting Depth (ft-MD)", "Pump Type", "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Serial Name", "Stages", "Frequency (Hz)", "Ampere (Amp)", "Voltage (Volt)", "Rotation (RPM)", "EQPM HP (HP)", "EQPM Rate (BFPD)", "Motor Voltage (Volt)", "Motor Amps (Amp)", "Automatic Gas Handler", "Shroud", "Protector", "Sensor", "Start Date", "DHP Date (dd/MM/yyyy)", "WHP", "Discharge Pressure (Psi)", "PBHP (Psi)", "Remarks"],
        "check_cols": ["Ownership", "Manufacturer", "Brand", "Supplier / Vendor", "Pump Efficiency (%)", "Pump Optimal Design Rate / Capacity Design (BFPD)", "Pump Setting Depth (ft-MD)", "Pump Type", "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Serial Name", "Stages", "Frequency (Hz)", "Ampere (Amp)", "Voltage (Volt)", "Rotation (RPM)", "EQPM HP (HP)", "EQPM Rate (BFPD)", "Motor Voltage (Volt)", "Motor Amps (Amp)", "Automatic Gas Handler", "Shroud", "Protector", "Sensor", "Start Date", "DHP Date (dd/MM/yyyy)", "WHP", "Discharge Pressure (Psi)", "PBHP (Psi)", "Remarks"],
        "conditional": {"Dynamic": "Dynamic Fluid Level (ft-MD)", "Static": "Static Fluid Level (ft-MD)"}
    },
    "GL": {
        "input_cols": WP_ADMIN_COLS + ["Ownership", "Manufacturer", "Brand", "Supplier / Vendor", "Dynamic Fluid Level (ft-MD)", "Static Fluid Level (ft-MD)", "Injection Fluid Pressure (Psig)", "InjectionTemperature (F)", "Number Gas Lift Valve", "Gas Injection Choke (MMSCFD)", "Rate Injection Choke (MMSCFD)", "Rate Liquid Optimization (MMSCFD)", "Gas Injection Rate (MMSCFD)", "PBHP (Psi)", "Remarks"],
        "check_cols": ["Ownership", "Manufacturer", "Brand", "Supplier / Vendor", "Injection Fluid Pressure (Psig)", "InjectionTemperature (F)", "Number Gas Lift Valve", "Gas Injection Choke (MMSCFD)", "Rate Injection Choke (MMSCFD)", "Rate Liquid Optimization (MMSCFD)", "Gas Injection Rate (MMSCFD)", "PBHP (Psi)", "Remarks"],
        "conditional": {"Dynamic": "Dynamic Fluid Level (ft-MD)", "Static": "Static Fluid Level (ft-MD)"}
    },
    "HJP": {
        "input_cols": WP_ADMIN_COLS + ["Ownership", "Manufacturer", "Brand", "Supplier / Vendor", "Nozzle (Inch)", "Throat (Inch)", "Injection Fluid Pressure (Psig)", "Injection Point (ft-MD)", "PBHP (Psi)", "Remarks"],
        "check_cols": ["Ownership", "Manufacturer", "Brand", "Supplier / Vendor", "Nozzle (Inch)", "Throat (Inch)", "Injection Fluid Pressure (Psig)", "Injection Point (ft-MD)", "PBHP (Psi)", "Remarks"],
        "conditional": {} 
    },
    "HPU": {
        "input_cols": WP_ADMIN_COLS + ["Ownership", "Manufacturer", "Brand", "Supplier / Vendor", "Dynamic Fluid Level (ft-MD)", "Static Fluid Level (ft-MD)", "Pump Efficiency (%)", "Pump Optimal Design Rate / Capacity Design (BFPD)", "Pump Type", "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Serial Name", "Pump Setting Depth (ft-MD)", "Stroke Length", "Stroke Per Minute (SPM)", "PBHP (Psi)", "Remarks"],
        "check_cols": ["Ownership", "Manufacturer", "Brand", "Supplier / Vendor", "Pump Efficiency (%)", "Pump Optimal Design Rate / Capacity Design (BFPD)", "Pump Type", "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Serial Name", "Pump Setting Depth (ft-MD)", "Stroke Length", "Stroke Per Minute (SPM)", "PBHP (Psi)", "Remarks"],
        "conditional": {"Dynamic": "Dynamic Fluid Level (ft-MD)", "Static": "Static Fluid Level (ft-MD)"}
    },
    "PCP": {
        "input_cols": WP_ADMIN_COLS + ["Ownership", "Manufacturer", "Brand", "Supplier / Vendor", "Dynamic Fluid Level (ft-MD)", "Static Fluid Level (ft-MD)", "Pump Efficiency (%)", "Pump Optimal Design Rate / Capacity Design (BFPD)", "Pump Type", "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Serial Name", "Pump Setting Depth (ft-MD)", "PBHP (Psi)", "Remarks"],
        "check_cols": ["Ownership", "Manufacturer", "Brand", "Supplier / Vendor", "Pump Efficiency (%)", "Pump Optimal Design Rate / Capacity Design (BFPD)", "Pump Type", "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Serial Name", "Pump Setting Depth (ft-MD)", "PBHP (Psi)", "Remarks"],
        "conditional": {"Dynamic": "Dynamic Fluid Level (ft-MD)", "Static": "Static Fluid Level (ft-MD)"}
    },
    "SRP": {
        "input_cols": WP_ADMIN_COLS + ["Ownership", "Manufacturer", "Brand", "Supplier / Vendor", "Stroke Length", "Stroke Per Minute (SPM)", "Dynamic Fluid Level (ft-MD)", "Static Fluid Level (ft-MD)", "Pump Efficiency (%)", "Pump Optimal Design Rate / Capacity Design (BFPD)", "Pump Type", "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Pump Setting Depth (ft-MD)", "PBHP (Psi)", "Remarks"],
        "check_cols": ["Ownership", "Manufacturer", "Brand", "Supplier / Vendor", "Stroke Length", "Stroke Per Minute (SPM)", "Pump Efficiency (%)", "Pump Optimal Design Rate / Capacity Design (BFPD)", "Pump Type", "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Pump Setting Depth (ft-MD)", "PBHP (Psi)", "Remarks"],
        "conditional": {"Dynamic": "Dynamic Fluid Level (ft-MD)", "Static": "Static Fluid Level (ft-MD)"}
    }
}

WELL_PARAM_HIERARCHY = ["Regional", "Zona", "Working Area", "Asset Operation", "Well"]

# --- D. CONFIG MODUL 4: EVENT ARTIFICIAL LIFT (WELL OFF) ---
EVENT_AL_COLUMNS = [
    "Event Name", "Regional", "Zona", "Working Area", "Asset Operation", "Well", "Entity ID", "Affected Well",
    "Event Type", "Loss Type", "Event Status", "Event Start Time (dd/MM/yyy HH:mm)", "Event End Time (dd/MM/yyy HH:mm)",
    "Event Downtime", "Event RKAP ID", "Event Description", "Loss Calculation",
    "Off Oil (Bbl)", "Off Gas (MMSCF)", "Off Condensate Formation", "Off Condendate Plant",
    "Low Oil (Bbl)", "Low Gas (MMSCF)", "Low Condensate Formation", "Low Condensate Plant",
    "System Source Name", "Equipment Source Name", "Parent Cause Name", "Child Cause Name",
    "Type Cause Name", "Family Cause Name", "Root Cause Description", "AIMS Equipment Tag ID",
    "Diagnostic Review Flag", "Diagnostic Analysis Status", "Is Event Artificial Lift", "Event Artificial Lift Type",
    "Is Replacement", "Lifting Method", "Ownership", "Manufacturer", "Brand", "Supplier/Vendor",
    "Install Date", "Run Life (Days)", "AL Purchase Cost (USD)", "Daily Operating Cost (USD)",
    "Cumulative Operating Cost (USD)", "Rig Cost of Installation (USD)", "Surface Recovery Cost (USD)",
    "Failed Component Name", "Failure Cause", "Event Artificial Lift Remarks"
]

EVENT_AL_VALIDATE_COLS = [
    "Is Replacement", "Lifting Method", "Ownership", "Manufacturer", "Brand", "Supplier/Vendor",
    "Install Date", "Run Life (Days)", "Daily Operating Cost (USD)", "Cumulative Operating Cost (USD)",
    "Rig Cost of Installation (USD)", "Failed Component Name", "Failure Cause", "Event Artificial Lift Remarks"
]

EVENT_AL_HIERARCHY = ["Regional", "Zona", "Working Area", "Asset Operation", "Well"]

# --- E. CONFIG MODUL 5: DAILY PRODUCTION DATA ---

DAILY_PROD_CONFIG = {
    "Working Area Production": {
        "columns": ["Regional", "Zona", "Working Area", "EntityID", "Production Date", "Oil (BOPD)", "Gas (MMSCFD)", "Water (BWPD)"],
        "validate": ["Oil (BOPD)", "Gas (MMSCFD)", "Water (BWPD)"],
        "hierarchy": ["Regional", "Zona", "Working Area"]
    },
    "Asset Operation Production": {
        "columns": ["Regional", "Zona", "Working Area", "Asset Operation", "EntityID", "Production Date", "Total Oil (BOPD)", "Oil (BOPD)", "Gas (MMSCFD)", "Water (BWPD)", "Cond. Form. (BCPD)", "Cond. Plant. (BCPD)"],
        "validate": ["Oil (BOPD)", "Gas (MMSCFD)", "Water (BWPD)", "Cond. Form. (BCPD)", "Cond. Plant. (BCPD)"],
        "hierarchy": ["Regional", "Zona", "Working Area", "Asset Operation"]
    },
    "Well Production": {
        "columns": ["No", "Regional", "Zona", "Working Area", "Asset Operation", "Well", "Entity ID", "Production Date", "Flowing Time", "Down Time", "Total Oil (BOPD)", "Oil (BOPD)", "Gas (MMSCFD)", "Water (BWPD)", "Cond.Form. (BCPD)"],
        "validate": ["Oil (BOPD)", "Gas (MMSCFD)", "Water (BWPD)", "Cond.Form. (BCPD)"],
        "hierarchy": ["Regional", "Zona", "Working Area", "Asset Operation", "Well"]
    }
}

# ==========================================
# 3. FUNGSI-FUNGSI LOGIKA (OPTIMIZED)
# ==========================================

@st.cache_data(ttl=3600)
def load_excel_file(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        df.columns = df.columns.str.strip()
        df = df.dropna(how='all')
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def highlight_nulls(s):
    is_missing = pd.isna(s) | (s == "")
    return ['background-color: #ffeeb0' if v else '' for v in is_missing]

def display_dataframe_optimized(df, target_cols=None, use_highlight=True):
    total_cells = df.shape[0] * df.shape[1]
    if use_highlight and total_cells <= MAX_CELLS_FOR_STYLING:
        try:
            if target_cols:
                valid_cols = [c for c in target_cols if c in df.columns]
                st.dataframe(df.style.apply(highlight_nulls, axis=1, subset=valid_cols), use_container_width=True)
            else:
                st.dataframe(df, use_container_width=True)
        except:
            st.dataframe(df, use_container_width=True)
    else:
        if total_cells > MAX_CELLS_FOR_STYLING:
            st.caption(f"⚡ Mode Performa Tinggi: Pewarnaan dinonaktifkan ({total_cells:,} sel).")
        st.dataframe(df, use_container_width=True)

@st.cache_data
def calculate_simple_completeness(df, target_cols):
    if df.empty: return 0.0
    total_expected = len(df) * len(target_cols)
    if total_expected == 0: return 0.0
    filled = df[target_cols].replace('', pd.NA).count().sum()
    return (filled / total_expected) * 100

def get_missing_details(df, target_cols):
    valid_cols = [c for c in target_cols if c in df.columns]
    temp_df = df[valid_cols].replace('', pd.NA)
    return temp_df.isnull().sum().reset_index(name='Jumlah Kosong').rename(columns={'index': 'Nama Kolom'})

# --- FUNGSI KHUSUS WELL TEST ---
@st.cache_data
def calculate_well_test_completeness(df):
    if df.empty: return 0.0
    total_expected = len(df) * len(WELL_TEST_VALIDATE_BASIC)
    total_filled = df[WELL_TEST_VALIDATE_BASIC].replace('', pd.NA).count().sum()
    
    esp_mask = df['Lifting Method Name'].astype(str).str.contains('Electric Submersible Pump', case=False, na=False)
    if esp_mask.any():
        total_expected += esp_mask.sum()
        total_filled += df.loc[esp_mask, 'Pump Intake Pressure (Psig)'].replace('', pd.NA).notna().sum()
        
    gl_mask = df['Lifting Method Name'].astype(str).str.contains('Gas Lift', case=False, na=False)
    if gl_mask.any():
        total_expected += (gl_mask.sum() * 2)
        total_filled += df.loc[gl_mask, ['Gas Lift (MMSCFD)', 'Gas Utilization Factor']].replace('', pd.NA).notna().sum().sum()
        
    if total_expected == 0: return 0.0
    return (total_filled / total_expected) * 100

@st.cache_data
def validate_engineering_rules(df_test, df_asset):
    res = df_test.copy()
    num_cols = ["Test Duration(Hours)", "Oil (BOPD)", "Water (BWPD)", "Gas (MMSCFD)", "Condensate (BCPD)", "Fluid (BFPD)"]
    for col in num_cols:
        res[col] = pd.to_numeric(res[col], errors='coerce')

    res['Rule4_Pass'] = True
    for col in num_cols:
        invalid = (res[col] < 0) & res[col].notna()
        res.loc[invalid, 'Rule4_Pass'] = False

    is_producing_check = (res["Oil (BOPD)"] > 0) | (res["Water (BWPD)"] > 0) | \
                         (res["Gas (MMSCFD)"] > 0) | (res["Condensate (BCPD)"] > 0) | \
                         (res["Fluid (BFPD)"] > 0)
    all_prod_nan = res[["Oil (BOPD)", "Water (BWPD)", "Gas (MMSCFD)", "Condensate (BCPD)", "Fluid (BFPD)"]].isna().all(axis=1)
    res['Rule2_Pass'] = ~((res["Test Duration(Hours)"] > 0) & (~is_producing_check) & (~all_prod_nan))
    
    produces_something = (res["Oil (BOPD)"] > 0) | (res["Water (BWPD)"] > 0) | \
                         (res["Gas (MMSCFD)"] > 0) | (res["Condensate (BCPD)"] > 0)
    res['Rule3_Pass'] = ~((produces_something) & (res["Test Duration(Hours)"] <= 0))

    res['Rule1_Pass'] = True 
    missing_wells_df = pd.DataFrame()
    check_rule1_active = False
    
    if df_asset is not None and not df_asset.empty:
        if 'Well Status' in df_asset.columns and 'Well' in df_asset.columns:
            check_rule1_active = True
            asset_active = df_asset[df_asset['Well Status'].str.contains("Active Well Producing", case=False, na=False)].copy()
            hierarchy_cols = ["Regional", "Zona", "Working Area", "Asset Operation"]
            cols_exist_test = all(col in res.columns for col in hierarchy_cols)
            cols_exist_asset = all(col in asset_active.columns for col in hierarchy_cols)
            
            if cols_exist_test and cols_exist_asset:
                test_scope = res[hierarchy_cols].drop_duplicates()
                asset_in_scope = asset_active.merge(test_scope, on=hierarchy_cols, how='inner')
                test_wells_set = set(res['Well'].unique())
                missing_mask = ~asset_in_scope['Well'].isin(test_wells_set)
                missing_wells_df = asset_in_scope[missing_mask].copy()
                active_prod_wells = set(asset_active['Well'].unique())
            else:
                active_prod_wells = set(asset_active['Well'].unique())
                test_wells_set = set(res['Well'].unique())
                missing_wells_df = pd.DataFrame(columns=asset_active.columns)

            res['Test Date'] = pd.to_datetime(res['Test Date (dd/mm/yyyy)'], format='%d/%m/%Y', errors='coerce')
            active_wells_in_test = test_wells_set.intersection(active_prod_wells)
            max_date_file = res['Test Date'].max()
            if pd.isna(max_date_file): max_date_file = datetime.now()
            bad_frequency_wells = set()
            temp_df = res[res['Well'].isin(active_wells_in_test)][['Well', 'Test Date']].dropna()
            if not temp_df.empty:
                temp_df = temp_df.sort_values(['Well', 'Test Date'])
                temp_df['Prev_Date'] = temp_df.groupby('Well')['Test Date'].shift(1)
                temp_df['Diff_Days'] = (temp_df['Test Date'] - temp_df['Prev_Date']).dt.days
                bad_frequency_wells.update(temp_df[temp_df['Diff_Days'] > 90]['Well'].unique())
                last_dates = temp_df.groupby('Well')['Test Date'].max()
                tail_gaps = (max_date_file - last_dates).dt.days
                bad_frequency_wells.update(tail_gaps[tail_gaps > 90].index.tolist())
            res.loc[res['Well'].isin(bad_frequency_wells), 'Rule1_Pass'] = False

    res['Keterangan Error'] = ""
    if check_rule1_active:
        res.loc[~res['Rule1_Pass'], 'Keterangan Error'] += "⚠️ Frekuensi Test Kurang (Gap > 3 Bulan) | "
    res.loc[~res['Rule2_Pass'], 'Keterangan Error'] += "⚠️ Durasi > 0 tapi Produksi Nihil | "
    res.loc[~res['Rule3_Pass'], 'Keterangan Error'] += "⚠️ Produksi Ada tapi Durasi 0/Kosong | "
    res.loc[~res['Rule4_Pass'], 'Keterangan Error'] += "⚠️ Terdapat Nilai Negatif | "
    res['Keterangan Error'] = res['Keterangan Error'].str.rstrip(" | ")
    res.loc[res['Keterangan Error'] == "", 'Keterangan Error'] = "OK"
    return res, missing_wells_df

# --- FUNGSI KHUSUS WELL PARAMETER ---
@st.cache_data
def calculate_well_param_completeness(df, lift_type):
    if df.empty: return None
    config = WELL_PARAM_CONFIG[lift_type]
    base_cols = config['check_cols']
    cond_map = config['conditional'] 
    if 'Well Status' not in df.columns: return None
    df['Status_Norm'] = df['Well Status'].astype(str).str.strip()
    mask_active = df['Status_Norm'].str.contains("Active Well Producing|Active Well Non Production", case=False, na=False)
    df_active = df[mask_active].copy()
    if 'Lifting Method' in df_active.columns:
        lm_str = df_active['Lifting Method'].astype(str).str.strip()
        mask_lift = (~lm_str.isin(['nan', 'NaN', 'None', '']))
        df_active = df_active[mask_lift]
    if df_active.empty: return None 
    mask_producing = df_active['Status_Norm'].str.contains("Active Well Producing", case=False, na=False)
    mask_non_prod = df_active['Status_Norm'].str.contains("Active Well Non Production", case=False, na=False)
    total_expected = len(df_active) * len(base_cols)
    total_filled = df_active[base_cols].replace('', pd.NA).count().sum()
    if 'Dynamic' in cond_map:
        dyn_col = cond_map['Dynamic']
        if dyn_col in df_active.columns:
            total_expected += mask_producing.sum()
            total_filled += df_active.loc[mask_producing, dyn_col].replace('', pd.NA).notna().sum()
    if 'Static' in cond_map:
        stat_col = cond_map['Static']
        if stat_col in df_active.columns:
            total_expected += mask_non_prod.sum()
            total_filled += df_active.loc[mask_non_prod, stat_col].replace('', pd.NA).notna().sum()
    score = 0.0
    if total_expected > 0: score = (total_filled / total_expected) * 100
    return {"score": score, "filled": total_filled, "expected": total_expected, "active_rows": len(df_active)}

@st.cache_data
def validate_well_parameter_rules(df_input, lift_type):
    df_merged = df_input.copy()
    if 'Well Status' not in df_merged.columns:
        df_merged['Keterangan Error'] = "CRITICAL: Kolom 'Well Status' tidak ditemukan."
        return df_merged
    df_merged['Well Status'] = df_merged['Well Status'].fillna("Unknown")
    df_merged['Status_Norm'] = df_merged['Well Status'].astype(str).str.strip()
    df_merged['Keterangan Error'] = ""
    mask_producing = df_merged['Status_Norm'].str.contains("Active Well Producing", case=False, na=False)
    mask_non_prod = df_merged['Status_Norm'].str.contains("Active Well Non Production", case=False, na=False)
    mask_active = mask_producing | mask_non_prod

    def check_positive_vec(col_name, mask_rows):
        if col_name in df_merged.columns:
            val_col = pd.to_numeric(df_merged[col_name], errors='coerce')
            mask_invalid = (val_col <= 0) & (val_col.notna())
            target_invalid = mask_rows & mask_invalid
            if target_invalid.any(): df_merged.loc[target_invalid, 'Keterangan Error'] += f"{col_name} <= 0 | "

    def check_design_range_vec():
        required = ["Pump Optimal Design Rate / Capacity Design (BFPD)", "Min BLPD (BFPD)", "Max BLPD (BFPD)"]
        if all(c in df_merged.columns for c in required):
            des = pd.to_numeric(df_merged[required[0]], errors='coerce').fillna(0)
            min_v = pd.to_numeric(df_merged[required[1]], errors='coerce').fillna(0)
            max_v = pd.to_numeric(df_merged[required[2]], errors='coerce').fillna(0)
            data_exists = des.notna() & min_v.notna() & max_v.notna()
            mask_invalid = ~((des > min_v) & (des < max_v))
            mask_check = data_exists & mask_invalid & mask_active
            if mask_check.any(): df_merged.loc[mask_check, 'Keterangan Error'] += "Design Rate Out of Range | "

    if lift_type == "ESP":
        if "Pump Efficiency (%)" in df_merged.columns:
            eff = pd.to_numeric(df_merged["Pump Efficiency (%)"], errors='coerce')
            mask_eff = (eff > 100) & (eff.notna()) & mask_active
            df_merged.loc[mask_eff, 'Keterangan Error'] += "Efficiency > 100% | "
        for c in ["Pump Type", "Serial Name", "Automatic Gas Handler", "Shroud", "Protector", "Sensor"]:
            if c in df_merged.columns:
                mask_zero = ((df_merged[c] == 0) | (df_merged[c].astype(str) == "0")) & df_merged[c].notna() & mask_active
                df_merged.loc[mask_zero, 'Keterangan Error'] += f"{c} is 0 | "
        cols_prod = ["Dynamic Fluid Level (ft-MD)", "Pump Efficiency (%)", "Pump Optimal Design Rate / Capacity Design (BFPD)", "Pump Setting Depth (ft-MD)", "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Stages", "Frequency (Hz)", "Ampere (Amp)", "Voltage (Volt)", "Rotation (RPM)", "EQPM HP (HP)", "EQPM Rate (BFPD)", "Motor Voltage (Volt)", "Motor Amps (Amp)", "WHP", "Discharge Pressure (Psi)", "PBHP (Psi)"]
        for c in cols_prod: check_positive_vec(c, mask_producing)
        cols_non = ["Static Fluid Level (ft-MD)", "Pump Efficiency (%)", "Pump Optimal Design Rate / Capacity Design (BFPD)", "Pump Setting Depth (ft-MD)", "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Stages", "Frequency (Hz)", "Ampere (Amp)", "Voltage (Volt)", "Rotation (RPM)", "EQPM HP (HP)", "EQPM Rate (BFPD)", "Motor Voltage (Volt)", "Motor Amps (Amp)", "WHP", "Discharge Pressure (Psi)", "PBHP (Psi)"]
        for c in cols_non: check_positive_vec(c, mask_non_prod)
        check_design_range_vec()
    elif lift_type == "GL":
        cols_prod = ["Dynamic Fluid Level (ft-MD)", "Injection Fluid Pressure (Psig)", "InjectionTemperature (F)", "Number Gas Lift Valve", "Gas Injection Choke (MMSCFD)", "Rate Injection Choke (MMSCFD)", "Rate Liquid Optimization (MMSCFD)", "Gas Injection Rate (MMSCFD)", "PBHP (Psi)", "Remarks"]
        for c in cols_prod: check_positive_vec(c, mask_producing)
        cols_non = ["Static Fluid Level (ft-MD)", "Injection Fluid Pressure (Psig)", "InjectionTemperature (F)", "Number Gas Lift Valve", "Gas Injection Choke (MMSCFD)", "Rate Injection Choke (MMSCFD)", "Rate Liquid Optimization (MMSCFD)", "Gas Injection Rate (MMSCFD)", "PBHP (Psi)", "Remarks"]
        for c in cols_non: check_positive_vec(c, mask_non_prod)
    elif lift_type == "HJP":
        mask_any_active = mask_producing | mask_non_prod
        cols = ["Nozzle (Inch)", "Throat (Inch)", "Injection Fluid Pressure (Psig)", "Injection Point (ft-MD)", "PBHP (Psi)", "Remarks"]
        for c in cols: check_positive_vec(c, mask_any_active)
    elif lift_type == "HPU":
        for c in ["Pump Type", "Serial Name"]:
            if c in df_merged.columns:
                mask_zero = ((df_merged[c] == 0) | (df_merged[c].astype(str) == "0")) & df_merged[c].notna() & mask_active
                df_merged.loc[mask_zero, 'Keterangan Error'] += f"{c} is 0 | "
        cols_prod = ["Dynamic Fluid Level (ft-MD)", "Pump Efficiency (%)", "Pump Optimal Design Rate / Capacity Design (BFPD)", "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Pump Setting Depth (ft-MD)", "Stroke Length", "Stroke Per Minute (SPM)", "PBHP (Psi)", "Remarks"]
        for c in cols_prod: check_positive_vec(c, mask_producing)
        cols_non = ["Static Fluid Level (ft-MD)", "Pump Efficiency (%)", "Pump Optimal Design Rate / Capacity Design (BFPD)", "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Pump Setting Depth (ft-MD)", "Stroke Length", "Stroke Per Minute (SPM)", "PBHP (Psi)", "Remarks"]
        for c in cols_non: check_positive_vec(c, mask_non_prod)
        check_design_range_vec()
    elif lift_type == "PCP":
        for c in ["Pump Type", "Serial Name"]:
            if c in df_merged.columns:
                mask_zero = ((df_merged[c] == 0) | (df_merged[c].astype(str) == "0")) & df_merged[c].notna() & mask_active
                df_merged.loc[mask_zero, 'Keterangan Error'] += f"{c} is 0 | "
        cols_prod = ["Dynamic Fluid Level (ft-MD)", "Pump Efficiency (%)", "Pump Optimal Design Rate / Capacity Design (BFPD)", "Pump Type", "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Serial Name", "Pump Setting Depth (ft-MD)", "PBHP (Psi)", "Remarks"]
        for c in cols_prod: check_positive_vec(c, mask_producing)
        cols_non = ["Static Fluid Level (ft-MD)", "Pump Efficiency (%)", "Pump Optimal Design Rate / Capacity Design (BFPD)", "Pump Type", "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Serial Name", "Pump Setting Depth (ft-MD)", "PBHP (Psi)", "Remarks"]
        for c in cols_non: check_positive_vec(c, mask_non_prod)
        check_design_range_vec()
    elif lift_type == "SRP":
        for c in ["Pump Type", "Serial Name"]:
            if c in df_merged.columns:
                mask_zero = ((df_merged[c] == 0) | (df_merged[c].astype(str) == "0")) & df_merged[c].notna() & mask_active
                df_merged.loc[mask_zero, 'Keterangan Error'] += f"{c} is 0 | "
        cols_prod = ["Dynamic Fluid Level (ft-MD)", "Pump Efficiency (%)", "Pump Optimal Design Rate / Capacity Design (BFPD)", "Pump Type", "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Pump Setting Depth (ft-MD)", "PBHP (Psi)", "Remarks"]
        for c in cols_prod: check_positive_vec(c, mask_producing)
        cols_non = ["Static Fluid Level (ft-MD)", "Pump Efficiency (%)", "Pump Optimal Design Rate / Capacity Design (BFPD)", "Pump Type", "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Pump Setting Depth (ft-MD)", "PBHP (Psi)", "Remarks"]
        for c in cols_non: check_positive_vec(c, mask_non_prod)
        check_design_range_vec()

    df_merged['Keterangan Error'] = df_merged['Keterangan Error'].str.rstrip(" | ")
    df_merged.loc[df_merged['Keterangan Error'] == "", 'Keterangan Error'] = "OK"
    return df_merged

# --- FUNGSI KHUSUS EVENT ARTIFICIAL LIFT ---
@st.cache_data
def calculate_event_al_completeness(df):
    default_res = {"score": 0.0, "filled": 0, "expected": 0}
    if df.empty: return default_res
    if "Is Replacement" not in df.columns or "Loss Type" not in df.columns: return default_res
    is_replacement_col = df["Is Replacement"].astype(str).str.strip().str.lower()
    mask_replacement = (is_replacement_col == "yes") | (is_replacement_col == "nan") | (is_replacement_col == "") | (df["Is Replacement"].isna())
    loss_type_col = df["Loss Type"].astype(str).str.strip().str.lower()
    mask_loss = (loss_type_col == "off")
    mask_scope = mask_replacement & mask_loss
    df_scope = df[mask_scope].copy()
    if df_scope.empty: return default_res
    total_expected = len(df_scope) * len(EVENT_AL_VALIDATE_COLS)
    total_filled = df_scope[EVENT_AL_VALIDATE_COLS].replace('', pd.NA).count().sum()
    if "Ownership" in df_scope.columns and "AL Purchase Cost (USD)" in df_scope.columns:
        ownership_norm = df_scope["Ownership"].astype(str).str.strip().str.lower()
        mask_not_rent = ownership_norm != "rent"
        total_expected += mask_not_rent.sum()
        total_filled += df_scope.loc[mask_not_rent, "AL Purchase Cost (USD)"].replace('', pd.NA).notna().sum()
    if "Event Artificial Lift Type" in df_scope.columns and "Surface Recovery Cost (USD)" in df_scope.columns:
        type_norm = df_scope["Event Artificial Lift Type"].astype(str).str.strip().str.lower()
        mask_surface = type_norm == "surface"
        total_expected += mask_surface.sum()
        total_filled += df_scope.loc[mask_surface, "Surface Recovery Cost (USD)"].replace('', pd.NA).notna().sum()
    score = 0.0
    if total_expected > 0: score = (total_filled / total_expected) * 100
    return {"score": score, "filled": total_filled, "expected": total_expected}

# --- FUNGSI KHUSUS DAILY PRODUCTION ---
@st.cache_data
def validate_daily_prod_engineering(df_well, df_asset_reg):
    """
    Validasi Daily Prod - Well Production
    """
    res = df_well.copy()
    
    # Convert columns to numeric
    num_cols = ["Flowing Time", "Down Time", "Total Oil (BOPD)", "Oil (BOPD)", "Gas (MMSCFD)", "Water (BWPD)", "Cond.Form. (BCPD)"]
    for c in num_cols:
        if c in res.columns: res[c] = pd.to_numeric(res[c], errors='coerce').fillna(0)
    
    if "Production Date" in res.columns:
        res['Production Date'] = pd.to_datetime(res['Production Date'], errors='coerce')

    # Init Error Flags
    res['Error_Negative'] = False
    res['Error_Flow0_ProdNot0'] = False
    res['Error_FlowGt0_Prod0'] = False
    
    # Rule 3: Non Negative
    for c in num_cols:
        if c in res.columns:
            res.loc[res[c] < 0, 'Error_Negative'] = True
            
    # Rule 4: Flowing Time = 0 -> All Prod must be 0
    prod_cols = ["Total Oil (BOPD)", "Oil (BOPD)", "Gas (MMSCFD)", "Water (BWPD)", "Cond.Form. (BCPD)"]
    existing_prod_cols = [c for c in prod_cols if c in res.columns]
    
    if "Flowing Time" in res.columns:
        mask_flow0 = res["Flowing Time"] == 0
        # Check if any prod col is != 0
        prod_not_zero = (res[existing_prod_cols] != 0).any(axis=1)
        res.loc[mask_flow0 & prod_not_zero, 'Error_Flow0_ProdNot0'] = True
        
        # Rule 5: Flowing Time > 0 -> At least one Prod > 0
        mask_flow_gt0 = res["Flowing Time"] > 0
        # Check if ALL prod cols are 0 (or close to 0)
        prod_all_zero = (res[existing_prod_cols] == 0).all(axis=1)
        res.loc[mask_flow_gt0 & prod_all_zero, 'Error_FlowGt0_Prod0'] = True

    # Rule 1 & 2: Missing Wells & Daily Gap
    missing_wells_df = pd.DataFrame()
    res['Error_Daily_Gap'] = False
    
    if df_asset_reg is not None:
        # Filter Active Wells
        active_mask = df_asset_reg['Well Status'].str.contains("Active Well Producing|Active Well Non Production", case=False, na=False)
        asset_active = df_asset_reg[active_mask].copy()
        
        # Hirarki Check
        hier = ["Regional", "Zona", "Working Area", "Asset Operation"]
        if all(c in res.columns for c in hier) and all(c in asset_active.columns for c in hier):
            test_scope = res[hier].drop_duplicates()
            asset_in_scope = asset_active.merge(test_scope, on=hier, how='inner')
            
            # Missing Well
            prod_wells = set(res['Well'].unique())
            missing_mask = ~asset_in_scope['Well'].isin(prod_wells)
            missing_wells_df = asset_in_scope[missing_mask].copy()
            
            # Daily Gap Check
            if not res.empty and "Production Date" in res.columns:
                min_date = res['Production Date'].min()
                max_date = res['Production Date'].max()
                
                if pd.notna(min_date) and pd.notna(max_date):
                    all_dates = pd.date_range(start=min_date, end=max_date)
                    
                    well_dates = res.groupby('Well')['Production Date'].apply(set)
                    
                    active_wells_in_prod = set(asset_in_scope['Well'].unique()).intersection(prod_wells)
                    
                    gap_wells = []
                    expected_dates_set = set(all_dates)
                    
                    for w in active_wells_in_prod:
                        wd = well_dates.get(w, set())
                        if not expected_dates_set.issubset(wd):
                            gap_wells.append(w)
                            
                    res.loc[res['Well'].isin(gap_wells), 'Error_Daily_Gap'] = True

    # Build Error Msg
    res['Keterangan Error'] = ""
    res.loc[res['Error_Negative'], 'Keterangan Error'] += "⚠️ Nilai Negatif | "
    res.loc[res['Error_Flow0_ProdNot0'], 'Keterangan Error'] += "⚠️ Flow=0 tapi Produksi Ada | "
    res.loc[res['Error_FlowGt0_Prod0'], 'Keterangan Error'] += "⚠️ Flow>0 tapi Produksi Nihil | "
    res.loc[res['Error_Daily_Gap'], 'Keterangan Error'] += "⚠️ Data Harian Tidak Lengkap | "
    
    res['Keterangan Error'] = res['Keterangan Error'].str.rstrip(" | ")
    res.loc[res['Keterangan Error'] == "", 'Keterangan Error'] = "OK"
    
    return res, missing_wells_df

@st.cache_data
def check_production_reconciliation(df_well, df_asset):
    if df_well is None or df_asset is None: return None
    
    hier = ["Regional", "Zona", "Working Area", "Asset Operation", "Production Date"]
    
    for c in hier:
        if c not in df_well.columns or c not in df_asset.columns: return None
        
    # Normalize Col Names for aggregation
    well_map = {
        "Total Oil (BOPD)": "Total Oil", 
        "Oil (BOPD)": "Oil", 
        "Gas (MMSCFD)": "Gas", 
        "Cond.Form. (BCPD)": "Condensate"
    }
    asset_map = {
        "Total Oil (BOPD)": "Total Oil", 
        "Oil (BOPD)": "Oil", 
        "Gas (MMSCFD)": "Gas", 
        "Cond. Form. (BCPD)": "Condensate" 
    }
    
    # Ensure numeric
    for col in well_map.keys():
        if col in df_well.columns: df_well[col] = pd.to_numeric(df_well[col], errors='coerce').fillna(0)
    for col in asset_map.keys():
        if col in df_asset.columns: df_asset[col] = pd.to_numeric(df_asset[col], errors='coerce').fillna(0)

    # Aggregation
    well_agg = df_well.groupby(hier)[list(well_map.keys())].sum().reset_index()
    well_agg = well_agg.rename(columns=well_map)
    
    asset_sub = df_asset[hier + list(asset_map.keys())].copy()
    asset_sub = asset_sub.rename(columns=asset_map)
    
    # Merge
    merged = pd.merge(well_agg, asset_sub, on=hier, suffixes=('_Well', '_Asset'), how='outer')
    
    # Calculate Diff
    metrics = ["Total Oil", "Oil", "Gas", "Condensate"]
    for m in metrics:
        merged[f"{m}_Well"] = merged[f"{m}_Well"].fillna(0)
        merged[f"{m}_Asset"] = merged[f"{m}_Asset"].fillna(0)
        merged[f"Diff_{m}"] = merged[f"{m}_Well"] - merged[f"{m}_Asset"]
        
    # Identify Mismatch (Allow small float tolerance)
    tol = 0.01
    merged['Status'] = 'Match'
    mismatch_mask = (merged[[f"Diff_{m}" for m in metrics]].abs() > tol).any(axis=1)
    merged.loc[mismatch_mask, 'Status'] = 'Mismatch'
    
    return merged

# ==========================================
# 4. ANTARMUKA PENGGUNA (MAIN UI)
# ==========================================

def main():
    st.sidebar.title("🛢️ Menu Aplikasi")
    
    main_menu = st.sidebar.selectbox(
        "Pilih Modul Validasi:", 
        ["Asset Register", "Production Well Test", "Well Parameter (Well On)", 
         "Event Artificial Lift (Well Off)", "Daily Production Data"]
    )
    
    # --- MODUL 1: ASSET REGISTER ---
    if main_menu == "Asset Register":
        st.title("📂 Modul 1: Asset Register")
        sub_menu = st.sidebar.radio("Sub-Menu Asset:", list(ASSET_CONFIG.keys()))
        config = ASSET_CONFIG[sub_menu]
        uploaded_file = st.file_uploader(f"Upload Excel ({sub_menu})", type=['xlsx', 'xls'])
        if uploaded_file:
            df = load_excel_file(uploaded_file)
            if df is not None:
                if sub_menu == "Well":
                    st.session_state['asset_register_df'] = df
                    st.toast("Data Well tersimpan!", icon="💾")
                score = calculate_simple_completeness(df, config['validate_columns'])
                st.metric("Completeness", f"{score:.2f}%")
                display_dataframe_optimized(df, config['validate_columns'])

    # --- MODUL 2: PRODUCTION WELL TEST ---
    elif main_menu == "Production Well Test":
        st.title("🧪 Modul 2: Production Well Test")
        uploaded_test = st.file_uploader("Upload Excel Well Test", type=['xlsx', 'xls'])
        if uploaded_test:
            df_test = load_excel_file(uploaded_test)
            if df_test is not None:
                st.subheader("2. Validasi Kaidah Engineering")
                df_eng, missing_wells_df = validate_engineering_rules(df_test, st.session_state['asset_register_df'])
                if not missing_wells_df.empty:
                    st.warning(f"⚠️ Ditemukan **{len(missing_wells_df)} Sumur Active Producing** yang tidak memiliki data test:")
                    st.dataframe(missing_wells_df, use_container_width=True)
                
                st.write("### 🚨 Data Bermasalah")
                df_problems = df_eng[df_eng['Keterangan Error'] != "OK"].copy()
                
                if df_problems.empty:
                    st.success("✅ Tidak ditemukan pelanggaran.")
                else:
                    helpers = ['Rule1_Pass', 'Rule2_Pass', 'Rule3_Pass', 'Rule4_Pass']
                    df_display = df_problems.drop(columns=[c for c in helpers if c in df_problems.columns], errors='ignore')
                    cols = ['Keterangan Error'] + [c for c in df_display.columns if c != 'Keterangan Error']
                    df_display = df_display[cols]
                    display_dataframe_optimized(df_display, ['Keterangan Error'], use_highlight=True)
                    csv_err = df_display.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Data Bermasalah", csv_err, "Engineering_Issues.csv", "text/csv")

    # --- MODUL 3: WELL PARAMETER (WELL ON) ---
    elif main_menu == "Well Parameter (Well On)":
        st.title("⚙️ Modul 3: Well Parameter (Well On)")
        lift_type = st.sidebar.radio("Pilih Artificial Lift:", list(WELL_PARAM_CONFIG.keys()))
        uploaded_param = st.file_uploader(f"Upload Excel {lift_type}", type=['xlsx', 'xls'])
        if uploaded_param:
            df_param = load_excel_file(uploaded_param)
            if df_param is not None:
                st.subheader("Filter Data")
                df_filt = df_param.copy()
                cols_filt = st.columns(len(WELL_PARAM_HIERARCHY))
                for i, col in enumerate(WELL_PARAM_HIERARCHY):
                    if col in df_filt.columns:
                        df_filt[col] = df_filt[col].astype(str).replace('nan', '')
                        opts = ["Semua"] + sorted(df_filt[col].unique().tolist())
                        sel = cols_filt[i].selectbox(col, opts, key=f"wp_{col}")
                        if sel != "Semua":
                            df_filt = df_filt[df_filt[col] == sel]
                    else:
                        cols_filt[i].text(f"{col} (N/A)")
                
                res_total = calculate_well_param_completeness(df_filt, lift_type)
                if res_total:
                    st.metric("Total Completeness (Active Wells)", f"{res_total['score']:.2f}%")
                
                df_eng_param = validate_well_parameter_rules(df_filt, lift_type)
                st.write("### 🚨 Data Bermasalah")
                df_problems = df_eng_param[df_eng_param['Keterangan Error'] != "OK"].copy()
                
                if df_problems.empty:
                    st.success("✅ Tidak ditemukan pelanggaran.")
                else:
                    if 'Status_Norm' in df_problems.columns:
                        df_problems = df_problems.drop(columns=['Status_Norm'])
                    cols = ['Keterangan Error'] + [c for c in df_problems.columns if c != 'Keterangan Error']
                    df_display = df_problems[cols]
                    display_dataframe_optimized(df_display, ['Keterangan Error'], use_highlight=True)
                    csv_err = df_display.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Data Bermasalah", csv_err, f"Engineering_Issues_{lift_type}.csv", "text/csv")

    # --- MODUL 4: EVENT ARTIFICIAL LIFT (WELL OFF) ---
    elif main_menu == "Event Artificial Lift (Well Off)":
        st.title("📉 Modul 4: Event Artificial Lift (Well Off)")
        uploaded_event = st.file_uploader("Upload Excel Event AL", type=['xlsx', 'xls'])
        if uploaded_event:
            df_event = load_excel_file(uploaded_event)
            if df_event is not None:
                res_total = calculate_event_al_completeness(df_event)
                st.metric("Completeness", f"{res_total['score']:.2f}%")

    # --- MODUL 5: DAILY PRODUCTION DATA ---
    elif main_menu == "Daily Production Data":
        st.title("📊 Modul 5: Daily Production Data")
        sub_menu = st.sidebar.radio("Sub-Menu:", ["Working Area Production", "Asset Operation Production", "Well Production", "Production Reconciliation"])
        
        if sub_menu != "Production Reconciliation":
            config = DAILY_PROD_CONFIG[sub_menu]
            uploaded_prod = st.file_uploader(f"Upload Excel {sub_menu}", type=['xlsx', 'xls'])
            if uploaded_prod:
                df_prod = load_excel_file(uploaded_prod)
                if df_prod is not None:
                    if sub_menu == "Well Production":
                        st.session_state['daily_well_prod_df'] = df_prod
                    elif sub_menu == "Asset Operation Production":
                        st.session_state['daily_asset_prod_df'] = df_prod
                    
                    st.subheader("Filter Data")
                    df_filt = df_prod.copy()
                    cols_filt = st.columns(len(config['hierarchy']))
                    for i, col in enumerate(config['hierarchy']):
                        if col in df_filt.columns:
                            df_filt[col] = df_filt[col].astype(str).replace('nan', '')
                            opts = ["Semua"] + sorted(df_filt[col].unique().tolist())
                            sel = cols_filt[i].selectbox(col, opts, key=f"dp_{col}")
                            if sel != "Semua":
                                df_filt = df_filt[df_filt[col] == sel]
                        else:
                            cols_filt[i].text(f"{col} (N/A)")
                    
                    st.subheader("1. Validasi Kelengkapan Data")
                    score = calculate_simple_completeness(df_filt, config['validate'])
                    st.metric("Completeness", f"{score:.2f}%")
                    
                    if sub_menu == "Well Production":
                        st.subheader("2. Validasi Kaidah Engineering")
                        df_eng, missing_wells = validate_daily_prod_engineering(df_filt, st.session_state['asset_register_df'])
                        if not missing_wells.empty:
                            st.warning(f"⚠️ Ditemukan **{len(missing_wells)} Sumur Active** yang tidak memiliki data produksi harian di hirarki ini:")
                            st.dataframe(missing_wells, use_container_width=True)
                        st.write("### 🚨 Data Bermasalah")
                        df_probs = df_eng[df_eng['Keterangan Error'] != "OK"].copy()
                        if df_probs.empty:
                            st.success("✅ Tidak ditemukan pelanggaran rules.")
                        else:
                            helpers = ['Error_Negative', 'Error_Flow0_ProdNot0', 'Error_FlowGt0_Prod0', 'Error_Daily_Gap']
                            df_display = df_probs.drop(columns=[c for c in helpers if c in df_probs.columns])
                            cols = ['Keterangan Error'] + [c for c in df_display.columns if c != 'Keterangan Error']
                            df_display = df_display[cols]
                            display_dataframe_optimized(df_display, ['Keterangan Error'], use_highlight=True)
                            csv_err = df_display.to_csv(index=False).encode('utf-8')
                            st.download_button("📥 Download Data Bermasalah", csv_err, "Daily_Prod_Issues.csv", "text/csv")
                    
                    with st.expander("Detail Data Mentah"):
                        display_dataframe_optimized(df_filt, config['validate'])

        else: # Production Reconciliation
            st.subheader("⚖️ Production Reconciliation")
            df_well = st.session_state['daily_well_prod_df']
            df_asset = st.session_state['daily_asset_prod_df']
            
            if df_well is None or df_asset is None:
                st.warning("⚠️ Harap upload file 'Well Production' dan 'Asset Operation Production' di sub-menu masing-masing terlebih dahulu.")
            else:
                recon_df = check_production_reconciliation(df_well, df_asset)
                if recon_df is None:
                    st.error("Gagal melakukan rekonsiliasi. Pastikan kolom Hirarki dan Production Date ada di kedua file.")
                else:
                    st.subheader("Filter Data Rekonsiliasi")
                    recon_filt = recon_df.copy()
                    recon_hier = ["Regional", "Zona", "Working Area", "Asset Operation"]
                    cols_filt = st.columns(len(recon_hier))
                    for i, col in enumerate(recon_hier):
                        if col in recon_filt.columns:
                            recon_filt[col] = recon_filt[col].astype(str).replace('nan', '')
                            opts = ["Semua"] + sorted(recon_filt[col].unique().tolist())
                            sel = cols_filt[i].selectbox(col, opts, key=f"recon_{col}")
                            if sel != "Semua":
                                recon_filt = recon_filt[recon_filt[col] == sel]
                        else:
                            cols_filt[i].text(f"{col} (N/A)")
                    
                    total_rows = len(recon_filt)
                    mismatch_rows = len(recon_filt[recon_filt['Status'] == 'Mismatch'])
                    match_rows = total_rows - mismatch_rows
                    c1, c2 = st.columns(2)
                    c1.metric("Data Match", f"{match_rows} Baris", delta="OK")
                    c2.metric("Data Mismatch", f"{mismatch_rows} Baris", delta_color="inverse")
                    st.write("### 📋 Detail Rekonsiliasi")
                    df_show = recon_filt.sort_values('Status', ascending=False)
                    def color_status(val):
                        color = '#ffcccc' if val == 'Mismatch' else '#ccffcc'
                        return f'background-color: {color}'
                    try:
                        st.dataframe(df_show.style.applymap(color_status, subset=['Status']), use_container_width=True)
                    except:
                        st.dataframe(df_show, use_container_width=True)
                    csv_recon = recon_filt.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Hasil Rekonsiliasi", csv_recon, "Production_Reconciliation.csv", "text/csv")

if __name__ == "__main__":
    main()
