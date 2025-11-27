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

# Inisialisasi Session State (Masih ada untuk modul lain jika dibutuhkan)
if 'asset_register_df' not in st.session_state:
    st.session_state['asset_register_df'] = None

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

# Kolom Umum (Admin) yang pasti ada di setiap file input (Point B)
WP_ADMIN_COLS = [
    "Regional", "Zona", "Working Area", "Asset Operation", "Entity ID",
    "Well", "Well Status", "Skin Status", "Lifting Method", "Reservoir Name",
    "Main Fluid Type", "Latitude", "Longitude", "POP Date Target",
    "POP Date Actual", "Profile Summary"
]

# Definisi Kolom Input (Point B) & Kolom Validasi Kelengkapan (Point D)
WELL_PARAM_CONFIG = {
    "ESP": {
        "input_cols": WP_ADMIN_COLS + [
            "Ownership", "Manufacturer", "Brand", "Supplier / Vendor",
            "Dynamic Fluid Level (ft-MD)", "Static Fluid Level (ft-MD)", "Pump Efficiency (%)",
            "Pump Optimal Design Rate / Capacity Design (BFPD)", "Pump Setting Depth (ft-MD)",
            "Pump Type", "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Serial Name",
            "Stages", "Frequency (Hz)", "Ampere (Amp)", "Voltage (Volt)", "Rotation (RPM)",
            "EQPM HP (HP)", "EQPM Rate (BFPD)", "Motor Voltage (Volt)", "Motor Amps (Amp)",
            "Automatic Gas Handler", "Shroud", "Protector", "Sensor", "Start Date",
            "DHP Date (dd/MM/yyyy)", "WHP", "Discharge Pressure (Psi)", "PBHP (Psi)", "Remarks"
        ],
        "check_cols": [
            "Ownership", "Manufacturer", "Brand", "Supplier / Vendor", "Pump Efficiency (%)",
            "Pump Optimal Design Rate / Capacity Design (BFPD)", "Pump Setting Depth (ft-MD)",
            "Pump Type", "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Serial Name",
            "Stages", "Frequency (Hz)", "Ampere (Amp)", "Voltage (Volt)", "Rotation (RPM)",
            "EQPM HP (HP)", "EQPM Rate (BFPD)", "Motor Voltage (Volt)", "Motor Amps (Amp)",
            "Automatic Gas Handler", "Shroud", "Protector", "Sensor", "Start Date",
            "DHP Date (dd/MM/yyyy)", "WHP", "Discharge Pressure (Psi)", "PBHP (Psi)", "Remarks"
        ],
        "conditional": {"Dynamic": "Dynamic Fluid Level (ft-MD)", "Static": "Static Fluid Level (ft-MD)"}
    },
    "GL": {
        "input_cols": WP_ADMIN_COLS + [
            "Ownership", "Manufacturer", "Brand", "Supplier / Vendor",
            "Dynamic Fluid Level (ft-MD)", "Static Fluid Level (ft-MD)",
            "Injection Fluid Pressure (Psig)", "InjectionTemperature (F)", "Number Gas Lift Valve",
            "Gas Injection Choke (MMSCFD)", "Rate Injection Choke (MMSCFD)",
            "Rate Liquid Optimization (MMSCFD)", "Gas Injection Rate (MMSCFD)", "PBHP (Psi)", "Remarks"
        ],
        "check_cols": [
            "Ownership", "Manufacturer", "Brand", "Supplier / Vendor",
            "Injection Fluid Pressure (Psig)", "InjectionTemperature (F)", "Number Gas Lift Valve",
            "Gas Injection Choke (MMSCFD)", "Rate Injection Choke (MMSCFD)",
            "Rate Liquid Optimization (MMSCFD)", "Gas Injection Rate (MMSCFD)", "PBHP (Psi)", "Remarks"
        ],
        "conditional": {"Dynamic": "Dynamic Fluid Level (ft-MD)", "Static": "Static Fluid Level (ft-MD)"}
    },
    "HJP": {
        "input_cols": WP_ADMIN_COLS + [
            "Ownership", "Manufacturer", "Brand", "Supplier / Vendor",
            "Nozzle (Inch)", "Throat (Inch)", "Injection Fluid Pressure (Psig)",
            "Injection Point (ft-MD)", "PBHP (Psi)", "Remarks"
        ],
        "check_cols": [
            "Ownership", "Manufacturer", "Brand", "Supplier / Vendor",
            "Nozzle (Inch)", "Throat (Inch)", "Injection Fluid Pressure (Psig)",
            "Injection Point (ft-MD)", "PBHP (Psi)", "Remarks"
        ],
        "conditional": {} # Tidak ada Dynamic/Static di list validasi HJP
    },
    "HPU": {
        "input_cols": WP_ADMIN_COLS + [
            "Ownership", "Manufacturer", "Brand", "Supplier / Vendor",
            "Dynamic Fluid Level (ft-MD)", "Static Fluid Level (ft-MD)", "Pump Efficiency (%)",
            "Pump Optimal Design Rate / Capacity Design (BFPD)", "Pump Type",
            "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Serial Name",
            "Pump Setting Depth (ft-MD)", "Stroke Length", "Stroke Per Minute (SPM)", "PBHP (Psi)", "Remarks"
        ],
        "check_cols": [
            "Ownership", "Manufacturer", "Brand", "Supplier / Vendor", "Pump Efficiency (%)",
            "Pump Optimal Design Rate / Capacity Design (BFPD)", "Pump Type",
            "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Serial Name",
            "Pump Setting Depth (ft-MD)", "Stroke Length", "Stroke Per Minute (SPM)", "PBHP (Psi)", "Remarks"
        ],
        "conditional": {"Dynamic": "Dynamic Fluid Level (ft-MD)", "Static": "Static Fluid Level (ft-MD)"}
    },
    "PCP": {
        "input_cols": WP_ADMIN_COLS + [
            "Ownership", "Manufacturer", "Brand", "Supplier / Vendor",
            "Dynamic Fluid Level (ft-MD)", "Static Fluid Level (ft-MD)", "Pump Efficiency (%)",
            "Pump Optimal Design Rate / Capacity Design (BFPD)", "Pump Type",
            "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Serial Name",
            "Pump Setting Depth (ft-MD)", "PBHP (Psi)", "Remarks"
        ],
        "check_cols": [
            "Ownership", "Manufacturer", "Brand", "Supplier / Vendor", "Pump Efficiency (%)",
            "Pump Optimal Design Rate / Capacity Design (BFPD)", "Pump Type",
            "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Serial Name",
            "Pump Setting Depth (ft-MD)", "PBHP (Psi)", "Remarks"
        ],
        "conditional": {"Dynamic": "Dynamic Fluid Level (ft-MD)", "Static": "Static Fluid Level (ft-MD)"}
    },
    "SRP": {
        "input_cols": WP_ADMIN_COLS + [
            "Ownership", "Manufacturer", "Brand", "Supplier / Vendor",
            "Stroke Length", "Stroke Per Minute (SPM)",
            "Dynamic Fluid Level (ft-MD)", "Static Fluid Level (ft-MD)", "Pump Efficiency (%)",
            "Pump Optimal Design Rate / Capacity Design (BFPD)", "Pump Type",
            "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Pump Setting Depth (ft-MD)",
            "PBHP (Psi)", "Remarks"
        ],
        "check_cols": [
            "Ownership", "Manufacturer", "Brand", "Supplier / Vendor", "Stroke Length", "Stroke Per Minute (SPM)",
            "Pump Efficiency (%)", "Pump Optimal Design Rate / Capacity Design (BFPD)", "Pump Type",
            "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Pump Setting Depth (ft-MD)",
            "PBHP (Psi)", "Remarks"
        ],
        "conditional": {"Dynamic": "Dynamic Fluid Level (ft-MD)", "Static": "Static Fluid Level (ft-MD)"}
    }
}

WELL_PARAM_HIERARCHY = ["Regional", "Zona", "Working Area", "Asset Operation", "Well"]

# ==========================================
# 3. FUNGSI-FUNGSI LOGIKA (OPTIMIZED BACKEND)
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
                # Hanya highlight kolom yang relevan jika ada
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
    # Hanya cek kolom yang ada di dataframe
    valid_cols = [c for c in target_cols if c in df.columns]
    temp_df = df[valid_cols].replace('', pd.NA)
    return temp_df.isnull().sum().reset_index(name='Jumlah Kosong').rename(columns={'index': 'Nama Kolom'})

# --- FUNGSI KHUSUS WELL TEST (VECTORIZED) ---

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
        res[col] = pd.to_numeric(res[col], errors='coerce').fillna(0)

    is_producing = (res["Oil (BOPD)"] > 0) | (res["Water (BWPD)"] > 0) | (res["Gas (MMSCFD)"] > 0) | (res["Condensate (BCPD)"] > 0) | (res["Fluid (BFPD)"] > 0)
    res['Rule2_Pass'] = ~((res["Test Duration(Hours)"] > 0) & (~is_producing))
    
    produces_something = (res["Oil (BOPD)"] > 0) | (res["Water (BWPD)"] > 0) | (res["Gas (MMSCFD)"] > 0) | (res["Condensate (BCPD)"] > 0)
    res['Rule3_Pass'] = ~((produces_something) & (res["Test Duration(Hours)"] <= 0))
    res['Rule4_Pass'] = (res[num_cols] >= 0).all(axis=1)

    res['Rule1_Pass'] = True 
    check_rule1_active = False
    
    if df_asset is not None and not df_asset.empty:
        if 'Well Status' in df_asset.columns and 'Well' in df_asset.columns:
            check_rule1_active = True
            active_wells = set(df_asset[df_asset['Well Status'].str.contains("Active Well Producing", case=False, na=False)]['Well'].unique())
            res['Test Date'] = pd.to_datetime(res['Test Date (dd/mm/yyyy)'], format='%d/%m/%Y', errors='coerce')
            max_date = datetime.now() if res['Test Date'].dropna().empty else res['Test Date'].max()
            cutoff_date = max_date - pd.timedelta_range(start='1 days', periods=1, freq='90D')[0]
            recent_tests = set(res[res['Test Date'] >= cutoff_date]['Well'].unique())
            
            is_active_well = res['Well'].isin(active_wells)
            is_recent_test = res['Well'].isin(recent_tests)
            res['Rule1_Pass'] = ~is_active_well | (is_active_well & is_recent_test)

    res['Keterangan Error'] = ""
    if check_rule1_active:
        res.loc[~res['Rule1_Pass'], 'Keterangan Error'] += "⚠️ Active Well tidak ada test >3 bulan | "
    res.loc[~res['Rule2_Pass'], 'Keterangan Error'] += "⚠️ Durasi > 0 tapi Produksi Nihil | "
    res.loc[~res['Rule3_Pass'], 'Keterangan Error'] += "⚠️ Produksi Ada tapi Durasi 0/Kosong | "
    res.loc[~res['Rule4_Pass'], 'Keterangan Error'] += "⚠️ Terdapat Nilai Negatif | "
    
    res['Keterangan Error'] = res['Keterangan Error'].str.rstrip(" | ")
    res.loc[res['Keterangan Error'] == "", 'Keterangan Error'] = "OK"
    return res

# --- FUNGSI KHUSUS WELL PARAMETER (WELL ON) ---

@st.cache_data
def calculate_well_param_completeness(df, lift_type):
    """
    Hitung kelengkapan sesuai aturan: Hanya untuk Active Well.
    Kolom Dynamic/Static wajib bersyarat.
    """
    if df.empty: return 0.0
    
    config = WELL_PARAM_CONFIG[lift_type]
    base_cols = config['check_cols']
    cond_map = config['conditional'] # Dictionary nama kolom Dynamic/Static
    
    # 1. Filter hanya Active Well
    df['Status_Norm'] = df['Well Status'].astype(str).str.strip()
    mask_active = df['Status_Norm'].str.contains("Active Well Producing|Active Well Non Production", case=False, na=False)
    
    df_active = df[mask_active].copy()
    if df_active.empty: return 0.0 # Jika tidak ada sumur aktif, return 0 (atau 100? Biasnya 0 karena tidak ada data yg bisa dicek)

    mask_producing = df_active['Status_Norm'].str.contains("Active Well Producing", case=False, na=False)
    mask_non_prod = df_active['Status_Norm'].str.contains("Active Well Non Production", case=False, na=False)
    
    # 2. Hitung Basic Columns (Berlaku untuk SEMUA Active Well)
    total_expected = len(df_active) * len(base_cols)
    total_filled = df_active[base_cols].replace('', pd.NA).count().sum()
    
    # 3. Hitung Conditional Columns
    if 'Dynamic' in cond_map:
        dyn_col = cond_map['Dynamic']
        # Wajib untuk Producing
        total_expected += mask_producing.sum()
        total_filled += df_active.loc[mask_producing, dyn_col].replace('', pd.NA).notna().sum()
        
    if 'Static' in cond_map:
        stat_col = cond_map['Static']
        # Wajib untuk Non Production
        total_expected += mask_non_prod.sum()
        total_filled += df_active.loc[mask_non_prod, stat_col].replace('', pd.NA).notna().sum()
        
    if total_expected == 0: return 0.0
    return (total_filled / total_expected) * 100

@st.cache_data
def validate_well_parameter_rules(df_input, lift_type):
    """
    Validasi Engineering Well Parameter (Tanpa Asset Register).
    Menggunakan 'Well Status' dari file input.
    """
    df_merged = df_input.copy()
    
    # Pastikan kolom status ada
    if 'Well Status' not in df_merged.columns:
        df_merged['Keterangan Error'] = "CRITICAL: Kolom 'Well Status' tidak ditemukan."
        return df_merged

    df_merged['Well Status'] = df_merged['Well Status'].fillna("Unknown")
    df_merged['Status_Norm'] = df_merged['Well Status'].astype(str).str.strip()
    df_merged['Keterangan Error'] = ""

    mask_producing = df_merged['Status_Norm'].str.contains("Active Well Producing", case=False, na=False)
    mask_non_prod = df_merged['Status_Norm'].str.contains("Active Well Non Production", case=False, na=False)

    # Helper Vectorized Checks
    def check_positive_vec(col_name, mask_rows):
        if col_name in df_merged.columns:
            val_col = pd.to_numeric(df_merged[col_name], errors='coerce')
            # Error jika: (Value <= 0 OR NaN) AND Row Target
            mask_invalid = (val_col <= 0) | (val_col.isna())
            target_invalid = mask_rows & mask_invalid
            if target_invalid.any():
                df_merged.loc[target_invalid, 'Keterangan Error'] += f"{col_name} <= 0 | "

    def check_design_range_vec():
        required = ["Pump Optimal Design Rate / Capacity Design (BFPD)", "Min BLPD (BFPD)", "Max BLPD (BFPD)"]
        if all(c in df_merged.columns for c in required):
            des = pd.to_numeric(df_merged[required[0]], errors='coerce').fillna(0)
            min_v = pd.to_numeric(df_merged[required[1]], errors='coerce').fillna(0)
            max_v = pd.to_numeric(df_merged[required[2]], errors='coerce').fillna(0)
            
            # Logic: Min < Design < Max
            mask_invalid = ~((des > min_v) & (des < max_v))
            # Cek hanya jika design rate > 0
            mask_check = (des > 0) & mask_invalid
            if mask_check.any():
                df_merged.loc[mask_check, 'Keterangan Error'] += "Design Rate Out of Range | "

    # --- LOGIKA SPESIFIK ---
    
    if lift_type == "ESP":
        if "Pump Efficiency (%)" in df_merged.columns:
            eff = pd.to_numeric(df_merged["Pump Efficiency (%)"], errors='coerce')
            mask_eff = eff > 100
            df_merged.loc[mask_eff, 'Keterangan Error'] += "Efficiency > 100% | "
        
        # Kolom string/mix yang tidak boleh 0
        for c in ["Pump Type", "Serial Name", "Automatic Gas Handler", "Shroud", "Protector", "Sensor"]:
            if c in df_merged.columns:
                mask_zero = (df_merged[c] == 0) | (df_merged[c].astype(str) == "0")
                df_merged.loc[mask_zero, 'Keterangan Error'] += f"{c} is 0 | "

        cols_prod = [
            "Dynamic Fluid Level (ft-MD)", "Pump Efficiency (%)", "Pump Optimal Design Rate / Capacity Design (BFPD)",
            "Pump Setting Depth (ft-MD)", "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Stages",
            "Frequency (Hz)", "Ampere (Amp)", "Voltage (Volt)", "Rotation (RPM)", "EQPM HP (HP)",
            "EQPM Rate (BFPD)", "Motor Voltage (Volt)", "Motor Amps (Amp)", "WHP", "Discharge Pressure (Psi)", "PBHP (Psi)"
        ]
        for c in cols_prod: check_positive_vec(c, mask_producing)
        
        cols_non = [
            "Static Fluid Level (ft-MD)", "Pump Efficiency (%)", "Pump Optimal Design Rate / Capacity Design (BFPD)",
            "Pump Setting Depth (ft-MD)", "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Stages",
            "Frequency (Hz)", "Ampere (Amp)", "Voltage (Volt)", "Rotation (RPM)", "EQPM HP (HP)",
            "EQPM Rate (BFPD)", "Motor Voltage (Volt)", "Motor Amps (Amp)", "WHP", "Discharge Pressure (Psi)", "PBHP (Psi)"
        ]
        for c in cols_non: check_positive_vec(c, mask_non_prod)
        check_design_range_vec()

    elif lift_type == "GL":
        cols_prod = [
            "Dynamic Fluid Level (ft-MD)", "Injection Fluid Pressure (Psig)", "InjectionTemperature (F)",
            "Number Gas Lift Valve", "Gas Injection Choke (MMSCFD)", "Rate Injection Choke (MMSCFD)",
            "Rate Liquid Optimization (MMSCFD)", "Gas Injection Rate (MMSCFD)", "PBHP (Psi)", "Remarks"
        ]
        for c in cols_prod: check_positive_vec(c, mask_producing)
        
        cols_non = [
            "Static Fluid Level (ft-MD)", "Injection Fluid Pressure (Psig)", "InjectionTemperature (F)",
            "Number Gas Lift Valve", "Gas Injection Choke (MMSCFD)", "Rate Injection Choke (MMSCFD)",
            "Rate Liquid Optimization (MMSCFD)", "Gas Injection Rate (MMSCFD)", "PBHP (Psi)", "Remarks"
        ]
        for c in cols_non: check_positive_vec(c, mask_non_prod)

    elif lift_type == "HJP":
        mask_any_active = mask_producing | mask_non_prod
        cols = ["Nozzle (Inch)", "Throat (Inch)", "Injection Fluid Pressure (Psig)", "Injection Point (ft-MD)", "PBHP (Psi)", "Remarks"]
        for c in cols: check_positive_vec(c, mask_any_active)

    elif lift_type == "HPU":
        for c in ["Pump Type", "Serial Name"]:
            if c in df_merged.columns:
                mask_zero = (df_merged[c] == 0) | (df_merged[c].astype(str) == "0")
                df_merged.loc[mask_zero, 'Keterangan Error'] += f"{c} is 0 | "
        
        cols_prod = [
            "Dynamic Fluid Level (ft-MD)", "Pump Efficiency (%)", "Pump Optimal Design Rate / Capacity Design (BFPD)",
            "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Pump Setting Depth (ft-MD)",
            "Stroke Length", "Stroke Per Minute (SPM)", "PBHP (Psi)", "Remarks"
        ]
        for c in cols_prod: check_positive_vec(c, mask_producing)

        cols_non = [
            "Static Fluid Level (ft-MD)", "Pump Efficiency (%)", "Pump Optimal Design Rate / Capacity Design (BFPD)",
            "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Pump Setting Depth (ft-MD)",
            "Stroke Length", "Stroke Per Minute (SPM)", "PBHP (Psi)", "Remarks"
        ]
        for c in cols_non: check_positive_vec(c, mask_non_prod)
        check_design_range_vec()

    elif lift_type == "PCP":
        for c in ["Pump Type", "Serial Name"]:
            if c in df_merged.columns:
                mask_zero = (df_merged[c] == 0) | (df_merged[c].astype(str) == "0")
                df_merged.loc[mask_zero, 'Keterangan Error'] += f"{c} is 0 | "
        
        cols_prod = [
            "Dynamic Fluid Level (ft-MD)", "Pump Efficiency (%)", "Pump Optimal Design Rate / Capacity Design (BFPD)",
            "Pump Type", "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Serial Name",
            "Pump Setting Depth (ft-MD)", "PBHP (Psi)", "Remarks"
        ]
        for c in cols_prod: check_positive_vec(c, mask_producing)

        cols_non = [
            "Static Fluid Level (ft-MD)", "Pump Efficiency (%)", "Pump Optimal Design Rate / Capacity Design (BFPD)",
            "Pump Type", "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Serial Name",
            "Pump Setting Depth (ft-MD)", "PBHP (Psi)", "Remarks"
        ]
        for c in cols_non: check_positive_vec(c, mask_non_prod)
        check_design_range_vec()
        
    elif lift_type == "SRP":
        for c in ["Pump Type", "Serial Name"]:
            if c in df_merged.columns:
                mask_zero = (df_merged[c] == 0) | (df_merged[c].astype(str) == "0")
                df_merged.loc[mask_zero, 'Keterangan Error'] += f"{c} is 0 | "

        cols_prod = [
            "Dynamic Fluid Level (ft-MD)", "Pump Efficiency (%)", "Pump Optimal Design Rate / Capacity Design (BFPD)",
            "Pump Type", "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Pump Setting Depth (ft-MD)",
            "PBHP (Psi)", "Remarks"
        ]
        for c in cols_prod: check_positive_vec(c, mask_producing)
        
        cols_non = [
            "Static Fluid Level (ft-MD)", "Pump Efficiency (%)", "Pump Optimal Design Rate / Capacity Design (BFPD)",
            "Pump Type", "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Pump Setting Depth (ft-MD)",
            "PBHP (Psi)", "Remarks"
        ]
        for c in cols_non: check_positive_vec(c, mask_non_prod)
        check_design_range_vec()

    # Final Cleanup
    df_merged['Keterangan Error'] = df_merged['Keterangan Error'].str.rstrip(" | ")
    df_merged.loc[df_merged['Keterangan Error'] == "", 'Keterangan Error'] = "OK"
    return df_merged

# ==========================================
# 4. ANTARMUKA PENGGUNA (MAIN UI)
# ==========================================

def main():
    st.sidebar.title("🛢️ Menu Aplikasi")
    
    main_menu = st.sidebar.selectbox(
        "Pilih Modul Validasi:", 
        ["Asset Register", "Production Well Test", "Well Parameter (Well On)"]
    )
    
    # ---------------------------------------------------------
    # MODUL 1: ASSET REGISTER
    # ---------------------------------------------------------
    if main_menu == "Asset Register":
        st.title("📂 Modul 1: Asset Register")
        
        sub_menu = st.sidebar.radio("Sub-Menu Asset:", list(ASSET_CONFIG.keys()))
        config = ASSET_CONFIG[sub_menu]
        
        uploaded_file = st.file_uploader(f"Upload Excel ({sub_menu})", type=['xlsx', 'xls'])
        
        col_info, col_req = st.columns([1, 2])
        with col_info: st.info("Baris 1 Excel harus Header.")
        with col_req: 
            with st.expander("Lihat Kolom Wajib"): st.code(", ".join(config['all_columns']))
        
        if uploaded_file:
            df = load_excel_file(uploaded_file)
            
            if df is not None:
                missing = [c for c in config['all_columns'] if c not in df.columns]
                if missing:
                    st.error(f"Kolom hilang: {missing}")
                else:
                    st.success(f"Data {sub_menu} dimuat: {len(df)} baris.")
                    
                    if sub_menu == "Well":
                        st.session_state['asset_register_df'] = df
                        st.toast("Data Well tersimpan!", icon="💾")
                    
                    df_filt = df.copy()
                    cols_filt = st.columns(len(config['filter_hierarchy']))
                    for i, col in enumerate(config['filter_hierarchy']):
                        df_filt[col] = df_filt[col].astype(str).replace('nan', '')
                        opts = ["Semua"] + sorted(df_filt[col].unique().tolist())
                        sel = cols_filt[i].selectbox(col, opts, key=f"ar_{col}")
                        if sel != "Semua":
                            df_filt = df_filt[df_filt[col] == sel]
                    
                    score = calculate_simple_completeness(df_filt, config['validate_columns'])
                    st.metric("Completeness", f"{score:.2f}%")
                    
                    st.markdown("##### Detail Data")
                    display_dataframe_optimized(df_filt, config['validate_columns'])

                    missing_summary = get_missing_details(df_filt, config['validate_columns'])
                    if missing_summary['Jumlah Kosong'].sum() > 0:
                        st.warning("⚠️ **Rincian Kekurangan Data:**")
                        st.dataframe(missing_summary[missing_summary['Jumlah Kosong'] > 0], use_container_width=False)
                    else:
                        st.success("✅ Semua kolom wajib telah terisi penuh.")

    # ---------------------------------------------------------
    # MODUL 2: PRODUCTION WELL TEST
    # ---------------------------------------------------------
    elif main_menu == "Production Well Test":
        st.title("🧪 Modul 2: Production Well Test")
        
        asset_ready = st.session_state['asset_register_df'] is not None
        if not asset_ready: st.warning("⚠️ Data 'Well' di Asset Register belum diupload.")
        else: st.success("✅ Terhubung dengan data referensi Asset Register.")

        uploaded_test = st.file_uploader("Upload Excel Well Test", type=['xlsx', 'xls'])
        
        with st.expander("Lihat Spesifikasi Kolom"): st.code(", ".join(WELL_TEST_COLUMNS))
            
        if uploaded_test:
            df_test = load_excel_file(uploaded_test)
            
            if df_test is not None:
                missing = [c for c in WELL_TEST_COLUMNS if c not in df_test.columns]
                if missing:
                    st.error(f"Kolom hilang: {missing}")
                else:
                    st.success(f"Data dimuat: {len(df_test)} baris.")
                    
                    if asset_ready:
                        try:
                            asset_ref = st.session_state['asset_register_df'][['Well', 'Lifting Method']].drop_duplicates('Well')
                            if 'Lifting Method Name' in df_test.columns:
                                df_test = df_test.merge(asset_ref, on='Well', how='left', suffixes=('', '_asset'))
                                df_test['Lifting Method Name'] = df_test['Lifting Method Name'].fillna(df_test['Lifting Method'])
                                if 'Lifting Method' in df_test.columns and 'Lifting Method' != 'Lifting Method Name':
                                    df_test = df_test.drop(columns=['Lifting Method'])
                        except: pass
                    
                    df_filt = df_test.copy()
                    cols_filt = st.columns(len(WELL_TEST_HIERARCHY))
                    for i, col in enumerate(WELL_TEST_HIERARCHY):
                        df_filt[col] = df_filt[col].astype(str).replace('nan', '')
                        opts = ["Semua"] + sorted(df_filt[col].unique().tolist())
                        sel = cols_filt[i].selectbox(col, opts, key=f"wt_{col}")
                        if sel != "Semua":
                            df_filt = df_filt[df_filt[col] == sel]
                    
                    st.markdown("---")
                    
                    st.subheader("1. Validasi Kelengkapan Data")
                    comp_score = calculate_well_test_completeness(df_filt)
                    st.metric("Total Data Completeness", f"{comp_score:.2f}%")
                    
                    with st.expander("Detail Tabel"):
                        display_dataframe_optimized(df_filt, WELL_TEST_VALIDATE_BASIC)
                            
                    st.subheader("2. Validasi Kaidah Engineering")
                    df_eng = validate_engineering_rules(df_filt, st.session_state['asset_register_df'])
                    
                    st.write("### 🚨 Data Bermasalah")
                    df_problems = df_eng[df_eng['Keterangan Error'] != "OK"].copy()
                    
                    if df_problems.empty:
                        st.success("✅ Tidak ditemukan pelanggaran.")
                    else:
                        display_cols = ['Well', 'Test Date (dd/mm/yyyy)', 'Keterangan Error', 'Test Duration(Hours)', 'Oil (BOPD)']
                        display_dataframe_optimized(df_problems[display_cols], ['Keterangan Error'], use_highlight=True)
                        csv_eng = df_problems.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download Data Bermasalah", csv_eng, "Engineering_Issues.csv", "text/csv")

    # ---------------------------------------------------------
    # MODUL 3: WELL PARAMETER (WELL ON)
    # ---------------------------------------------------------
    elif main_menu == "Well Parameter (Well On)":
        st.title("⚙️ Modul 3: Well Parameter (Well On)")
        
        lift_type = st.sidebar.radio("Pilih Artificial Lift:", list(WELL_PARAM_CONFIG.keys()))
        config = WELL_PARAM_CONFIG[lift_type]
        target_input = config['input_cols']
        target_check = config['check_cols']
        
        uploaded_param = st.file_uploader(f"Upload Excel {lift_type}", type=['xlsx', 'xls'])
        
        with st.expander(f"Lihat Kolom Wajib {lift_type}"): st.code(", ".join(target_input))
            
        if uploaded_param:
            df_param = load_excel_file(uploaded_param)
            
            if df_param is not None:
                missing = [c for c in target_input if c not in df_param.columns]
                if missing:
                    st.error(f"Kolom hilang: {missing}")
                else:
                    st.success(f"Data {lift_type} dimuat: {len(df_param)} baris.")
                    
                    st.markdown("---")
                    st.subheader("Filter Data")
                    
                    df_filt = df_param.copy()
                    cols_filt = st.columns(len(WELL_PARAM_HIERARCHY))
                    last_selected_level = -1
                    
                    for i, col in enumerate(WELL_PARAM_HIERARCHY):
                        if col in df_filt.columns:
                            df_filt[col] = df_filt[col].astype(str).replace('nan', '')
                            opts = ["Semua"] + sorted(df_filt[col].unique().tolist())
                            sel = cols_filt[i].selectbox(col, opts, key=f"wp_{col}")
                            
                            if sel != "Semua":
                                df_filt = df_filt[df_filt[col] == sel]
                                last_selected_level = i
                        else:
                            cols_filt[i].text(f"{col} (N/A)")

                    st.markdown("---")
                    
                    # 1. KELENGKAPAN (Optimized)
                    st.subheader("1. Validasi Kelengkapan Data")
                    st.info("Pengecekan hanya untuk Active Well Producing & Active Well Non Production.")
                    
                    score = calculate_well_param_completeness(df_filt, lift_type)
                    st.metric("Total Completeness (Active Wells)", f"{score:.2f}%")

                    next_level_idx = last_selected_level + 1
                    if next_level_idx < len(WELL_PARAM_HIERARCHY):
                        breakdown_col = WELL_PARAM_HIERARCHY[next_level_idx]
                        if breakdown_col in df_filt.columns:
                            st.markdown(f"**Breakdown Completeness per {breakdown_col}:**")
                            
                            try:
                                # Hitung Breakdown
                                # Karena logic completeness well param cukup kompleks (conditional), 
                                # kita tidak bisa pakai groupby simple. Kita iterasi group unik saja (biasanya jumlah grup tidak terlalu banyak)
                                groups = df_filt[breakdown_col].unique()
                                breakdown_data = []
                                for g in groups:
                                    sub_df = df_filt[df_filt[breakdown_col] == g]
                                    sub_score = calculate_well_param_completeness(sub_df, lift_type)
                                    breakdown_data.append({breakdown_col: g, "Completeness (%)": sub_score, "Jumlah Data": len(sub_df)})
                                
                                st.dataframe(pd.DataFrame(breakdown_data).sort_values("Completeness (%)"), use_container_width=True)
                            except:
                                st.warning("Gagal membuat breakdown otomatis.")
                    
                    with st.expander("Detail Tabel Data"):
                        display_dataframe_optimized(df_filt, target_check)
                            
                    missing_summary = get_missing_details(df_filt, target_check)
                    if missing_summary['Jumlah Kosong'].sum() > 0:
                        st.warning("Rincian Kekurangan Data (Semua Status):")
                        st.dataframe(missing_summary[missing_summary['Jumlah Kosong'] > 0], use_container_width=False)
                    
                    st.subheader("2. Validasi Kaidah Engineering")
                    
                    df_eng_param = validate_well_parameter_rules(df_filt, lift_type)
                    
                    pass_rate = (df_eng_param['Keterangan Error'] == "OK").sum() / len(df_eng_param) * 100 if len(df_eng_param) > 0 else 0
                    st.metric("Engineering Compliance Rate", f"{pass_rate:.2f}%")
                    
                    st.write("### 🚨 Data Bermasalah")
                    df_problems = df_eng_param[df_eng_param['Keterangan Error'] != "OK"].copy()
                    
                    if df_problems.empty:
                        st.success("✅ Tidak ditemukan pelanggaran.")
                    else:
                        show_cols = ['Well', 'Well Status', 'Keterangan Error']
                        # Tambahkan beberapa kolom sample dari target_check
                        show_cols += target_check[:5]
                        show_cols = [c for c in show_cols if c in df_problems.columns]
                        
                        display_dataframe_optimized(df_problems[show_cols], ['Keterangan Error'], use_highlight=True)
                            
                        csv_err = df_problems.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download Data Bermasalah", csv_err, f"Engineering_Issues_{lift_type}.csv", "text/csv")

if __name__ == "__main__":
    main()
