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

# Menaikkan batas limit styling pandas
pd.set_option("styler.render.max_elements", 5000000)

# Inisialisasi Session State
if 'asset_register_df' not in st.session_state:
    st.session_state['asset_register_df'] = None

# ==========================================
# 2. DEFINISI ATURAN & SKEMA
# ==========================================

# --- CONFIG MODUL 1: ASSET REGISTER ---
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

# --- CONFIG MODUL 2: PRODUCTION WELL TEST ---
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

# --- CONFIG MODUL 3: WELL PARAMETER (WELL ON) ---
WELL_PARAM_COMMON = [
    "Regional", "Zona", "Working Area", "Asset Operation", "Entity ID",
    "Well", "Well Status", "Skin Status", "Lifting Method", "Reservoir Name",
    "Main Fluid Type", "Latitude", "Longitude", "POP Date Target",
    "POP Date Actual", "Profile Summary"
]

WELL_PARAM_HIERARCHY = ["Regional", "Zona", "Working Area", "Asset Operation", "Well"]

WELL_PARAM_CONFIG = {
    "ESP": WELL_PARAM_COMMON + [
        "Ownership", "Manufacturer", "Brand", "Supplier / Vendor",
        "Dynamic Fluid Level (ft-MD)", "Static Fluid Level (ft-MD)", "Pump Efficiency (%)",
        "Pump Optimal Design Rate / Capacity Design (BFPD)", "Pump Setting Depth (ft-MD)",
        "Pump Type", "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Serial Name",
        "Stages", "Frequency (Hz)", "Ampere (Amp)", "Voltage (Volt)", "Rotation (RPM)",
        "EQPM HP (HP)", "EQPM Rate (BFPD)", "Motor Voltage (Volt)", "Motor Amps (Amp)",
        "Automatic Gas Handler", "Shroud", "Protector", "Sensor", "Start Date",
        "DHP Date (dd/MM/yyyy)", "WHP", "Discharge Pressure (Psi)", "PBHP (Psi)", "Remarks"
    ],
    "GL": WELL_PARAM_COMMON + [
        "Ownership", "Manufacturer", "Brand", "Supplier / Vendor",
        "Dynamic Fluid Level (ft-MD)", "Static Fluid Level (ft-MD)",
        "Injection Fluid Pressure (Psig)", "InjectionTemperature (F)", "Number Gas Lift Valve",
        "Gas Injection Choke (MMSCFD)", "Rate Injection Choke (MMSCFD)",
        "Rate Liquid Optimization (MMSCFD)", "Gas Injection Rate (MMSCFD)", "PBHP (Psi)", "Remarks"
    ],
    "HJP": WELL_PARAM_COMMON + [
        "Ownership", "Manufacturer", "Brand", "Supplier / Vendor",
        "Nozzle (Inch)", "Throat (Inch)", "Injection Fluid Pressure (Psig)",
        "Injection Point (ft-MD)", "PBHP (Psi)", "Remarks"
    ],
    "HPU": WELL_PARAM_COMMON + [
        "Ownership", "Manufacturer", "Brand", "Supplier / Vendor",
        "Dynamic Fluid Level (ft-MD)", "Static Fluid Level (ft-MD)", "Pump Efficiency (%)",
        "Pump Optimal Design Rate / Capacity Design (BFPD)", "Pump Type",
        "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Serial Name",
        "Pump Setting Depth (ft-MD)", "Stroke Length", "Stroke Per Minute (SPM)", "PBHP (Psi)", "Remarks"
    ],
    "PCP": WELL_PARAM_COMMON + [
        "Ownership", "Manufacturer", "Brand", "Supplier / Vendor",
        "Dynamic Fluid Level (ft-MD)", "Static Fluid Level (ft-MD)", "Pump Efficiency (%)",
        "Pump Optimal Design Rate / Capacity Design (BFPD)", "Pump Type",
        "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Serial Name",
        "Pump Setting Depth (ft-MD)", "PBHP (Psi)", "Remarks"
    ],
    "SRP": WELL_PARAM_COMMON + [
        "Ownership", "Manufacturer", "Brand", "Supplier / Vendor",
        "Stroke Length", "Stroke Per Minute (SPM)",
        "Dynamic Fluid Level (ft-MD)", "Static Fluid Level (ft-MD)", "Pump Efficiency (%)",
        "Pump Optimal Design Rate / Capacity Design (BFPD)", "Pump Type",
        "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Pump Setting Depth (ft-MD)",
        "PBHP (Psi)", "Remarks"
    ]
}

# ==========================================
# 3. FUNGSI-FUNGSI LOGIKA (OPTIMIZED BACKEND)
# ==========================================

# OPTIMISASI 1: Caching fungsi pembacaan file
@st.cache_data(ttl=3600)
def load_excel_file(uploaded_file):
    """Membaca file excel dengan caching agar tidak dibaca ulang saat filter berubah."""
    try:
        df = pd.read_excel(uploaded_file)
        df.columns = df.columns.str.strip()
        # Hapus baris kosong sepenuhnya
        df = df.dropna(how='all')
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def highlight_nulls(s):
    """Warna kuning untuk sel kosong/NaN."""
    is_missing = pd.isna(s) | (s == "")
    return ['background-color: #ffeeb0' if v else '' for v in is_missing]

# OPTIMISASI 2: Caching perhitungan completeness
@st.cache_data
def calculate_simple_completeness(df, target_cols):
    """Menghitung % kelengkapan standar."""
    if df.empty: return 0.0
    total_expected = len(df) * len(target_cols)
    if total_expected == 0: return 0.0
    filled = df[target_cols].replace('', pd.NA).count().sum()
    return (filled / total_expected) * 100

def get_missing_details(df, target_cols):
    """Detail kolom yang kosong."""
    temp_df = df[target_cols].replace('', pd.NA)
    return temp_df.isnull().sum().reset_index(name='Jumlah Kosong').rename(columns={'index': 'Nama Kolom'})

# --- FUNGSI KHUSUS WELL TEST ---

@st.cache_data
def calculate_well_test_completeness(df):
    """
    Menghitung kelengkapan Well Test dengan Logika Kondisional (ESP/Gas Lift).
    """
    if df.empty: return 0.0
    
    total_expected = 0
    total_filled = 0
    
    # 1. Hitung kolom basic (Vectorized count)
    total_expected += len(df) * len(WELL_TEST_VALIDATE_BASIC)
    total_filled += df[WELL_TEST_VALIDATE_BASIC].replace('', pd.NA).count().sum()
    
    # 2. Logika Kondisional (Vectorized)
    # Cek ESP
    esp_mask = df['Lifting Method Name'].astype(str).str.contains('Electric Submersible Pump', case=False, na=False)
    if esp_mask.any():
        total_expected += esp_mask.sum()
        filled_esp = df.loc[esp_mask, 'Pump Intake Pressure (Psig)'].replace('', pd.NA).notna().sum()
        total_filled += filled_esp
        
    # Cek Gas Lift
    gl_mask = df['Lifting Method Name'].astype(str).str.contains('Gas Lift', case=False, na=False)
    if gl_mask.any():
        total_expected += (gl_mask.sum() * 2)
        filled_gl_1 = df.loc[gl_mask, 'Gas Lift (MMSCFD)'].replace('', pd.NA).notna().sum()
        filled_gl_2 = df.loc[gl_mask, 'Gas Utilization Factor'].replace('', pd.NA).notna().sum()
        total_filled += filled_gl_1 + filled_gl_2
        
    if total_expected == 0: return 0.0
    return (total_filled / total_expected) * 100

# OPTIMISASI 3: Caching Validasi Engineering & Vectorization
@st.cache_data
def validate_engineering_rules(df_test, df_asset):
    """
    Melakukan validasi Engineering Rules Modul Well Test.
    Menggunakan Vectorization agar cepat (menghindari apply row-by-row).
    """
    res = df_test.copy()
    
    # Convert numeric vector
    num_cols = ["Test Duration(Hours)", "Oil (BOPD)", "Water (BWPD)", "Gas (MMSCFD)", "Condensate (BCPD)", "Fluid (BFPD)"]
    for col in num_cols:
        res[col] = pd.to_numeric(res[col], errors='coerce').fillna(0)

    # --- RULE 2, 3, 4 (Vectorized) ---
    is_producing = (res["Oil (BOPD)"] > 0) | (res["Water (BWPD)"] > 0) | (res["Gas (MMSCFD)"] > 0) | (res["Condensate (BCPD)"] > 0) | (res["Fluid (BFPD)"] > 0)
    res['Rule2_Pass'] = ~((res["Test Duration(Hours)"] > 0) & (~is_producing))
    
    produces_something = (res["Oil (BOPD)"] > 0) | (res["Water (BWPD)"] > 0) | (res["Gas (MMSCFD)"] > 0) | (res["Condensate (BCPD)"] > 0)
    res['Rule3_Pass'] = ~((produces_something) & (res["Test Duration(Hours)"] <= 0))
    
    res['Rule4_Pass'] = (res[num_cols] >= 0).all(axis=1)

    # --- RULE 1: Frekuensi Test (Vectorized Logic) ---
    res['Rule1_Pass'] = True 
    check_rule1_active = False
    
    if df_asset is not None and not df_asset.empty:
        if 'Well Status' in df_asset.columns and 'Well' in df_asset.columns:
            check_rule1_active = True
            
            # Gunakan SET untuk lookup O(1) yang cepat
            active_wells = set(df_asset[df_asset['Well Status'].str.contains("Active Well Producing", case=False, na=False)]['Well'].unique())
            
            res['Test Date'] = pd.to_datetime(res['Test Date (dd/mm/yyyy)'], format='%d/%m/%Y', errors='coerce')
            
            if res['Test Date'].dropna().empty:
                max_date = datetime.now()
            else:
                max_date = res['Test Date'].max()
            
            cutoff_date = max_date - pd.timedelta_range(start='1 days', periods=1, freq='90D')[0]
            
            # Wells yang punya tes recent
            recent_tests = set(res[res['Test Date'] >= cutoff_date]['Well'].unique())
            
            # Vectorized Check Rule 1
            # Pass jika: Bukan Active Well OR (Active Well AND ada di Recent Tests)
            is_active_well = res['Well'].isin(active_wells)
            is_recent_test = res['Well'].isin(recent_tests)
            res['Rule1_Pass'] = ~is_active_well | (is_active_well & is_recent_test)

    # --- GENERATE REMARKS ---
    # Gunakan Vectorized String operations (lebih cepat dari apply pada data besar, tapi sedikit lebih kompleks dibaca)
    # Untuk kejelasan kode, kita tetap pakai apply di sini tapi hanya pada list pendek string
    
    # Siapkan kondisi error boolean
    cond1 = (check_rule1_active) & (res['Rule1_Pass'] == False)
    cond2 = ~res['Rule2_Pass']
    cond3 = ~res['Rule3_Pass']
    cond4 = ~res['Rule4_Pass']
    
    # Bangun pesan error
    res['Keterangan Error'] = "OK"
    
    # Cara cepat: Buat kolom temporary untuk pesan
    # Ini jauh lebih cepat daripada apply row-by-row
    res.loc[cond1, 'Err1'] = "⚠️ Active Well tidak ada test >3 bulan | "
    res.loc[cond2, 'Err2'] = "⚠️ Durasi > 0 tapi Produksi Nihil | "
    res.loc[cond3, 'Err3'] = "⚠️ Produksi Ada tapi Durasi 0/Kosong | "
    res.loc[cond4, 'Err4'] = "⚠️ Terdapat Nilai Negatif | "
    
    # Gabungkan (fillna dengan string kosong dulu)
    res_err_cols = ['Err1', 'Err2', 'Err3', 'Err4']
    for c in res_err_cols:
        if c not in res.columns: res[c] = ""
        else: res[c] = res[c].fillna("")
        
    res['Error_Msg_Temp'] = res['Err1'] + res['Err2'] + res['Err3'] + res['Err4']
    
    # Update 'Keterangan Error' hanya jika ada error
    has_error = res['Error_Msg_Temp'] != ""
    res.loc[has_error, 'Keterangan Error'] = res.loc[has_error, 'Error_Msg_Temp'].str.rstrip(" | ")
    
    # Bersihkan kolom temp
    res.drop(columns=res_err_cols + ['Error_Msg_Temp'], inplace=True, errors='ignore')
    
    return res

# --- FUNGSI KHUSUS WELL PARAMETER (WELL ON) ---

@st.cache_data
def validate_well_parameter_rules(df_input, lift_type, df_asset):
    """
    Validasi Engineering Kompleks. Dicache agar saat filter berubah tidak hitung ulang.
    """
    df_merged = df_input.copy()
    
    # Merge Status jika perlu
    if 'Well Status' not in df_merged.columns:
        if df_asset is None or df_asset.empty:
            df_merged['Well Status'] = "Unknown"
        else:
            asset_status = df_asset[['Well', 'Well Status']].drop_duplicates('Well')
            df_merged = df_merged.merge(asset_status, on='Well', how='left')
            df_merged['Well Status'] = df_merged['Well Status'].fillna("Unknown")
    else:
        df_merged['Well Status'] = df_merged['Well Status'].fillna("Unknown")

    df_merged['Status_Norm'] = df_merged['Well Status'].astype(str).str.strip()
    
    # Karena logika ini sangat nested dan conditional, vectorization akan sangat rumit.
    # Namun karena kita menggunakan @st.cache_data, ini hanya akan lambat di awal upload file.
    # Interaksi user (filter) akan tetap cepat.
    
    def check_row(row):
        errs = []
        status = row['Status_Norm']
        
        def check_positive(col_name):
            val = row.get(col_name)
            try:
                if pd.isna(val) or float(val) <= 0:
                    errs.append(f"{col_name} <= 0")
            except: pass

        def check_design_range():
            try:
                des = float(row.get("Pump Optimal Design Rate / Capacity Design (BFPD)", 0))
                min_v = float(row.get("Min BLPD (BFPD)", 0))
                max_v = float(row.get("Max BLPD (BFPD)", 0))
                if not (min_v < des < max_v):
                    errs.append("Design Rate tidak di antara Min & Max")
            except: pass 

        # --- ESP ---
        if lift_type == "ESP":
            try:
                if float(row.get("Pump Efficiency (%)", 0)) > 100: errs.append("Efficiency > 100%")
            except: pass
            
            for c in ["Pump Type", "Serial Name", "Automatic Gas Handler", "Shroud", "Protector", "Sensor"]:
                val = row.get(c)
                if val == 0 or val == "0": errs.append(f"{c} is 0")

            cols_check = []
            if "Active Well Producing" in status:
                cols_check = [
                    "Dynamic Fluid Level (ft-MD)", "Pump Efficiency (%)", "Pump Optimal Design Rate / Capacity Design (BFPD)",
                    "Pump Setting Depth (ft-MD)", "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Stages",
                    "Frequency (Hz)", "Ampere (Amp)", "Voltage (Volt)", "Rotation (RPM)", "EQPM HP (HP)",
                    "EQPM Rate (BFPD)", "Motor Voltage (Volt)", "Motor Amps (Amp)", "WHP", "Discharge Pressure (Psi)", "PBHP (Psi)"
                ]
            elif "Active Well Non Production" in status:
                cols_check = [
                    "Static Fluid Level (ft-MD)", "Pump Efficiency (%)", "Pump Optimal Design Rate / Capacity Design (BFPD)",
                    "Pump Setting Depth (ft-MD)", "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Stages",
                    "Frequency (Hz)", "Ampere (Amp)", "Voltage (Volt)", "Rotation (RPM)", "EQPM HP (HP)",
                    "EQPM Rate (BFPD)", "Motor Voltage (Volt)", "Motor Amps (Amp)", "WHP", "Discharge Pressure (Psi)", "PBHP (Psi)"
                ]
            for c in cols_check: check_positive(c)
            check_design_range()

        # --- GL ---
        elif lift_type == "GL":
            cols_check = []
            if "Active Well Producing" in status:
                cols_check = [
                    "Dynamic Fluid Level (ft-MD)", "Injection Fluid Pressure (Psig)", "InjectionTemperature (F)",
                    "Number Gas Lift Valve", "Gas Injection Choke (MMSCFD)", "Rate Injection Choke (MMSCFD)",
                    "Rate Liquid Optimization (MMSCFD)", "Gas Injection Rate (MMSCFD)", "PBHP (Psi)", "Remarks"
                ]
            elif "Active Well Non Production" in status:
                cols_check = [
                    "Static Fluid Level (ft-MD)", "Injection Fluid Pressure (Psig)", "InjectionTemperature (F)",
                    "Number Gas Lift Valve", "Gas Injection Choke (MMSCFD)", "Rate Injection Choke (MMSCFD)",
                    "Rate Liquid Optimization (MMSCFD)", "Gas Injection Rate (MMSCFD)", "PBHP (Psi)", "Remarks"
                ]
            for c in cols_check: check_positive(c)

        # --- HJP ---
        elif lift_type == "HJP":
            if "Active Well Producing" in status or "Active Well Non Production" in status:
                for c in ["Nozzle (Inch)", "Throat (Inch)", "Injection Fluid Pressure (Psig)", "Injection Point (ft-MD)", "PBHP (Psi)", "Remarks"]:
                    check_positive(c)

        # --- HPU ---
        elif lift_type == "HPU":
            if row.get("Pump Type") == 0: errs.append("Pump Type is 0")
            if row.get("Serial Name") == 0: errs.append("Serial Name is 0")
            
            cols_check = []
            if "Active Well Producing" in status:
                cols_check = [
                    "Dynamic Fluid Level (ft-MD)", "Pump Efficiency (%)", "Pump Optimal Design Rate / Capacity Design (BFPD)",
                    "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Pump Setting Depth (ft-MD)",
                    "Stroke Length", "Stroke Per Minute (SPM)", "PBHP (Psi)", "Remarks"
                ]
            elif "Active Well Non Production" in status:
                cols_check = [
                    "Static Fluid Level (ft-MD)", "Pump Efficiency (%)", "Pump Optimal Design Rate / Capacity Design (BFPD)",
                    "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Pump Setting Depth (ft-MD)",
                    "Stroke Length", "Stroke Per Minute (SPM)", "PBHP (Psi)", "Remarks"
                ]
            for c in cols_check: check_positive(c)
            check_design_range()

        # --- PCP ---
        elif lift_type == "PCP":
            if row.get("Pump Type") == 0: errs.append("Pump Type is 0")
            if row.get("Serial Name") == 0: errs.append("Serial Name is 0")
            
            cols_check = []
            if "Active Well Producing" in status:
                cols_check = [
                    "Dynamic Fluid Level (ft-MD)", "Pump Efficiency (%)", "Pump Optimal Design Rate / Capacity Design (BFPD)",
                    "Pump Type", "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Serial Name",
                    "Pump Setting Depth (ft-MD)", "PBHP (Psi)", "Remarks"
                ]
            elif "Active Well Non Production" in status:
                cols_check = [
                    "Static Fluid Level (ft-MD)", "Pump Efficiency (%)", "Pump Optimal Design Rate / Capacity Design (BFPD)",
                    "Pump Type", "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Serial Name",
                    "Pump Setting Depth (ft-MD)", "PBHP (Psi)", "Remarks"
                ]
            for c in cols_check: check_positive(c)
            check_design_range()
            
        # --- SRP ---
        elif lift_type == "SRP":
            if row.get("Pump Type") == 0: errs.append("Pump Type is 0")
            if row.get("Serial Name") == 0: errs.append("Serial Name is 0")

            cols_check = []
            if "Active Well Producing" in status:
                cols_check = [
                    "Dynamic Fluid Level (ft-MD)", "Pump Efficiency (%)", "Pump Optimal Design Rate / Capacity Design (BFPD)",
                    "Pump Type", "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Pump Setting Depth (ft-MD)",
                    "PBHP (Psi)", "Remarks"
                ]
            elif "Active Well Non Production" in status:
                cols_check = [
                    "Static Fluid Level (ft-MD)", "Pump Efficiency (%)", "Pump Optimal Design Rate / Capacity Design (BFPD)",
                    "Pump Type", "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Pump Setting Depth (ft-MD)",
                    "PBHP (Psi)", "Remarks"
                ]
            for c in cols_check: check_positive(c)
            check_design_range()

        return " | ".join(errs) if errs else "OK"

    df_merged['Keterangan Error'] = df_merged.apply(check_row, axis=1)
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
            # Gunakan fungsi cached untuk load file
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
                    
                    # FILTERING (Dilakukan pada memori tanpa membaca ulang file)
                    df_filt = df.copy()
                    cols_filt = st.columns(len(config['filter_hierarchy']))
                    for i, col in enumerate(config['filter_hierarchy']):
                        df_filt[col] = df_filt[col].astype(str).replace('nan', '')
                        opts = ["Semua"] + sorted(df_filt[col].unique().tolist())
                        sel = cols_filt[i].selectbox(col, opts, key=f"ar_{col}")
                        if sel != "Semua":
                            df_filt = df_filt[df_filt[col] == sel]
                    
                    # HASIL (Menggunakan fungsi cached)
                    score = calculate_simple_completeness(df_filt, config['validate_columns'])
                    st.metric("Completeness", f"{score:.2f}%")
                    
                    st.markdown("##### Detail Data")
                    try:
                        st.dataframe(df_filt.style.apply(highlight_nulls, axis=1, subset=config['validate_columns']), use_container_width=True)
                    except:
                        st.dataframe(df_filt, use_container_width=True)

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
                    
                    # Merge Lifting Method (Cukup cepat, tidak perlu cache berat)
                    if asset_ready:
                        try:
                            asset_ref = st.session_state['asset_register_df'][['Well', 'Lifting Method']].drop_duplicates('Well')
                            if 'Lifting Method Name' in df_test.columns:
                                df_test = df_test.merge(asset_ref, on='Well', how='left', suffixes=('', '_asset'))
                                df_test['Lifting Method Name'] = df_test['Lifting Method Name'].fillna(df_test['Lifting Method'])
                                if 'Lifting Method' in df_test.columns and 'Lifting Method' != 'Lifting Method Name':
                                    df_test = df_test.drop(columns=['Lifting Method'])
                        except: pass
                    
                    # FILTERING
                    df_filt = df_test.copy()
                    cols_filt = st.columns(len(WELL_TEST_HIERARCHY))
                    for i, col in enumerate(WELL_TEST_HIERARCHY):
                        df_filt[col] = df_filt[col].astype(str).replace('nan', '')
                        opts = ["Semua"] + sorted(df_filt[col].unique().tolist())
                        sel = cols_filt[i].selectbox(col, opts, key=f"wt_{col}")
                        if sel != "Semua":
                            df_filt = df_filt[df_filt[col] == sel]
                    
                    st.markdown("---")
                    
                    # 1. COMPLETENESS (Cached)
                    st.subheader("1. Validasi Kelengkapan Data")
                    comp_score = calculate_well_test_completeness(df_filt)
                    st.metric("Total Data Completeness", f"{comp_score:.2f}%")
                    
                    with st.expander("Detail Tabel"):
                        try:
                            st.dataframe(df_filt.style.apply(highlight_nulls, axis=1, subset=WELL_TEST_VALIDATE_BASIC), use_container_width=True)
                        except:
                            st.dataframe(df_filt, use_container_width=True)
                            
                    # 2. ENGINEERING RULES (Cached)
                    st.subheader("2. Validasi Kaidah Engineering")
                    df_eng = validate_engineering_rules(df_filt, st.session_state['asset_register_df'])
                    
                    # Output Error Only
                    st.write("### 🚨 Data Bermasalah")
                    df_problems = df_eng[df_eng['Keterangan Error'] != "OK"].copy()
                    
                    if df_problems.empty:
                        st.success("✅ Tidak ditemukan pelanggaran.")
                    else:
                        display_cols = ['Well', 'Test Date (dd/mm/yyyy)', 'Keterangan Error', 'Test Duration(Hours)', 'Oil (BOPD)']
                        try:
                            st.dataframe(df_problems[display_cols].style.applymap(lambda _: 'background-color: #ffcccc', subset=['Keterangan Error']), use_container_width=True)
                        except:
                            st.dataframe(df_problems[display_cols], use_container_width=True)
                        
                        csv_eng = df_problems.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download Data Bermasalah", csv_eng, "Engineering_Issues.csv", "text/csv")

    # ---------------------------------------------------------
    # MODUL 3: WELL PARAMETER (WELL ON)
    # ---------------------------------------------------------
    elif main_menu == "Well Parameter (Well On)":
        st.title("⚙️ Modul 3: Well Parameter (Well On)")
        
        asset_ready = st.session_state['asset_register_df'] is not None
        if not asset_ready: st.info("ℹ️ Tips: Upload data Asset Register terlebih dahulu.")
        
        lift_type = st.sidebar.radio("Pilih Artificial Lift:", list(WELL_PARAM_CONFIG.keys()))
        target_cols = WELL_PARAM_CONFIG[lift_type]
        
        uploaded_param = st.file_uploader(f"Upload Excel {lift_type}", type=['xlsx', 'xls'])
        
        with st.expander(f"Lihat Kolom Wajib {lift_type}"): st.code(", ".join(target_cols))
            
        if uploaded_param:
            df_param = load_excel_file(uploaded_param)
            
            if df_param is not None:
                missing = [c for c in target_cols if c not in df_param.columns]
                if missing:
                    st.error(f"Kolom hilang: {missing}")
                else:
                    st.success(f"Data {lift_type} dimuat: {len(df_param)} baris.")
                    
                    # FILTERING HIERARKI
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
                    
                    # 1. KELENGKAPAN (Cached)
                    st.subheader("1. Validasi Kelengkapan Data")
                    score = calculate_simple_completeness(df_filt, target_cols)
                    st.metric("Total Completeness (Filtered)", f"{score:.2f}%")

                    # BREAKDOWN
                    next_level_idx = last_selected_level + 1
                    if next_level_idx < len(WELL_PARAM_HIERARCHY):
                        breakdown_col = WELL_PARAM_HIERARCHY[next_level_idx]
                        if breakdown_col in df_filt.columns:
                            st.markdown(f"**Breakdown Completeness per {breakdown_col}:**")
                            
                            unique_groups = df_filt[breakdown_col].unique()
                            # Optimisasi: Tidak looping DataFrame berkali-kali jika grup banyak
                            # Gunakan groupby pandas yang jauh lebih cepat
                            try:
                                # Hitung total cells per group
                                group_counts = df_filt.groupby(breakdown_col)[target_cols].apply(lambda x: x.count().sum())
                                group_sizes = df_filt.groupby(breakdown_col).size() * len(target_cols)
                                group_scores = (group_counts / group_sizes * 100).reset_index(name="Completeness (%)")
                                group_scores['Jumlah Data'] = df_filt.groupby(breakdown_col).size().values
                                
                                st.dataframe(group_scores.sort_values("Completeness (%)"), use_container_width=True)
                            except:
                                st.warning("Gagal membuat breakdown otomatis.")
                    
                    with st.expander("Detail Tabel Data"):
                        try:
                            st.dataframe(df_filt.style.apply(highlight_nulls, axis=1, subset=target_cols), use_container_width=True)
                        except:
                            st.dataframe(df_filt, use_container_width=True)
                            
                    missing_summary = get_missing_details(df_filt, target_cols)
                    if missing_summary['Jumlah Kosong'].sum() > 0:
                        st.warning("Rincian Kekurangan Data:")
                        st.dataframe(missing_summary[missing_summary['Jumlah Kosong'] > 0], use_container_width=False)
                    
                    # 2. ENGINEERING RULES (Cached)
                    st.subheader("2. Validasi Kaidah Engineering")
                    
                    df_eng_param = validate_well_parameter_rules(df_filt, lift_type, st.session_state['asset_register_df'])
                    
                    pass_rate = (df_eng_param['Keterangan Error'] == "OK").sum() / len(df_eng_param) * 100 if len(df_eng_param) > 0 else 0
                    st.metric("Engineering Compliance Rate", f"{pass_rate:.2f}%")
                    
                    st.write("### 🚨 Data Bermasalah")
                    df_problems = df_eng_param[df_eng_param['Keterangan Error'] != "OK"].copy()
                    
                    if df_problems.empty:
                        st.success("✅ Tidak ditemukan pelanggaran.")
                    else:
                        show_cols = ['Well', 'Well Status', 'Keterangan Error']
                        important_params = target_cols[16:22] if len(target_cols) > 22 else target_cols[16:]
                        show_cols += important_params
                        show_cols = [c for c in show_cols if c in df_problems.columns]
                        
                        try:
                            st.dataframe(df_problems[show_cols].style.applymap(lambda _: 'background-color: #ffcccc', subset=['Keterangan Error']), use_container_width=True)
                        except:
                            st.dataframe(df_problems[show_cols], use_container_width=True)
                            
                        csv_err = df_problems.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download Data Bermasalah", csv_err, f"Engineering_Issues_{lift_type}.csv", "text/csv")

if __name__ == "__main__":
    main()
