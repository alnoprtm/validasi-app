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

# Inisialisasi Session State untuk menyimpan data Asset Register (KHUSUS SUB-MENU WELL)
if 'asset_register_df' not in st.session_state:
    st.session_state['asset_register_df'] = None

# ==========================================
# 2. DEFINISI ATURAN & SKEMA (DATABASE LOGIC)
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
# Header Umum (Common Columns) untuk semua jenis Lift
WELL_PARAM_COMMON = [
    "Regional", "Zona", "Working Area", "Asset Operation", "Entity ID",
    "Well", "Well Status", "Skin Status", "Lifting Method", "Reservoir Name",
    "Main Fluid Type", "Latitude", "Longitude", "POP Date Target",
    "POP Date Actual", "Profile Summary"
]

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
# 3. FUNGSI-FUNGSI LOGIKA (BACKEND)
# ==========================================

def highlight_nulls(s):
    """Warna kuning untuk sel kosong/NaN."""
    is_missing = pd.isna(s) | (s == "")
    return ['background-color: #ffeeb0' if v else '' for v in is_missing]

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

def calculate_well_test_completeness(df):
    """
    Menghitung kelengkapan Well Test dengan Logika Kondisional (ESP/Gas Lift).
    """
    if df.empty: return 0.0
    
    total_expected = 0
    total_filled = 0
    
    # 1. Hitung kolom basic (Wajib untuk semua baris)
    total_expected += len(df) * len(WELL_TEST_VALIDATE_BASIC)
    total_filled += df[WELL_TEST_VALIDATE_BASIC].replace('', pd.NA).count().sum()
    
    # 2. Logika Kondisional per Baris
    # Kita menggunakan iterasi vector atau apply agar cepat
    
    # Cek ESP -> Wajib Pump Intake
    esp_mask = df['Lifting Method Name'].astype(str).str.contains('Electric Submersible Pump', case=False, na=False)
    if esp_mask.any():
        total_expected += esp_mask.sum() # Tambah 1 kolom wajib untuk baris ESP
        filled_esp = df.loc[esp_mask, 'Pump Intake Pressure (Psig)'].replace('', pd.NA).notna().sum()
        total_filled += filled_esp
        
    # Cek Gas Lift -> Wajib Gas Lift (MMSCFD) & GUF
    gl_mask = df['Lifting Method Name'].astype(str).str.contains('Gas Lift', case=False, na=False)
    if gl_mask.any():
        total_expected += (gl_mask.sum() * 2) # Tambah 2 kolom wajib
        filled_gl_1 = df.loc[gl_mask, 'Gas Lift (MMSCFD)'].replace('', pd.NA).notna().sum()
        filled_gl_2 = df.loc[gl_mask, 'Gas Utilization Factor'].replace('', pd.NA).notna().sum()
        total_filled += filled_gl_1 + filled_gl_2
        
    if total_expected == 0: return 0.0
    return (total_filled / total_expected) * 100

def validate_engineering_rules(df_test, df_asset):
    """
    Melakukan validasi Engineering Rules Modul Well Test.
    """
    res = df_test.copy()
    
    # Pastikan kolom numerik terbaca sebagai angka
    num_cols = ["Test Duration(Hours)", "Oil (BOPD)", "Water (BWPD)", "Gas (MMSCFD)", "Condensate (BCPD)", "Fluid (BFPD)"]
    for col in num_cols:
        res[col] = pd.to_numeric(res[col], errors='coerce').fillna(0)

    # --- RULE 2, 3, 4 ---
    is_producing = (res["Oil (BOPD)"] > 0) | (res["Water (BWPD)"] > 0) | (res["Gas (MMSCFD)"] > 0) | (res["Condensate (BCPD)"] > 0) | (res["Fluid (BFPD)"] > 0)
    res['Rule2_Pass'] = ~((res["Test Duration(Hours)"] > 0) & (~is_producing))
    
    produces_something = (res["Oil (BOPD)"] > 0) | (res["Water (BWPD)"] > 0) | (res["Gas (MMSCFD)"] > 0) | (res["Condensate (BCPD)"] > 0)
    res['Rule3_Pass'] = ~((produces_something) & (res["Test Duration(Hours)"] <= 0))
    
    res['Rule4_Pass'] = (res[num_cols] >= 0).all(axis=1)

    # --- RULE 1: Frekuensi Test ---
    res['Rule1_Pass'] = True 
    check_rule1_active = False
    
    if df_asset is not None and not df_asset.empty:
        if 'Well Status' in df_asset.columns and 'Well' in df_asset.columns:
            check_rule1_active = True
            active_wells = df_asset[df_asset['Well Status'].str.contains("Active Well Producing", case=False, na=False)]['Well'].unique()
            
            res['Test Date'] = pd.to_datetime(res['Test Date (dd/mm/yyyy)'], format='%d/%m/%Y', errors='coerce')
            if res['Test Date'].dropna().empty:
                max_date = datetime.now()
            else:
                max_date = res['Test Date'].max()
            
            cutoff_date = max_date - pd.timedelta_range(start='1 days', periods=1, freq='90D')[0]
            recent_tests = res[res['Test Date'] >= cutoff_date]['Well'].unique()
            
            def check_rule1(row):
                well_name = row['Well']
                if well_name not in active_wells: return True 
                if well_name in recent_tests: return True 
                return False 
            
            res['Rule1_Pass'] = res.apply(check_rule1, axis=1)

    # --- KETERANGAN ERROR ---
    def generate_remarks(row):
        errors = []
        if check_rule1_active and row['Rule1_Pass'] is False: errors.append("⚠️ Active Well tidak ada test >3 bulan")
        if not row['Rule2_Pass']: errors.append("⚠️ Durasi > 0 tapi Produksi Nihil")
        if not row['Rule3_Pass']: errors.append("⚠️ Produksi Ada tapi Durasi 0/Kosong")
        if not row['Rule4_Pass']: errors.append("⚠️ Terdapat Nilai Negatif")
        return " | ".join(errors) if errors else "OK"

    res['Keterangan Error'] = res.apply(generate_remarks, axis=1)
    return res

# --- FUNGSI KHUSUS WELL PARAMETER (WELL ON) ---

def validate_well_parameter_rules(df_input, lift_type, df_asset):
    """
    Validasi Engineering Kompleks untuk Modul Well Parameter.
    Mengutamakan 'Well Status' dari file input jika ada.
    """
    df_merged = df_input.copy()
    
    # 1. Menyiapkan Kolom Status
    # Cek apakah 'Well Status' sudah ada di input (seharusnya ada berdasarkan config baru)
    if 'Well Status' not in df_merged.columns:
        # Fallback: Merge dengan Asset Register jika tidak ada di input
        if df_asset is None or df_asset.empty:
            df_merged['Well Status'] = "Unknown"
        else:
            asset_status = df_asset[['Well', 'Well Status']].drop_duplicates('Well')
            df_merged = df_merged.merge(asset_status, on='Well', how='left')
            df_merged['Well Status'] = df_merged['Well Status'].fillna("Unknown")
    else:
        # Jika sudah ada, pastikan tidak null
        df_merged['Well Status'] = df_merged['Well Status'].fillna("Unknown")

    # Normalisasi string (case insensitive)
    df_merged['Status_Norm'] = df_merged['Well Status'].astype(str).str.strip()
    
    def check_row(row):
        errs = []
        status = row['Status_Norm']
        
        # Helper untuk cek > 0
        def check_positive(col_name):
            val = row.get(col_name)
            try:
                if pd.isna(val) or float(val) <= 0:
                    errs.append(f"{col_name} <= 0")
            except:
                pass # Abaikan jika bukan angka, nanti akan tertangkap validasi tipe data lain jika perlu

        # Helper untuk cek Range (Min < Design < Max)
        def check_design_range():
            try:
                des = float(row.get("Pump Optimal Design Rate / Capacity Design (BFPD)", 0))
                min_v = float(row.get("Min BLPD (BFPD)", 0))
                max_v = float(row.get("Max BLPD (BFPD)", 0))
                if not (min_v < des < max_v):
                    errs.append("Design Rate tidak di antara Min & Max")
            except:
                pass 

        # --- LOGIKA UMUM PER LIFT TYPE ---
        
        # 1. ESP
        if lift_type == "ESP":
            try:
                if float(row.get("Pump Efficiency (%)", 0)) > 100: errs.append("Efficiency > 100%")
            except: pass
            
            always_positive = ["Pump Type", "Serial Name", "Automatic Gas Handler", "Shroud", "Protector", "Sensor"]
            for c in always_positive:
                val = row.get(c)
                if val == 0 or val == "0": errs.append(f"{c} is 0")

            if "Active Well Producing" in status:
                cols = [
                    "Dynamic Fluid Level (ft-MD)", "Pump Efficiency (%)", "Pump Optimal Design Rate / Capacity Design (BFPD)",
                    "Pump Setting Depth (ft-MD)", "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Stages",
                    "Frequency (Hz)", "Ampere (Amp)", "Voltage (Volt)", "Rotation (RPM)", "EQPM HP (HP)",
                    "EQPM Rate (BFPD)", "Motor Voltage (Volt)", "Motor Amps (Amp)", "WHP", "Discharge Pressure (Psi)", "PBHP (Psi)"
                ]
                for c in cols: check_positive(c)
                
            elif "Active Well Non Production" in status:
                cols = [
                    "Static Fluid Level (ft-MD)", "Pump Efficiency (%)", "Pump Optimal Design Rate / Capacity Design (BFPD)",
                    "Pump Setting Depth (ft-MD)", "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Stages",
                    "Frequency (Hz)", "Ampere (Amp)", "Voltage (Volt)", "Rotation (RPM)", "EQPM HP (HP)",
                    "EQPM Rate (BFPD)", "Motor Voltage (Volt)", "Motor Amps (Amp)", "WHP", "Discharge Pressure (Psi)", "PBHP (Psi)"
                ]
                for c in cols: check_positive(c)
                
            check_design_range()

        # 2. GL
        elif lift_type == "GL":
            if "Active Well Producing" in status:
                cols = [
                    "Dynamic Fluid Level (ft-MD)", "Injection Fluid Pressure (Psig)", "InjectionTemperature (F)",
                    "Number Gas Lift Valve", "Gas Injection Choke (MMSCFD)", "Rate Injection Choke (MMSCFD)",
                    "Rate Liquid Optimization (MMSCFD)", "Gas Injection Rate (MMSCFD)", "PBHP (Psi)", "Remarks"
                ]
                for c in cols: check_positive(c) 
                
            elif "Active Well Non Production" in status:
                cols = [
                    "Static Fluid Level (ft-MD)", "Injection Fluid Pressure (Psig)", "InjectionTemperature (F)",
                    "Number Gas Lift Valve", "Gas Injection Choke (MMSCFD)", "Rate Injection Choke (MMSCFD)",
                    "Rate Liquid Optimization (MMSCFD)", "Gas Injection Rate (MMSCFD)", "PBHP (Psi)", "Remarks"
                ]
                for c in cols: check_positive(c)

        # 3. HJP
        elif lift_type == "HJP":
            if "Active Well Producing" in status or "Active Well Non Production" in status:
                cols = ["Nozzle (Inch)", "Throat (Inch)", "Injection Fluid Pressure (Psig)", "Injection Point (ft-MD)", "PBHP (Psi)", "Remarks"]
                for c in cols: check_positive(c)

        # 4. HPU
        elif lift_type == "HPU":
            if row.get("Pump Type") == 0: errs.append("Pump Type is 0")
            if row.get("Serial Name") == 0: errs.append("Serial Name is 0")
            
            if "Active Well Producing" in status:
                cols = [
                    "Dynamic Fluid Level (ft-MD)", "Pump Efficiency (%)", "Pump Optimal Design Rate / Capacity Design (BFPD)",
                    "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Pump Setting Depth (ft-MD)",
                    "Stroke Length", "Stroke Per Minute (SPM)", "PBHP (Psi)", "Remarks"
                ]
                for c in cols: check_positive(c)

            elif "Active Well Non Production" in status:
                cols = [
                    "Static Fluid Level (ft-MD)", "Pump Efficiency (%)", "Pump Optimal Design Rate / Capacity Design (BFPD)",
                    "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Pump Setting Depth (ft-MD)",
                    "Stroke Length", "Stroke Per Minute (SPM)", "PBHP (Psi)", "Remarks"
                ]
                for c in cols: check_positive(c)
            
            check_design_range()

        # 5. PCP
        elif lift_type == "PCP":
            if row.get("Pump Type") == 0: errs.append("Pump Type is 0")
            if row.get("Serial Name") == 0: errs.append("Serial Name is 0")
            
            if "Active Well Producing" in status:
                cols = [
                    "Dynamic Fluid Level (ft-MD)", "Pump Efficiency (%)", "Pump Optimal Design Rate / Capacity Design (BFPD)",
                    "Pump Type", "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Serial Name",
                    "Pump Setting Depth (ft-MD)", "PBHP (Psi)", "Remarks"
                ]
                for c in cols: check_positive(c)
            elif "Active Well Non Production" in status:
                cols = [
                    "Static Fluid Level (ft-MD)", "Pump Efficiency (%)", "Pump Optimal Design Rate / Capacity Design (BFPD)",
                    "Pump Type", "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Serial Name",
                    "Pump Setting Depth (ft-MD)", "PBHP (Psi)", "Remarks"
                ]
                for c in cols: check_positive(c)
                
            check_design_range()
            
        # 6. SRP
        elif lift_type == "SRP":
            if row.get("Pump Type") == 0: errs.append("Pump Type is 0")
            if row.get("Serial Name") == 0: errs.append("Serial Name is 0")

            if "Active Well Producing" in status:
                cols = [
                    "Dynamic Fluid Level (ft-MD)", "Pump Efficiency (%)", "Pump Optimal Design Rate / Capacity Design (BFPD)",
                    "Pump Type", "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Pump Setting Depth (ft-MD)",
                    "PBHP (Psi)", "Remarks"
                ]
                for c in cols: check_positive(c)
            elif "Active Well Non Production" in status:
                cols = [
                    "Static Fluid Level (ft-MD)", "Pump Efficiency (%)", "Pump Optimal Design Rate / Capacity Design (BFPD)",
                    "Pump Type", "Min BLPD (BFPD)", "Max BLPD (BFPD)", "Pump Size", "Pump Setting Depth (ft-MD)",
                    "PBHP (Psi)", "Remarks"
                ]
                for c in cols: check_positive(c)

            check_design_range()

        return " | ".join(errs) if errs else "OK"

    df_merged['Keterangan Error'] = df_merged.apply(check_row, axis=1)
    return df_merged

# ==========================================
# 4. ANTARMUKA PENGGUNA (MAIN UI)
# ==========================================

def main():
    st.sidebar.title("🛢️ Menu Aplikasi")
    
    # Main Menu Switcher
    main_menu = st.sidebar.selectbox(
        "Pilih Modul Validasi:", 
        ["Asset Register", "Production Well Test", "Well Parameter (Well On)"]
    )
    
    # ---------------------------------------------------------
    # MODUL 1: ASSET REGISTER
    # ---------------------------------------------------------
    if main_menu == "Asset Register":
        st.title("📂 Modul 1: Asset Register")
        st.markdown("Upload data aset untuk validasi kelengkapan dan sebagai referensi modul lain.")
        
        # Sub-Menu Asset
        sub_menu = st.sidebar.radio("Sub-Menu Asset:", list(ASSET_CONFIG.keys()))
        config = ASSET_CONFIG[sub_menu]
        
        # File Upload
        uploaded_file = st.file_uploader(f"Upload Excel ({sub_menu})", type=['xlsx', 'xls'])
        
        col_info, col_req = st.columns([1, 2])
        with col_info:
            st.info("Baris 1 Excel harus Header.")
        with col_req:
            with st.expander("Lihat Kolom Wajib"):
                st.code(", ".join(config['all_columns']))
        
        if uploaded_file:
            try:
                df = pd.read_excel(uploaded_file)
                df.columns = df.columns.str.strip()
                
                # Validasi Kolom
                missing = [c for c in config['all_columns'] if c not in df.columns]
                if missing:
                    st.error(f"Kolom hilang: {missing}")
                else:
                    st.success(f"Data {sub_menu} dimuat: {len(df)} baris.")
                    df = df.dropna(how='all')

                    # --- LOGIKA PENYIMPANAN SESSION STATE ---
                    if sub_menu == "Well":
                        st.session_state['asset_register_df'] = df
                        st.toast("Data Well berhasil disimpan ke memori untuk referensi Modul lain!", icon="💾")
                    
                    # --- FILTERING ---
                    df_filt = df.copy()
                    cols_filt = st.columns(len(config['filter_hierarchy']))
                    for i, col in enumerate(config['filter_hierarchy']):
                        df_filt[col] = df_filt[col].astype(str).replace('nan', '')
                        opts = ["Semua"] + sorted(df_filt[col].unique().tolist())
                        sel = cols_filt[i].selectbox(col, opts, key=f"ar_{col}")
                        if sel != "Semua":
                            df_filt = df_filt[df_filt[col] == sel]
                    
                    # --- HASIL ---
                    score = calculate_simple_completeness(df_filt, config['validate_columns'])
                    st.metric("Completeness", f"{score:.2f}%")
                    
                    # Tabel Utama
                    st.markdown("##### Detail Data")
                    try:
                        st.dataframe(df_filt.style.apply(highlight_nulls, axis=1, subset=config['validate_columns']), use_container_width=True)
                    except:
                        st.dataframe(df_filt, use_container_width=True)

                    # Tabel Ringkasan Data Kosong
                    missing_summary = get_missing_details(df_filt, config['validate_columns'])
                    if missing_summary['Jumlah Kosong'].sum() > 0:
                        st.warning("⚠️ **Rincian Kekurangan Data (Per Kolom Wajib):**")
                        st.dataframe(missing_summary[missing_summary['Jumlah Kosong'] > 0], use_container_width=False)
                    else:
                        st.success("✅ Semua kolom wajib telah terisi penuh.")
                        
            except Exception as e:
                st.error(f"Error: {e}")

    # ---------------------------------------------------------
    # MODUL 2: PRODUCTION WELL TEST
    # ---------------------------------------------------------
    elif main_menu == "Production Well Test":
        st.title("🧪 Modul 2: Production Well Test")
        
        # Cek Referensi Asset Register
        asset_ready = st.session_state['asset_register_df'] is not None
        
        if not asset_ready:
            st.warning("⚠️ **Peringatan:** Data 'Well' di Asset Register belum diupload.")
        else:
            st.success("✅ Terhubung dengan data referensi Asset Register.")

        uploaded_test = st.file_uploader("Upload Excel Well Test", type=['xlsx', 'xls'])
        
        with st.expander("Lihat Spesifikasi Kolom Well Test"):
            st.code(", ".join(WELL_TEST_COLUMNS))
            
        if uploaded_test:
            try:
                df_test = pd.read_excel(uploaded_test)
                df_test.columns = df_test.columns.str.strip()
                
                missing = [c for c in WELL_TEST_COLUMNS if c not in df_test.columns]
                if missing:
                    st.error(f"Kolom hilang: {missing}")
                else:
                    st.success(f"Data dimuat: {len(df_test)} baris.")
                    df_test = df_test.dropna(how='all')
                    
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
                    
                    # 1. COMPLETENESS
                    st.subheader("1. Validasi Kelengkapan Data")
                    comp_score = calculate_well_test_completeness(df_filt)
                    st.metric("Total Data Completeness", f"{comp_score:.2f}%")
                    
                    with st.expander("Detail Tabel Kelengkapan"):
                        try:
                            st.dataframe(df_filt.style.apply(highlight_nulls, axis=1, subset=WELL_TEST_VALIDATE_BASIC), use_container_width=True)
                        except:
                            st.dataframe(df_filt, use_container_width=True)
                            
                    # 2. ENGINEERING RULES
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

            except Exception as e:
                st.error(f"Error: {e}")

    # ---------------------------------------------------------
    # MODUL 3: WELL PARAMETER (WELL ON)
    # ---------------------------------------------------------
    elif main_menu == "Well Parameter (Well On)":
        st.title("⚙️ Modul 3: Well Parameter (Well On)")
        
        # Cek Referensi Asset Register (Optional tapi recommended)
        asset_ready = st.session_state['asset_register_df'] is not None
        if not asset_ready:
            st.info("ℹ️ Tips: Upload data Asset Register terlebih dahulu untuk validasi status sumur yang lebih akurat (jika status tidak tersedia di file input).")
        
        # Sub-Menu Lift Type
        lift_type = st.sidebar.radio("Pilih Artificial Lift:", list(WELL_PARAM_CONFIG.keys()))
        target_cols = WELL_PARAM_CONFIG[lift_type]
        
        uploaded_param = st.file_uploader(f"Upload Excel {lift_type}", type=['xlsx', 'xls'])
        
        with st.expander(f"Lihat Kolom Wajib {lift_type}"):
            st.code(", ".join(target_cols))
            
        if uploaded_param:
            try:
                df_param = pd.read_excel(uploaded_param)
                df_param.columns = df_param.columns.str.strip()
                
                # Cek Header
                missing = [c for c in target_cols if c not in df_param.columns]
                if missing:
                    st.error(f"Kolom hilang: {missing}")
                else:
                    st.success(f"Data {lift_type} dimuat: {len(df_param)} baris.")
                    df_param = df_param.dropna(how='all')
                    
                    st.markdown("---")
                    
                    # 1. KELENGKAPAN (COMPLETENESS)
                    # Sesuai prompt, completeness ditampilkan seperti modul sebelumnya
                    st.subheader("1. Validasi Kelengkapan Data")
                    score = calculate_simple_completeness(df_param, target_cols)
                    st.metric("Completeness", f"{score:.2f}%")
                    
                    with st.expander("Detail Tabel Kelengkapan"):
                        try:
                            st.dataframe(df_param.style.apply(highlight_nulls, axis=1, subset=target_cols), use_container_width=True)
                        except:
                            st.dataframe(df_param, use_container_width=True)
                            
                    # Ringkasan Kosong
                    missing_summary = get_missing_details(df_param, target_cols)
                    if missing_summary['Jumlah Kosong'].sum() > 0:
                        st.warning("Rincian Kekurangan Data:")
                        st.dataframe(missing_summary[missing_summary['Jumlah Kosong'] > 0], use_container_width=False)
                    
                    # 2. ENGINEERING RULES
                    st.subheader("2. Validasi Kaidah Engineering")
                    
                    df_eng_param = validate_well_parameter_rules(df_param, lift_type, st.session_state['asset_register_df'])
                    
                    # Hitung Pass Rate
                    pass_rate = (df_eng_param['Keterangan Error'] == "OK").sum() / len(df_eng_param) * 100 if len(df_eng_param) > 0 else 0
                    st.metric("Engineering Compliance Rate", f"{pass_rate:.2f}%")
                    
                    # Output Error Only
                    st.write("### 🚨 Data Bermasalah")
                    st.caption("Validasi berdasarkan status sumur (Active Producing / Active Non Production)")
                    
                    df_problems = df_eng_param[df_eng_param['Keterangan Error'] != "OK"].copy()
                    
                    if df_problems.empty:
                        st.success("✅ Tidak ditemukan pelanggaran kaidah engineering.")
                    else:
                        # Tampilkan kolom Well, Status, Keterangan Error, dan beberapa parameter utama
                        show_cols = ['Well', 'Well Status', 'Keterangan Error']
                        # Tambahkan beberapa kolom data penting dari config lift type sebagai preview (selain kolom umum)
                        # Kita ambil kolom dari index 16 keatas (karena 0-15 adalah kolom umum)
                        important_params = target_cols[16:22] if len(target_cols) > 22 else target_cols[16:]
                        show_cols += important_params
                        
                        # Pastikan kolom ada
                        show_cols = [c for c in show_cols if c in df_problems.columns]
                        
                        try:
                            st.dataframe(
                                df_problems[show_cols].style.applymap(lambda _: 'background-color: #ffcccc', subset=['Keterangan Error']),
                                use_container_width=True
                            )
                        except:
                            st.dataframe(df_problems[show_cols], use_container_width=True)
                            
                        csv_err = df_problems.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download Data Bermasalah", csv_err, f"Engineering_Issues_{lift_type}.csv", "text/csv")
                        
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
