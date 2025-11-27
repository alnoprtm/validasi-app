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

# Inisialisasi Session State untuk menyimpan data Asset Register
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
    Melakukan validasi Engineering Rules 1-4.
    Mengembalikan DataFrame dengan kolom tambahan hasil validasi (True/False)
    dan kolom keterangan error.
    """
    res = df_test.copy()
    
    # Pastikan kolom numerik terbaca sebagai angka
    num_cols = ["Test Duration(Hours)", "Oil (BOPD)", "Water (BWPD)", "Gas (MMSCFD)", "Condensate (BCPD)", "Fluid (BFPD)"]
    for col in num_cols:
        res[col] = pd.to_numeric(res[col], errors='coerce').fillna(0)

    # --- RULE 2: Jika Duration > 0, Produksi tidak boleh 0 semua ---
    # Logic: Jika Duration > 0 AND (Oil=0 & Water=0 & Gas=0 & Cond=0 & Fluid=0) -> FALSE (Invalid)
    is_producing = (res["Oil (BOPD)"] > 0) | (res["Water (BWPD)"] > 0) | (res["Gas (MMSCFD)"] > 0) | (res["Condensate (BCPD)"] > 0) | (res["Fluid (BFPD)"] > 0)
    res['Rule2_Pass'] = ~((res["Test Duration(Hours)"] > 0) & (~is_producing))
    
    # --- RULE 3: Jika Salah satu produksi > 0, Duration harus > 0 ---
    produces_something = (res["Oil (BOPD)"] > 0) | (res["Water (BWPD)"] > 0) | (res["Gas (MMSCFD)"] > 0) | (res["Condensate (BCPD)"] > 0)
    res['Rule3_Pass'] = ~((produces_something) & (res["Test Duration(Hours)"] <= 0))
    
    # --- RULE 4: Semua value harus >= 0 (Tidak boleh negatif) ---
    res['Rule4_Pass'] = (res[num_cols] >= 0).all(axis=1)

    # --- RULE 1: Frekuensi Test (Butuh data Asset Register) ---
    res['Rule1_Pass'] = True # Default True jika tidak ada data aset (anggap tidak bisa divalidasi)
    check_rule1_active = False # Flag apakah validasi ini berjalan
    
    if df_asset is not None and not df_asset.empty:
        if 'Well Status' in df_asset.columns and 'Well' in df_asset.columns:
            check_rule1_active = True
            # 1. Ambil list Active Well Producing
            active_wells = df_asset[df_asset['Well Status'].str.contains("Active Well Producing", case=False, na=False)]['Well'].unique()
            
            # 2. Cek Tanggal Test
            res['Test Date'] = pd.to_datetime(res['Test Date (dd/mm/yyyy)'], format='%d/%m/%Y', errors='coerce')
            
            # Tentukan Cutoff Date (Hari ini atau Max Date di data - 90 hari)
            if res['Test Date'].dropna().empty:
                max_date = datetime.now()
            else:
                max_date = res['Test Date'].max()
            
            cutoff_date = max_date - pd.timedelta_range(start='1 days', periods=1, freq='90D')[0]
            
            # Cari sumur yang memiliki setidaknya satu tes dalam rentang 90 hari terakhir
            # Sesuai aturan: "Minimal terdapat data well test setiap 3 bulan sekali"
            # Artinya jika dalam 3 bulan terakhir ada tes (berapapun jumlahnya), well tersebut PATUH.
            recent_tests = res[res['Test Date'] >= cutoff_date]['Well'].unique()
            
            def check_rule1(row):
                well_name = row['Well']
                # Jika bukan Active Well, tidak wajib tes -> Lolos
                if well_name not in active_wells:
                    return True 
                # Jika Active Well, harus ada di daftar recent_tests
                if well_name in recent_tests:
                    return True 
                # Active tapi tidak ada tes dalam 3 bulan terakhir -> Gagal
                return False 
            
            res['Rule1_Pass'] = res.apply(check_rule1, axis=1)

    # --- MEMBUAT KOLOM KETERANGAN ERROR ---
    def generate_remarks(row):
        errors = []
        
        # Cek Rule 1
        if check_rule1_active and row['Rule1_Pass'] is False:
            errors.append("⚠️ Active Well tidak ada test >3 bulan")
            
        # Cek Rule 2
        if not row['Rule2_Pass']:
            errors.append("⚠️ Durasi > 0 tapi Produksi Nihil")
            
        # Cek Rule 3
        if not row['Rule3_Pass']:
            errors.append("⚠️ Produksi Ada tapi Durasi 0/Kosong")
            
        # Cek Rule 4
        if not row['Rule4_Pass']:
            errors.append("⚠️ Terdapat Nilai Negatif")
            
        return " | ".join(errors) if errors else "OK"

    res['Keterangan Error'] = res.apply(generate_remarks, axis=1)
    
    return res

# ==========================================
# 4. ANTARMUKA PENGGUNA (MAIN UI)
# ==========================================

def main():
    st.sidebar.title("🛢️ Menu Aplikasi")
    
    # Main Menu Switcher
    main_menu = st.sidebar.selectbox("Pilih Modul Validasi:", ["Asset Register", "Production Well Test"])
    
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

                    # SIMPAN KE SESSION STATE JIKA SUB-MENU ADALAH "WELL"
                    # Ini penting untuk referensi di Modul 2
                    if sub_menu == "Well":
                        st.session_state['asset_register_df'] = df
                        st.toast("Data Well tersimpan untuk referensi Modul Well Test!", icon="💾")

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

                    # Tabel Ringkasan Data Kosong (Dikembalikan)
                    missing_summary = get_missing_details(df_filt, config['validate_columns'])
                    if missing_summary['Jumlah Kosong'].sum() > 0:
                        st.warning("⚠️ **Rincian Kekurangan Data (Per Kolom Wajib):**")
                        st.dataframe(
                            missing_summary[missing_summary['Jumlah Kosong'] > 0], 
                            use_container_width=False
                        )
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
            st.warning("⚠️ Data 'Well' di Asset Register belum diupload. Validasi Rule 1 (Frekuensi 3 bulan) & Referensi Lifting Method tidak akan maksimal.")
        else:
            st.success("✅ Terhubung dengan data Asset Register.")

        uploaded_test = st.file_uploader("Upload Excel Well Test", type=['xlsx', 'xls'])
        
        with st.expander("Lihat Spesifikasi Kolom Well Test"):
            st.code(", ".join(WELL_TEST_COLUMNS))
            
        if uploaded_test:
            try:
                df_test = pd.read_excel(uploaded_test)
                df_test.columns = df_test.columns.str.strip()
                
                # Validasi Header
                missing = [c for c in WELL_TEST_COLUMNS if c not in df_test.columns]
                if missing:
                    st.error(f"Kolom hilang: {missing}")
                else:
                    st.success(f"Data Well Test dimuat: {len(df_test)} baris.")
                    df_test = df_test.dropna(how='all')
                    
                    # --- CROSS-REFERENCE LIFTING METHOD (Jika Kolom Kosong) ---
                    # Jika user ingin menarik lifting method dari Asset Register
                    if asset_ready:
                        # Merge sederhana based on Well Name / Entity ID
                        # Asumsi join key adalah 'Well'
                        asset_ref = st.session_state['asset_register_df'][['Well', 'Lifting Method']].drop_duplicates('Well')
                        # Jika di file test kosong, ambil dari asset
                        if 'Lifting Method Name' in df_test.columns:
                            # Lakukan mapping hanya jika kosong
                            df_test = df_test.merge(asset_ref, on='Well', how='left', suffixes=('', '_asset'))
                            df_test['Lifting Method Name'] = df_test['Lifting Method Name'].fillna(df_test['Lifting Method'])
                    
                    # --- FILTERING HIERARKI ---
                    df_filt = df_test.copy()
                    cols_filt = st.columns(len(WELL_TEST_HIERARCHY))
                    for i, col in enumerate(WELL_TEST_HIERARCHY):
                        df_filt[col] = df_filt[col].astype(str).replace('nan', '')
                        opts = ["Semua"] + sorted(df_filt[col].unique().tolist())
                        sel = cols_filt[i].selectbox(col, opts, key=f"wt_{col}")
                        if sel != "Semua":
                            df_filt = df_filt[df_filt[col] == sel]
                    
                    st.markdown("---")
                    
                    # --- 1. VALIDASI KELENGKAPAN DATA (COMPLETENESS) ---
                    st.subheader("1. Validasi Kelengkapan Data")
                    comp_score = calculate_well_test_completeness(df_filt)
                    
                    col_met1, col_met2 = st.columns(2)
                    with col_met1:
                        st.metric("Total Data Completeness", f"{comp_score:.2f}%")
                        st.caption("*Memperhitungkan kondisi wajib ESP & Gas Lift")
                    
                    # Tampilkan Detail Kelengkapan
                    with st.expander("Detail Tabel Kelengkapan"):
                        # Warnai kolom basic saja untuk visualisasi
                        try:
                            st.dataframe(df_filt.style.apply(highlight_nulls, axis=1, subset=WELL_TEST_VALIDATE_BASIC), use_container_width=True)
                        except:
                            st.dataframe(df_filt, use_container_width=True)
                            
                    # --- 2. VALIDASI KAIDAH ENGINEERING ---
                    st.subheader("2. Validasi Kaidah Engineering")
                    
                    # Jalankan logic engineering
                    df_eng = validate_engineering_rules(df_filt, st.session_state['asset_register_df'])
                    
                    # Hitung Skor Kualitas Engineering (Persentase Pass dari TOTAL data)
                    rule1_score = (df_eng['Rule1_Pass'] == True).sum() / len(df_eng) * 100 if len(df_eng) > 0 else 0
                    rule2_score = df_eng['Rule2_Pass'].sum() / len(df_eng) * 100 if len(df_eng) > 0 else 0
                    rule3_score = df_eng['Rule3_Pass'].sum() / len(df_eng) * 100 if len(df_eng) > 0 else 0
                    rule4_score = df_eng['Rule4_Pass'].sum() / len(df_eng) * 100 if len(df_eng) > 0 else 0
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Rule 1 (Freq 3 Mo)", f"{rule1_score:.1f}%", help="Active Well wajib test minimal sekali dalam 3 bulan terakhir")
                    c2.metric("Rule 2 (Non-Zero)", f"{rule2_score:.1f}%", help="Jika Durasi>0, Produksi gaboleh 0 semua")
                    c3.metric("Rule 3 (Duration)", f"{rule3_score:.1f}%", help="Jika Produksi>0, Durasi wajib >0")
                    c4.metric("Rule 4 (Positive)", f"{rule4_score:.1f}%", help="Tidak boleh ada angka negatif")
                    
                    # --- OUTPUT HANYA BARIS YANG BERMASALAH ---
                    st.write("### 🚨 Data Bermasalah (Problematic Rows)")
                    st.caption("Menampilkan baris data yang melanggar minimal satu aturan engineering.")
                    
                    # Filter: Ambil baris yang Keterangan Error-nya TIDAK "OK"
                    df_problems = df_eng[df_eng['Keterangan Error'] != "OK"].copy()
                    
                    if df_problems.empty:
                        st.success("✅ Tidak ditemukan pelanggaran kaidah engineering pada data ini.")
                    else:
                        # Tampilkan Kolom Penting + Keterangan Error
                        display_cols = [
                            'Well', 'Test Date (dd/mm/yyyy)', 'Keterangan Error',
                            'Test Duration(Hours)', 'Oil (BOPD)', 'Water (BWPD)', 'Gas (MMSCFD)'
                        ]
                        
                        # Warnai tabel error
                        # Kita beri warna latar merah muda pada baris yang error
                        st.dataframe(
                            df_problems[display_cols],
                            use_container_width=True
                        )
                        
                        # Download Result (Hanya yang error)
                        csv_eng = df_problems.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Data Bermasalah (.csv)", 
                            data=csv_eng, 
                            file_name="Engineering_Issues.csv", 
                            mime="text/csv"
                        )

            except Exception as e:
                st.error(f"Gagal memproses Well Test: {e}")

if __name__ == "__main__":
    main()
