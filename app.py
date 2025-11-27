import streamlit as st
import pandas as pd
import io

# ==========================================
# 1. KONFIGURASI HALAMAN & STATE
# ==========================================
st.set_page_config(
    page_title="Validasi Asset Register",
    page_icon="🛢️",
    layout="wide"
)

# --- PERBAIKAN ERROR ---
# Menaikkan batas limit styling pandas agar bisa mewarnai tabel besar
# Kita set ke 5 Juta sel (cukup untuk data yang sangat besar)
pd.set_option("styler.render.max_elements", 5000000)

# ==========================================
# 2. DEFINISI ATURAN & SKEMA (DATABASE LOGIC)
# ==========================================
MODULE_CONFIG = {
    "Zona": {
        "all_columns": [
            "EntityID", "Region", "Zone", "Latitude", "Longitude"
        ],
        "validate_columns": [
            "Latitude", "Longitude"
        ],
        "filter_hierarchy": ["Region", "Zone"]
    },
    
    "Working Area": {
        "all_columns": [
            "Entity ID", "Entity Type", "Region", "Zone", "Working Area Name", 
            "Participating Interest", "Latitude", "Longitude", "Profile Summary"
        ],
        "validate_columns": [
            "Latitude", "Longitude"
        ],
        "filter_hierarchy": ["Region", "Zone", "Working Area Name"]
    },
    
    "Asset Operation": {
        "all_columns": [
            "Entity ID", "Region", "Zona", "Working Area", "Working Area Type", 
            "Asset Operation", "Latitude", "Longitude"
        ],
        "validate_columns": [
            "Latitude", "Longitude"
        ],
        "filter_hierarchy": ["Region", "Zona", "Working Area", "Asset Operation"]
    },
    
    "Well": {
        "all_columns": [
            "Regional", "Zona", "Working Area", "Asset Operation", "Entity ID", 
            "Well", "Well Status", "Skin Status", "Lifting Method", "Reservoir Name", 
            "Main Fluid Type", "Latitude", "Longitude", "POP Date Target", 
            "POP Date Actual", "Profile Summary"
        ],
        "validate_columns": [
            "Well Status", "Skin Status", "Lifting Method", "Reservoir Name", 
            "Main Fluid Type", "Latitude", "Longitude", "POP Date Target", 
            "POP Date Actual", "Profile Summary"
        ],
        "filter_hierarchy": ["Regional", "Zona", "Working Area", "Asset Operation", "Well"]
    }
}

# ==========================================
# 3. FUNGSI-FUNGSI LOGIKA (BACKEND)
# ==========================================

def calculate_completeness_percentage(df, target_cols):
    """Menghitung % kelengkapan data."""
    if df.empty:
        return 0.0
    
    total_expected_cells = len(df) * len(target_cols)
    if total_expected_cells == 0:
        return 0.0

    # Anggap string kosong sebagai NaN agar tidak terhitung sebagai 'terisi'
    filled_cells = df[target_cols].replace('', pd.NA).count().sum()
    
    return (filled_cells / total_expected_cells) * 100

def get_missing_details(df, target_cols):
    """Detail kolom yang kosong."""
    temp_df = df[target_cols].replace('', pd.NA)
    return temp_df.isnull().sum().reset_index(name='Jumlah Kosong').rename(columns={'index': 'Nama Kolom'})

def highlight_nulls(s):
    """Warna kuning untuk sel kosong."""
    is_missing = pd.isna(s) | (s == "")
    return ['background-color: #ffeeb0' if v else '' for v in is_missing]

# ==========================================
# 4. ANTARMUKA PENGGUNA (FRONTEND STREAMLIT)
# ==========================================

def main():
    st.title("🗂️ Validasi Data: Asset Register")
    st.markdown("Upload file Excel untuk melakukan validasi kelengkapan data.")
    
    # --- SIDEBAR: PEMILIHAN SUB-MENU ---
    st.sidebar.header("Navigasi Modul")
    selected_module = st.sidebar.radio(
        "Pilih Sub-Menu:",
        list(MODULE_CONFIG.keys())
    )
    
    # Ambil konfigurasi sesuai pilihan user
    config = MODULE_CONFIG[selected_module]
    required_cols = config['all_columns']
    validate_cols = config['validate_columns']
    hierarchy_cols = config['filter_hierarchy']

    # --- BAGIAN 1: INPUT DATA (FILE UPLOAD) ---
    st.markdown("---")
    
    col_up1, col_up2 = st.columns([2, 1])
    
    with col_up1:
        st.subheader(f"1. Upload Data Excel: {selected_module}")
        uploaded_file = st.file_uploader("Pilih file Excel (.xlsx)", type=['xlsx', 'xls'])

    with col_up2:
        st.info("ℹ️ **Syarat Format File:**")
        st.markdown(f"- **Baris 1:** Harus Header (Nama Kolom)")
        st.markdown("- **Kolom Wajib:**")
        st.code(", ".join(required_cols))

    # --- BAGIAN 2: PROSES DATA ---
    if uploaded_file is not None:
        try:
            # Membaca Excel
            df = pd.read_excel(uploaded_file)
            df.columns = df.columns.str.strip() # Bersihkan spasi di header
            
            # --- VALIDASI KOLOM (Header Check) ---
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error("❌ **Format File Excel Tidak Sesuai!**")
                st.error(f"Kolom berikut tidak ditemukan di Excel Anda: {', '.join(missing_cols)}")
                st.warning("Mohon perbaiki nama kolom di Excel Anda agar sesuai dengan daftar 'Kolom Wajib' di atas.")
            
            else:
                st.success(f"✅ File berhasil dibaca: {len(df)} baris data.")
                
                # Bersihkan baris yang sepenuhnya kosong
                df = df.dropna(how='all')
                
                st.markdown("---")
                st.subheader("2. Filter & Analisis Kelengkapan")
                
                # --- BAGIAN 3: DYNAMIC FILTERING (HIERARKI) ---
                df_filtered = df.copy()
                cols_filter = st.columns(len(hierarchy_cols))
                
                for idx, col_name in enumerate(hierarchy_cols):
                    # Konversi ke string & replace nan
                    df_filtered[col_name] = df_filtered[col_name].astype(str).replace('nan', '')
                    unique_values = ["Semua"] + sorted(list(df_filtered[col_name].unique()))
                    
                    with cols_filter[idx]:
                        selected_val = st.selectbox(
                            f"Filter {col_name}", 
                            unique_values, 
                            key=f"filter_{col_name}"
                        )
                    
                    if selected_val != "Semua":
                        df_filtered = df_filtered[df_filtered[col_name] == selected_val]

                # --- BAGIAN 4: HASIL VALIDASI (METRICS) ---
                score = calculate_completeness_percentage(df_filtered, validate_cols)
                
                st.markdown("### Hasil Validasi")
                c1, c2, c3 = st.columns([1, 2, 1])
                
                with c1:
                    if score == 100:
                        st.balloons()
                    st.metric(
                        label="Persentase Kelengkapan", 
                        value=f"{score:.2f}%",
                        delta="Perfect" if score == 100 else f"-{100-score:.2f}%"
                    )
                
                with c2:
                    st.caption(f"Validasi pada {len(validate_cols)} parameter utama:")
                    st.code(", ".join(validate_cols))

                # --- BAGIAN 5: TABEL DETAIL & DOWNLOAD ---
                st.markdown("#### Detail Data (Terfilter)")
                
                # LOGIKA UNTUK MENANGANI DATA BESAR
                # Jika data terlalu besar (>100.000 baris), styling bisa membuat lemot.
                # Tapi karena kita sudah set option di atas, seharusnya aman.
                try:
                    st.dataframe(
                        df_filtered.style.apply(highlight_nulls, axis=1, subset=validate_cols),
                        use_container_width=True
                    )
                except Exception as e:
                    # Fallback jika masih error: Tampilkan tabel tanpa warna
                    st.warning("⚠️ Data terlalu besar untuk pewarnaan (highlight). Menampilkan tabel standar.")
                    st.dataframe(df_filtered, use_container_width=True)
                
                # Ringkasan data kosong
                missing_summary = get_missing_details(df_filtered, validate_cols)
                if missing_summary['Jumlah Kosong'].sum() > 0:
                    st.warning("⚠️ **Rincian Kekurangan Data:**")
                    st.dataframe(
                        missing_summary[missing_summary['Jumlah Kosong'] > 0], 
                        use_container_width=False
                    )
                
                # Download Button
                csv_data = df_filtered.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Hasil Validasi (.csv)",
                    data=csv_data,
                    file_name=f"Validasi_{selected_module}_result.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error("Terjadi kesalahan saat membaca file Excel.")
            st.error(f"Error detail: {e}")

if __name__ == "__main__":
    main()
