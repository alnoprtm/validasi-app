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
    """
    Menghitung % kelengkapan: (Sel Terisi / Total Sel Target) * 100
    """
    if df.empty:
        return 0.0
    
    # Total sel yang harusnya ada isinya
    total_expected_cells = len(df) * len(target_cols)
    
    if total_expected_cells == 0:
        return 0.0

    # Menghitung jumlah sel yang tidak NULL/NaN di kolom target
    # Kita anggap string kosong "" sebagai null juga
    filled_cells = df[target_cols].replace('', pd.NA).count().sum()
    
    return (filled_cells / total_expected_cells) * 100

def get_missing_details(df, target_cols):
    """
    Menghitung jumlah data kosong per kolom untuk visualisasi
    """
    # Pastikan data kosong dianggap NA
    temp_df = df[target_cols].replace('', pd.NA)
    return temp_df.isnull().sum().reset_index(name='Jumlah Kosong').rename(columns={'index': 'Nama Kolom'})

# ==========================================
# 4. ANTARMUKA PENGGUNA (FRONTEND STREAMLIT)
# ==========================================

def main():
    st.title("🗂️ Validasi Data: Asset Register")
    st.markdown("Aplikasi validasi kelengkapan data operasional.")
    
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

    # --- BAGIAN 1: INPUT DATA (MENGGUNAKAN DATA EDITOR) ---
    st.markdown("---")
    st.subheader(f"1. Input Data: {selected_module}")
    
    with st.expander("ℹ️ Cara Input Data (Klik disini)", expanded=True):
        st.write("""
        1. Siapkan data di Excel Anda. Pastikan urutan kolom **sama** dengan tabel di bawah.
        2. Blok data di Excel (Ctrl+C).
        3. Klik sel pojok kiri atas pada tabel di bawah, lalu Paste (Ctrl+V).
        4. Klik tombol 'Lakukan Validasi' jika sudah selesai.
        """)
    
    # Membuat Template DataFrame Kosong dengan Header yang Benar
    # Ini memastikan kolom sudah "Terbagi" sesuai permintaan Anda
    df_template = pd.DataFrame(columns=required_cols)
    
    # Menampilkan Editor Tabel Interaktif
    # num_rows="dynamic" memungkinkan Anda menambah baris dengan paste
    edited_df = st.data_editor(
        df_template,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key=f"editor_{selected_module}" # Key unik agar tidak crash saat ganti menu
    )

    # Tombol Action
    # Kita butuh tombol karena data editor sifatnya interaktif
    process_btn = st.button("🔍 Lakukan Validasi Data", type="primary")

    # --- BAGIAN 2: PROSES DATA ---
    if process_btn:
        if not edited_df.empty:
            st.success(f"✅ Data diterima: {len(edited_df)} baris.")
            
            # Bersihkan data (hapus baris yang semua kolomnya kosong/None)
            df_clean = edited_df.dropna(how='all')
            
            if df_clean.empty:
                st.warning("Data kosong. Silakan paste data terlebih dahulu.")
            else:
                st.markdown("---")
                st.subheader("2. Filter & Analisis Kelengkapan")
                
                # --- BAGIAN 3: DYNAMIC FILTERING (HIERARKI) ---
                df_filtered = df_clean.copy()
                cols_filter = st.columns(len(hierarchy_cols))
                
                for idx, col_name in enumerate(hierarchy_cols):
                    # Konversi ke string agar aman saat filter
                    df_filtered[col_name] = df_filtered[col_name].astype(str).replace('nan', '')
                    
                    unique_values = ["Semua"] + sorted(list(df_filtered[col_name].unique()))
                    
                    with cols_filter[idx]:
                        selected_val = st.selectbox(
                            f"{col_name}", 
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
                    st.caption(f"Validasi dilakukan pada {len(validate_cols)} kolom parameter:")
                    st.code(", ".join(validate_cols))

                # --- BAGIAN 5: TABEL DETAIL & DOWNLOAD ---
                st.markdown("#### Detail Data (Terfilter)")
                
                # Fungsi Highlight
                def highlight_nulls(s):
                    # Cek null atau string kosong
                    is_missing = pd.isna(s) | (s == "")
                    return ['background-color: #ffeeb0' if v else '' for v in is_missing]
                
                # Tampilkan tabel hasil
                st.dataframe(
                    df_filtered.style.apply(highlight_nulls, axis=1, subset=validate_cols),
                    use_container_width=True
                )
                
                # Ringkasan data kosong
                missing_summary = get_missing_details(df_filtered, validate_cols)
                if missing_summary['Jumlah Kosong'].sum() > 0:
                    st.warning("⚠️ **Rincian Kekurangan Data:**")
                    st.dataframe(missing_summary[missing_summary['Jumlah Kosong'] > 0], use_container_width=False)
                
                # Download
                csv_data = df_filtered.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Data Validasi (.csv)",
                    data=csv_data,
                    file_name=f"Validasi_{selected_module}_result.csv",
                    mime="text/csv"
                )
        else:
            st.warning("Tabel masih kosong. Silakan Copy-Paste data dari Excel.")

if __name__ == "__main__":
    main()
