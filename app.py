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
# Ini adalah "Otak" dari validasi. Kita mendefinisikan kolom apa saja yang wajib ada,
# kolom mana yang dihitung persennya, dan urutan filter hierarkinya.

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

def load_data_from_clipboard(raw_text):
    """
    Membaca data copy-paste dari Excel.
    Excel menggunakan Tab sebagai pemisah kolom.
    """
    if not raw_text:
        return None
    try:
        # Menggunakan engine python untuk parsing yang lebih aman
        df = pd.read_csv(io.StringIO(raw_text), sep='\t', engine='python')
        
        # Membersihkan spasi tak terlihat di nama kolom (trimming)
        df.columns = df.columns.str.strip()
        
        # Menghapus baris yang kosong melompong (jika ada tercopy baris kosong)
        df = df.dropna(how='all')
        return df
    except Exception as e:
        st.error(f"Gagal membaca data: {e}")
        return None

def check_column_compliance(df, required_cols):
    """
    Memastikan kolom yang di-paste user sesuai dengan aturan sistem.
    """
    missing = [col for col in required_cols if col not in df.columns]
    return missing

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
    filled_cells = df[target_cols].count().sum()
    
    return (filled_cells / total_expected_cells) * 100

def get_missing_details(df, target_cols):
    """
    Menghitung jumlah data kosong per kolom untuk visualisasi
    """
    return df[target_cols].isnull().sum().reset_index(name='Jumlah Kosong').rename(columns={'index': 'Nama Kolom'})

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

    # --- BAGIAN 1: INPUT DATA ---
    st.markdown("---")
    col_input1, col_input2 = st.columns([2, 1])
    
    with col_input1:
        st.subheader(f"1. Input Data: {selected_module}")
        st.caption("Copy data dari Excel (termasuk Header) dan Paste di bawah ini.")
        raw_text = st.text_area("Area Paste Data:", height=150, placeholder="Paste di sini...")

    with col_input2:
        st.info("**Kolom Wajib Ada:**")
        st.code(", ".join(required_cols), language="text")

    # --- BAGIAN 2: PROSES DATA ---
    if raw_text:
        df = load_data_from_clipboard(raw_text)
        
        if df is not None:
            # 2.1 Cek apakah nama kolom sesuai
            missing_cols = check_column_compliance(df, required_cols)
            
            if missing_cols:
                st.error("❌ **Format Data Salah!** Kolom berikut tidak ditemukan:")
                st.write(missing_cols)
                st.warning("Pastikan header di Excel sama persis dengan daftar Kolom Wajib Ada di atas.")
            else:
                st.success(f"✅ Data terbaca: {len(df)} baris.")
                
                st.markdown("---")
                st.subheader("2. Filter & Analisis Kelengkapan")
                
                # --- BAGIAN 3: DYNAMIC FILTERING (HIERARKI) ---
                # Ini adalah logika untuk filter bertingkat (Region -> Zone -> WK)
                
                df_filtered = df.copy()
                cols_filter = st.columns(len(hierarchy_cols))
                
                filters_applied = {}
                
                for idx, col_name in enumerate(hierarchy_cols):
                    # Ambil nilai unik dari data yang SUDAH terfilter sebelumnya
                    unique_values = ["Semua"] + sorted(list(df_filtered[col_name].astype(str).unique()))
                    
                    with cols_filter[idx]:
                        selected_val = st.selectbox(
                            f"{col_name}", 
                            unique_values, 
                            key=f"filter_{col_name}"
                        )
                    
                    # Terapkan filter jika user tidak memilih "Semua"
                    if selected_val != "Semua":
                        df_filtered = df_filtered[df_filtered[col_name].astype(str) == selected_val]
                        filters_applied[col_name] = selected_val

                # --- BAGIAN 4: HASIL VALIDASI (METRICS) ---
                
                # Hitung persentase berdasarkan data yang sudah difilter
                score = calculate_completeness_percentage(df_filtered, validate_cols)
                
                # Tampilkan Metric Besar
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
                
                # Highlight baris yang ada data kosong (kuning)
                def highlight_nulls(s):
                    is_null = pd.isnull(s)
                    return ['background-color: #ffeeb0' if v else '' for v in is_null]
                
                # Tampilkan dataframe dengan highlight
                st.dataframe(
                    df_filtered.style.apply(highlight_nulls, axis=1, subset=validate_cols),
                    use_container_width=True
                )
                
                # Ringkasan data kosong
                missing_summary = get_missing_details(df_filtered, validate_cols)
                if missing_summary['Jumlah Kosong'].sum() > 0:
                    st.warning("⚠️ **Rincian Kekurangan Data:**")
                    st.dataframe(missing_summary[missing_summary['Jumlah Kosong'] > 0], use_container_width=False)
                
                # Download Button
                csv_data = df_filtered.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Data Validasi (.csv)",
                    data=csv_data,
                    file_name=f"Validasi_{selected_module}_result.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()
