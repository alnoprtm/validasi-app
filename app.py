import streamlit as st
import pandas as pd
import io

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Validasi Asset Register", layout="wide")

# --- KONFIGURASI MODUL (SCHEMA & RULES) ---
# Di sini kita mendefinisikan kolom apa saja yang wajib ada dan kolom mana yang divalidasi
MODULE_CONFIG = {
    "Zona": {
        "columns": ["EntityID", "Region", "Zone", "Latitude", "Longitude"],
        "validate_cols": ["Latitude", "Longitude"],
        "hierarchy": ["Region", "Zone"]
    },
    "Working Area": {
        "columns": ["Entity ID", "Entity Type", "Region", "Zone", "Working Area Name", 
                    "Participating Interest", "Latitude", "Longitude", "Profile Summary"],
        "validate_cols": ["Latitude", "Longitude"],
        "hierarchy": ["Region", "Zone", "Working Area Name"]
    },
    "Asset Operation": {
        "columns": ["Entity ID", "Region", "Zona", "Working Area", "Working Area Type", 
                    "Asset Operation", "Latitude", "Longitude"],
        "validate_cols": ["Latitude", "Longitude"],
        "hierarchy": ["Region", "Zona", "Working Area", "Asset Operation"]
    },
    "Well": {
        "columns": ["Regional", "Zona", "Working Area", "Asset Operation", "Entity ID", 
                    "Well", "Well Status", "Skin Status", "Lifting Method", "Reservoir Name", 
                    "Main Fluid Type", "Latitude", "Longitude", "POP Date Target", 
                    "POP Date Actual", "Profile Summary"],
        "validate_cols": ["Well Status", "Skin Status", "Lifting Method", "Reservoir Name", 
                          "Main Fluid Type", "Latitude", "Longitude", "POP Date Target", 
                          "POP Date Actual", "Profile Summary"],
        "hierarchy": ["Regional", "Zona", "Working Area", "Asset Operation", "Well"]
    }
}

# --- FUNGSI UTILITAS ---

def parse_data(input_text):
    """Mengubah text copy-paste Excel menjadi DataFrame"""
    if not input_text:
        return None
    try:
        # Menggunakan separator tab (\t) yang umum saat copy dari excel
        df = pd.read_csv(io.StringIO(input_text), sep='\t')
        # Membersihkan spasi di nama kolom agar matching tepat
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Error membaca data: {e}")
        return None

def calculate_completeness(df, target_cols):
    """
    Menghitung % kelengkapan data.
    Rumus: (Jumlah Sel Terisi / Total Sel yang Harus Diisi) * 100
    """
    if df.empty:
        return 0.0
    
    total_cells = len(df) * len(target_cols)
    # Menghitung sel yang TIDAK null (terisi)
    filled_cells = df[target_cols].count().sum() 
    
    if total_cells == 0:
        return 0.0
        
    return (filled_cells / total_cells) * 100

def get_missing_summary(df, target_cols):
    """Memberikan detail kolom mana saja yang kosong"""
    missing_info = df[target_cols].isnull().sum()
    return missing_info[missing_info > 0]

# --- UI UTAMA ---

def main():
    st.title("📂 Modul Validasi: Asset Register")
    st.markdown("---")

    # 1. Sidebar Navigasi Sub-Menu
    sub_menu = st.sidebar.radio(
        "Pilih Sub-Menu:",
        ["Zona", "Working Area", "Asset Operation", "Well"]
    )

    config = MODULE_CONFIG[sub_menu]
    required_cols = config['columns']
    validate_cols = config['validate_cols']
    hierarchy = config['hierarchy']

    # 2. Area Input Data
    st.subheader(f"Input Data: {sub_menu}")
    st.info(f"Pastikan header kolom sesuai spesifikasi. Kolom yang divalidasi kelengkapannya: {', '.join(validate_cols)}")
    
    input_text = st.text_area("Paste data Excel di sini (termasuk Header):", height=150)

    if input_text:
        df = parse_data(input_text)
        
        if df is not None:
            # Validasi Header Kolom
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"⚠️ Nama kolom tidak sesuai! Kolom berikut hilang/salah nama: {', '.join(missing_cols)}")
                st.warning(f"Kolom yang diharapkan: {', '.join(required_cols)}")
            else:
                st.success(f"Data berhasil dimuat: {len(df)} baris.")

                # --- FITUR FILTERING (HIERARKI) ---
                st.markdown("### 🔍 Filter & Hasil Validasi")
                
                # Logic untuk membuat dropdown filter bertingkat
                filtered_df = df.copy()
                selected_filters = {}
                
                cols_filter = st.columns(len(hierarchy))
                
                for i, level_col in enumerate(hierarchy):
                    # Kita tidak memfilter level terakhir (misal 'Well' atau 'Zone' di menu Zone) 
                    # agar user bisa melihat daftar item di level tersebut, 
                    # kecuali user ingin spesifik satu item.
                    
                    unique_vals = ['All'] + sorted(filtered_df[level_col].astype(str).unique().tolist())
                    
                    with cols_filter[i]:
                        selected = st.selectbox(f"Filter {level_col}", unique_vals, key=f"filter_{level_col}")
                        
                    if selected != 'All':
                        filtered_df = filtered_df[filtered_df[level_col].astype(str) == selected]
                        selected_filters[level_col] = selected

                # --- KALKULASI PERSENTASE ---
                completeness_score = calculate_completeness(filtered_df, validate_cols)
                
                # Tampilkan Score Besar
                st.metric(
                    label="Tingkat Kelengkapan Data (Completeness)",
                    value=f"{completeness_score:.2f}%",
                    delta_color="normal"
                )

                # --- DETAIL ANALISIS ---
                st.markdown("#### 📋 Detail Data yang Ditampilkan")
                
                # Visualisasi Bar untuk kolom yang kosong
                missing_counts = get_missing_summary(filtered_df, validate_cols)
                
                col_display1, col_display2 = st.columns([2, 1])
                
                with col_display1:
                    st.dataframe(filtered_df, use_container_width=True)
                
                with col_display2:
                    st.caption("Jumlah Data Kosong per Parameter:")
                    if not missing_counts.empty:
                        st.dataframe(missing_counts, use_container_width=True)
                        st.warning("Ada parameter yang belum terisi!")
                    else:
                        st.success("Semua parameter target terisi penuh untuk filter ini.")

                # Opsi Download Data yang sedang difilter
                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Data Terfilter (CSV)",
                    data=csv,
                    file_name=f"Validated_{sub_menu}.csv",
                    mime='text/csv',
                )

if __name__ == '__main__':
    main()
