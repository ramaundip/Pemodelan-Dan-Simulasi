# ANGGOTA
# Muhammad Anugerah Ramadhan - 24060122140180
# Muhammad Rayyis Budi - 24060122140112
# Yohanes Panjaitan - 24060122140108

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import datetime
import time

# ================================
# KONFIGURASI HALAMAN
# ================================
st.set_page_config(
    page_title="Bitcoin Monte Carlo Simulator",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B35;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ================================
# FUNGSI UTILITAS
# ================================

@st.cache_data(ttl=3600)  # Cache selama 1 jam
def load_bitcoin_data(symbol, start_date):
    """Mengunduh data Bitcoin dengan caching"""
    try:
        data = yf.download(symbol, start=start_date, progress=False)['Close']
        return data, None
    except Exception as e:
        return None, str(e)

def calculate_statistics(data):
    """Menghitung drift dan volatilitas"""
    log_returns = np.log(1 + data.pct_change())
    u = log_returns.mean()
    var = log_returns.var()
    drift = u - (0.5 * var)
    stdev = log_returns.std()
    # Pastikan S0 adalah scalar float, bukan Series
    S0 = float(data.iloc[-1]) if hasattr(data.iloc[-1], 'iloc') else float(data.iloc[-1])
    return drift, stdev, S0, log_returns

def run_monte_carlo(S0, drift, stdev, days, iterations):
    """Menjalankan simulasi Monte Carlo"""
    # Convert to float to ensure scalar values
    S0 = float(S0)
    drift = float(drift)
    stdev = float(stdev)
    
    daily_returns = np.exp(
        drift + stdev * norm.ppf(np.random.rand(days, iterations))
    )
    
    price_paths = np.zeros_like(daily_returns)
    price_paths[0] = S0
    
    for t in range(1, days):
        price_paths[t] = price_paths[t-1] * daily_returns[t]
    
    return price_paths

def create_path_chart(price_paths, S0, days):
    """Membuat grafik interaktif path simulasi"""
    fig = go.Figure()
    
    # Sample beberapa path untuk visualisasi
    sample_size = min(100, price_paths.shape[1])
    sample_indices = np.random.choice(price_paths.shape[1], sample_size, replace=False)
    
    # Plot sample paths
    for idx in sample_indices:
        fig.add_trace(go.Scatter(
            x=list(range(days)),
            y=price_paths[:, idx],
            mode='lines',
            line=dict(color='cyan', width=0.5),
            opacity=0.3,
            showlegend=False,
            hovertemplate='Hari: %{x}<br>Harga: $%{y:,.2f}<extra></extra>'
        ))
    
    # Plot mean path
    mean_path = price_paths.mean(axis=1)
    fig.add_trace(go.Scatter(
        x=list(range(days)),
        y=mean_path,
        mode='lines',
        line=dict(color='red', width=4),
        name='Rata-rata',
        hovertemplate='Hari: %{x}<br>Rata-rata: $%{y:,.2f}<extra></extra>'
    ))
    
    # Add current price line
    fig.add_hline(y=S0, line_dash="dash", line_color="yellow", 
                  annotation_text=f"Harga Saat Ini: ${S0:,.2f}")
    
    fig.update_layout(
        title="Simulasi Path Harga Bitcoin",
        xaxis_title="Hari ke-",
        yaxis_title="Harga (USD)",
        template="plotly_dark",
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_histogram(final_prices, stats):
    """Membuat histogram distribusi harga akhir"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=final_prices,
        nbinsx=50,
        marker_color='orange',
        opacity=0.7,
        name='Distribusi'
    ))
    
    # Add vertical lines for statistics
    fig.add_vline(x=stats['expected'], line_dash="dash", line_color="red", 
                  annotation_text=f"Mean: ${stats['expected']:,.0f}")
    fig.add_vline(x=stats['best_case'], line_dash="dot", line_color="green", 
                  annotation_text=f"Top 5%: ${stats['best_case']:,.0f}")
    fig.add_vline(x=stats['worst_case'], line_dash="dot", line_color="red", 
                  annotation_text=f"Bottom 5%: ${stats['worst_case']:,.0f}")
    
    fig.update_layout(
        title="Distribusi Probabilitas Harga Akhir",
        xaxis_title="Harga (USD)",
        yaxis_title="Frekuensi",
        template="plotly_dark",
        height=400,
        showlegend=True
    )
    
    return fig

def create_volatility_chart(log_returns):
    """Membuat grafik volatilitas historis"""
    fig = go.Figure()
    
    rolling_vol = log_returns.rolling(window=30).std() * np.sqrt(252)
    
    fig.add_trace(go.Scatter(
        x=log_returns.index,
        y=rolling_vol,
        mode='lines',
        fill='tozeroy',
        line=dict(color='purple', width=2),
        name='Volatilitas 30-hari'
    ))
    
    fig.update_layout(
        title="Volatilitas Historis Bitcoin (Annualized)",
        xaxis_title="Tanggal",
        yaxis_title="Volatilitas",
        template="plotly_dark",
        height=300
    )
    
    return fig

# ================================
# SIDEBAR - PARAMETER INPUT
# ================================

st.sidebar.title("⚙️ Pengaturan Simulasi")
st.sidebar.markdown("---")

# Crypto selection
crypto_symbol = st.sidebar.selectbox(
    "Pilih Cryptocurrency",
    ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD"],
    index=0
)

# Date range
start_date = st.sidebar.date_input(
    "Tanggal Mulai Data Historis",
    value=datetime(2023, 1, 1),
    max_value=datetime.now()
)

st.sidebar.markdown("---")

# Simulation parameters
days_to_simulate = st.sidebar.slider(
    "Jumlah Hari Simulasi",
    min_value=7,
    max_value=365,
    value=30,
    step=1,
    help="Berapa hari ke depan ingin diprediksi"
)

iterations = st.sidebar.slider(
    "Jumlah Iterasi Monte Carlo",
    min_value=100,
    max_value=10000,
    value=1000,
    step=100,
    help="Semakin banyak iterasi, semakin akurat (tapi lebih lambat)"
)

st.sidebar.markdown("---")

# Confidence levels
confidence_level = st.sidebar.select_slider(
    "Confidence Level",
    options=[90, 95, 99],
    value=95,
    help="Tingkat kepercayaan untuk best/worst case"
)

# Run button
run_simulation = st.sidebar.button("🚀 JALANKAN SIMULASI", type="primary")

st.sidebar.markdown("---")
st.sidebar.info("""
📊 **Tips:**
- Iterasi lebih banyak = hasil lebih stabil
- Hari lebih panjang = prediksi jangka panjang
- Gunakan data historis minimal 1 tahun
""")

# ================================
# MAIN CONTENT
# ================================

# Header
st.markdown('<p class="main-header">₿ SIMULASI MONTE CARLO - PREDIKSI HARGA CRYPTO</p>', 
            unsafe_allow_html=True)

# Initialize session state
if 'simulation_data' not in st.session_state:
    st.session_state.simulation_data = None

# Load data
with st.spinner(f"📥 Mengunduh data {crypto_symbol}..."):
    data, error = load_bitcoin_data(crypto_symbol, start_date.strftime('%Y-%m-%d'))

if error:
    st.error(f"❌ Error: {error}")
    st.stop()

if data is None or len(data) < 30:
    st.error("❌ Data tidak cukup. Pilih tanggal mulai yang lebih awal.")
    st.stop()

# Calculate statistics
drift, stdev, S0, log_returns = calculate_statistics(data)

# Convert S0 to float if it's a Series
if isinstance(S0, pd.Series):
    S0 = float(S0.iloc[-1])
else:
    S0 = float(S0)

# Display current info
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("💰 Harga Saat Ini", f"${S0:,.2f}")

with col2:
    daily_change = float(data.pct_change().iloc[-1]) * 100
    st.metric("📈 Perubahan Hari Ini", f"{daily_change:.2f}%", 
              delta=f"{daily_change:.2f}%")

with col3:
    st.metric("📊 Data Points", f"{len(data):,}")

with col4:
    annual_vol = float(stdev) * np.sqrt(252) * 100
    st.metric("📉 Volatilitas Tahunan", f"{annual_vol:.2f}%")

st.markdown("---")

# Run simulation when button clicked
if run_simulation:
    with st.spinner(f"🔄 Menjalankan {iterations:,} simulasi untuk {days_to_simulate} hari..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        start_time = time.time()
        
        # Run simulation
        price_paths = run_monte_carlo(S0, drift, stdev, days_to_simulate, iterations)
        
        # Calculate statistics
        final_prices = price_paths[-1]
        upper_percentile = 50 + confidence_level/2
        lower_percentile = 50 - confidence_level/2
        
        stats = {
            'expected': np.mean(final_prices),
            'median': np.median(final_prices),
            'best_case': np.percentile(final_prices, upper_percentile),
            'worst_case': np.percentile(final_prices, lower_percentile),
            'std': np.std(final_prices),
            'max': np.max(final_prices),
            'min': np.min(final_prices)
        }
        
        # Store in session state
        st.session_state.simulation_data = {
            'price_paths': price_paths,
            'stats': stats,
            'final_prices': final_prices,
            'S0': S0,
            'days': days_to_simulate,
            'iterations': iterations,
            'log_returns': log_returns
        }
        
        elapsed_time = time.time() - start_time
        progress_bar.progress(100)
        status_text.success(f"✅ Simulasi selesai dalam {elapsed_time:.2f} detik!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()

# Display results if simulation has been run
if st.session_state.simulation_data:
    sim_data = st.session_state.simulation_data
    
    st.success(f"✅ Simulasi berhasil dengan {sim_data['iterations']:,} iterasi untuk {sim_data['days']} hari ke depan")
    
    # Statistics cards
    st.markdown("### 📊 Hasil Prediksi")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        change_pct = ((sim_data['stats']['expected'] - sim_data['S0']) / sim_data['S0']) * 100
        st.metric(
            "🎯 Rata-rata (Expected)", 
            f"${sim_data['stats']['expected']:,.2f}",
            delta=f"{change_pct:+.2f}%"
        )
    
    with col2:
        best_change = ((sim_data['stats']['best_case'] - sim_data['S0']) / sim_data['S0']) * 100
        st.metric(
            f"🚀 Optimis (Top {confidence_level}%)", 
            f"${sim_data['stats']['best_case']:,.2f}",
            delta=f"{best_change:+.2f}%"
        )
    
    with col3:
        worst_change = ((sim_data['stats']['worst_case'] - sim_data['S0']) / sim_data['S0']) * 100
        st.metric(
            f"⚠️ Pesimis (Bottom {confidence_level}%)", 
            f"${sim_data['stats']['worst_case']:,.2f}",
            delta=f"{worst_change:+.2f}%"
        )
    
    with col4:
        st.metric(
            "📏 Standar Deviasi", 
            f"${sim_data['stats']['std']:,.2f}"
        )
    
    st.markdown("---")
    
    # Charts
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Path Simulasi", "📊 Distribusi", "📉 Volatilitas", "📋 Detail"])
    
    with tab1:
        st.plotly_chart(
            create_path_chart(sim_data['price_paths'], sim_data['S0'], sim_data['days']), 
            use_container_width=True
        )
    
    with tab2:
        st.plotly_chart(
            create_histogram(sim_data['final_prices'], sim_data['stats']), 
            use_container_width=True
        )
        
        # Additional statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📊 Median", f"${sim_data['stats']['median']:,.2f}")
            st.metric("📈 Maksimum", f"${sim_data['stats']['max']:,.2f}")
        with col2:
            st.metric("📉 Minimum", f"${sim_data['stats']['min']:,.2f}")
            range_val = sim_data['stats']['max'] - sim_data['stats']['min']
            st.metric("📏 Range", f"${range_val:,.2f}")
    
    with tab3:
        st.plotly_chart(
            create_volatility_chart(sim_data['log_returns']), 
            use_container_width=True
        )
        
        st.info(f"""
        📊 **Statistik Volatilitas:**
        - Volatilitas Harian: {float(stdev)*100:.2f}%
        - Volatilitas Tahunan: {float(stdev)*np.sqrt(252)*100:.2f}%
        - Drift (Trend): {float(drift)*252*100:.2f}% per tahun
        """)
    
    with tab4:
        st.markdown("### 📋 Detail Simulasi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **Parameter Simulasi:**
            - Cryptocurrency: `{crypto_symbol}`
            - Harga Awal (S0): `${sim_data['S0']:,.2f}`
            - Jumlah Hari: `{sim_data['days']}`
            - Jumlah Iterasi: `{sim_data['iterations']:,}`
            - Confidence Level: `{confidence_level}%`
            """)
        
        with col2:
            st.markdown(f"""
            **Statistik Model:**
            - Drift (μ): `{float(drift):.6f}`
            - Volatilitas (σ): `{float(stdev):.6f}`
            - Data Historis: `{len(data)} hari`
            - Tanggal Mulai: `{start_date}`
            """)
        
        # Download data
        st.markdown("---")
        st.markdown("### 💾 Download Hasil")
        
        # Prepare dataframe
        results_df = pd.DataFrame({
            'Metrik': ['Harga Saat Ini', 'Expected', 'Optimis', 'Pesimis', 'Median', 'Std Dev', 'Min', 'Max'],
            'Nilai (USD)': [
                sim_data['S0'],
                sim_data['stats']['expected'],
                sim_data['stats']['best_case'],
                sim_data['stats']['worst_case'],
                sim_data['stats']['median'],
                sim_data['stats']['std'],
                sim_data['stats']['min'],
                sim_data['stats']['max']
            ]
        })
        
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Hasil (CSV)",
            data=csv,
            file_name=f"monte_carlo_{crypto_symbol}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

else:
    st.info("👈 Atur parameter di sidebar dan klik **'JALANKAN SIMULASI'** untuk memulai!")
    
    # Show historical data chart
    st.markdown("### 📈 Data Historis")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data.values,
        mode='lines',
        name='Harga',
        line=dict(color='cyan', width=2)
    ))
    fig.update_layout(
        title=f"Harga Historis {crypto_symbol}",
        xaxis_title="Tanggal",
        yaxis_title="Harga (USD)",
        template="plotly_dark",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>⚠️ <b>Disclaimer:</b> Ini adalah simulasi probabilistik untuk tujuan edukasi. 
    Bukan saran investasi. Cryptocurrency sangat volatil dan berisiko tinggi.</p>
</div>
""", unsafe_allow_html=True)