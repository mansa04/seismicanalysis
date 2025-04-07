import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
from scipy.fft import fft, fftfreq
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans

def load_and_preprocess_data(uploaded_file):
    try:
        # Read the CSV file with semicolon separator
        df = pd.read_csv(uploaded_file, sep=';')
        
        # Display initial data info
        st.write("Initial data shape:", df.shape)
        
        # Select columns for analysis (focusing on numeric columns)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # Group columns by type for easier selection
        waveform_cols = [col for col in numeric_columns if any(x in col for x in ['U_T', 'V_T', 'W_T'])]
        parameter_cols = [col for col in numeric_columns if col not in waveform_cols]
        
        st.write("Available column groups:")
        st.write("- Waveform columns:", len(waveform_cols))
        st.write("- Parameter columns:", len(parameter_cols))
        
        return df, waveform_cols, parameter_cols
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

def perform_fft_analysis(data, sampling_rate):
    n = len(data)
    fft_result = fft(data)
    freqs = fftfreq(n, 1/sampling_rate)
    return freqs[:n//2], np.abs(fft_result)[:n//2]

def calculate_pga_pgv(df, component):
    pga = df[f'{component}_pga'].max()
    pgv = df[f'{component}_pgv'].max()
    return pga, pgv

def perform_spectral_analysis(data, fs):
    f, t, Sxx = signal.spectrogram(data, fs=fs)
    return f, t, 10 * np.log10(Sxx + 1e-10)

def main():
    st.title("Advanced Seismic Data Analysis")
    
    with st.sidebar:
        st.header("Analysis Controls")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        
        if uploaded_file is not None:
            analysis_type = st.selectbox(
                "Select Analysis Type",
                ["Data Preview", 
                 "Waveform Analysis",
                 "Frequency Analysis",
                 "PGA/PGV Analysis",
                 "Statistical Analysis",
                 "Pattern Recognition",
                 "Event Classification"]
            )
    
    if uploaded_file is not None:
        # Load and preprocess data
        df, waveform_cols, parameter_cols = load_and_preprocess_data(uploaded_file)
        
        if df is not None:
            if analysis_type == "Data Preview":
                st.header("Data Overview")
                
                # Basic statistics
                st.subheader("Data Statistics")
                st.dataframe(df[waveform_cols].describe())
                
                # Correlation matrix
                st.subheader("Correlation Matrix")
                corr = df[waveform_cols[:10]].corr()  # First 10 columns for visibility
                fig = px.imshow(corr, color_continuous_scale='RdBu')
                st.plotly_chart(fig)
                
            elif analysis_type == "Waveform Analysis":
                st.header("Advanced Waveform Analysis")
                
                # Component selection
                component = st.selectbox("Select Component", ["U", "V", "W"])
                
                # Time window selection
                window_size = st.slider("Time Window (samples)", 
                                     min_value=100, 
                                     max_value=len(df), 
                                     value=1000)
                
                # Get component columns
                component_cols = [col for col in waveform_cols if col.startswith(f'{component}_T')]
                
                if component_cols:
                    # Multi-tab analysis
                    tab1, tab2, tab3 = st.tabs(["Time Domain", "Spectrogram", "Phase Analysis"])
                    
                    with tab1:
                        fig = go.Figure()
                        for col in component_cols[:5]:  # First 5 periods
                            fig.add_trace(go.Scatter(
                                y=df[col][:window_size],
                                name=col,
                                mode='lines'
                            ))
                        fig.update_layout(title=f'{component} Component Waveforms')
                        st.plotly_chart(fig)
                    
                    with tab2:
                        selected_col = st.selectbox("Select Period", component_cols)
                        f, t, Sxx = perform_spectral_analysis(df[selected_col].values[:window_size], fs=100)
                        
                        fig = go.Figure(data=go.Heatmap(
                            z=Sxx,
                            x=t,
                            y=f,
                            colorscale='Viridis'
                        ))
                        fig.update_layout(title='Spectrogram Analysis')
                        st.plotly_chart(fig)
                    
                    with tab3:
                        # Phase analysis
                        selected_col = st.selectbox("Select Period for Phase", component_cols)
                        analytic_signal = signal.hilbert(df[selected_col].values[:window_size])
                        phase = np.angle(analytic_signal)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(y=phase, mode='lines', name='Phase'))
                        fig.update_layout(title='Phase Analysis')
                        st.plotly_chart(fig)

            elif analysis_type == "Frequency Analysis":
                st.header("Frequency Domain Analysis")
                
                component = st.selectbox("Select Component", ["U", "V", "W"])
                selected_col = st.selectbox(
                    "Select Period",
                    [col for col in waveform_cols if col.startswith(f'{component}_T')]
                )
                
                # FFT Analysis
                freqs, magnitude = perform_fft_analysis(df[selected_col].values, sampling_rate=100)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=freqs,
                    y=magnitude,
                    mode='lines',
                    name='FFT'
                ))
                fig.update_layout(title='Frequency Spectrum')
                st.plotly_chart(fig)
                
                # Power Spectral Density
                f, Pxx = signal.welch(df[selected_col].values, fs=100)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=f,
                    y=10 * np.log10(Pxx),
                    mode='lines',
                    name='PSD'
                ))
                fig.update_layout(title='Power Spectral Density')
                st.plotly_chart(fig)
            
            elif analysis_type == "PGA/PGV Analysis":
                st.header("Peak Ground Motion Analysis")
                
                components = ['U', 'V', 'W']
                pga_data = []
                pgv_data = []
                
                for comp in components:
                    pga, pgv = calculate_pga_pgv(df, comp)
                    pga_data.append(pga)
                    pgv_data.append(pgv)
                
                # Plot PGA/PGV comparison
                fig = go.Figure(data=[
                    go.Bar(name='PGA', x=components, y=pga_data),
                    go.Bar(name='PGV', x=components, y=pgv_data)
                ])
                fig.update_layout(title='Peak Ground Motion Comparison')
                st.plotly_chart(fig)
            
            elif analysis_type == "Statistical Analysis":
                st.header("Statistical Analysis")
                
                component = st.selectbox("Select Component", ["U", "V", "W"])
                component_cols = [col for col in waveform_cols if col.startswith(f'{component}_T')]
                
                if component_cols:
                    # Statistical measures
                    stats_df = pd.DataFrame({
                        'Mean': df[component_cols].mean(),
                        'Std': df[component_cols].std(),
                        'Skewness': df[component_cols].skew(),
                        'Kurtosis': df[component_cols].kurtosis()
                    })
                    
                    st.subheader("Statistical Measures")
                    st.dataframe(stats_df)
                    
                    # Distribution plot
                    selected_col = st.selectbox("Select Period for Distribution", component_cols)
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=df[selected_col],
                        nbinsx=50,
                        name='Distribution'
                    ))
                    fig.update_layout(title=f'Distribution of {selected_col}')
                    st.plotly_chart(fig)
            
            elif analysis_type == "Pattern Recognition":
                st.header("Pattern Recognition")
                
                # Select features for clustering
                selected_features = st.multiselect(
                    "Select Features",
                    waveform_cols,
                    default=waveform_cols[:3] if waveform_cols else None
                )
                
                if selected_features:
                    n_clusters = st.slider("Number of Clusters", 2, 10, 3)
                    
                    # Prepare data
                    X = df[selected_features].values
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Perform clustering
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    clusters = kmeans.fit_predict(X_scaled)
                    
                    # PCA for visualization
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_scaled)
                    
                    # Plot clusters
                    fig = px.scatter(
                        x=X_pca[:, 0],
                        y=X_pca[:, 1],
                        color=clusters,
                        title='Cluster Analysis'
                    )
                    st.plotly_chart(fig)
            
            elif analysis_type == "Event Classification":
                st.header("Event Classification")
                
                # Event parameters
                event_params = ['ML', 'Mw', 'Ms', 'ev_depth_km']
                available_params = [p for p in event_params if p in df.columns]
                
                if available_params:
                    selected_param = st.selectbox("Select Event Parameter", available_params)
                    
                    # Create classification based on parameter
                    param_bins = pd.qcut(df[selected_param], q=5)
                    
                    # Plot distribution
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=df[selected_param], nbinsx=20))
                    fig.update_layout(title=f'Distribution of {selected_param}')
                    st.plotly_chart(fig)
                    
                    # Show statistics by class
                    st.subheader("Statistics by Class")
                    st.dataframe(df.groupby(param_bins)[waveform_cols[:5]].mean())

if __name__ == "__main__":
    main()
