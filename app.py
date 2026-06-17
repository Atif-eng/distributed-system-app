import streamlit as st
import pandas as pd
import numpy as np
import time

# Dask and Multiprocessing imports
from dask.distributed import Client, LocalCluster
import dask.dataframe as dd
import xgboost as xgb
from xgboost import dask as dxgb

# --- PAGE SETUP ---
st.set_page_config(page_title="Distributed Loan System", layout="wide")
st.title("Distributed Dataset Processor")
st.markdown("This UI allows you to upload a custom dataset from your laptop and process it efficiently across distributed virtual nodes (CPU cores).")

# Safe Cluster Initialization for Windows/Laptop Environment
@st.cache_resource
def setup_laptop_cluster():
    cluster = LocalCluster(n_workers=2, threads_per_worker=2, memory_limit='3GB')
    client = Client(cluster)
    return client

def main():
    client = setup_laptop_cluster()

    # Sidebar Nodes Telemetry
    scheduler_url = client.scheduler.address if client.scheduler else "Local Engine"
    st.sidebar.header("📡 Virtual Node Network")
    st.sidebar.text(f"Scheduler: {scheduler_url}")
    workers_dict = client.scheduler_info().get('workers', {})
    st.sidebar.metric("Active Workers (CPU Cores)", len(workers_dict))

    # --- DATASET UPLOAD SECTION ---
    st.header("📁 Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        # Load sample data into memory for preview
        with st.spinner("Loading dataset preview..."):
            # Reading only the first few rows to prevent memory crashes on large files
            df_preview = pd.read_csv(uploaded_file, nrows=5)
            st.success("Dataset successfully uploaded!")
            
            st.subheader("📋 Dataset Preview (First 5 Rows)")
            st.dataframe(df_preview)
            
            # Dynamic Target Column Selection
            columns = df_preview.columns.tolist()
            target_col = st.selectbox("Select your Target Column (Label):", columns, index=len(columns)-1)
            
            # Auto-select feature columns (excluding the target column)
            feature_cols = [col for col in columns if col != target_col]

        # --- BENCHMARK LAB ---
        st.header("📊 Distributed Performance Lab")
        st.write("Click the button below to start benchmarking on your uploaded dataset.")
        
        if st.button("Run Distributed System Analysis"):
            with st.spinner("Processing full dataset..."):
                # Reset file pointer to read the full dataset
                uploaded_file.seek(0)
                full_df = pd.read_csv(uploaded_file)
                
                # Basic Preprocessing: Keep only numeric columns and drop missing values to prevent model crashes
                full_df = full_df.select_dtypes(include=[np.number]).dropna()
                
                if full_df.shape[0] < 10:
                    st.error("Error: The dataset does not contain enough numeric rows for machine learning processing.")
                    return
                
                X = full_df[[c for c in feature_cols if c in full_df.columns]]
                y = full_df[target_col] if target_col in full_df.columns else full_df.iloc[:, -1]
                
                st.info(f"Processing Scale: {X.shape[0]} Rows | {X.shape[1]} Features")

            # 1. Non-Distributed (Serial) Mode
            with st.spinner("Training Non-Distributed Model (Using 1 CPU Core)..."):
                start_time = time.time()
                serial_model = xgb.XGBClassifier(n_jobs=1, tree_method="hist")
                serial_model.fit(X, y)
                time_serial = time.time() - start_time

            # 2. Distributed (Dask Cluster) Mode
            with st.spinner("Training Distributed Model (Data Sharding across Cores)..."):
                start_time = time.time()
                
                # DATA SHARDING: Dividing data into chunks based on available computing partitions
                dX = dd.from_pandas(X, npartitions=4)
                dy = dd.from_pandas(y, npartitions=4)
                
                dtrain = dxgb.DaskDMatrix(client, dX, dy)
                dxgb.train(client, {'objective': 'binary:logistic', 'tree_method': 'hist'}, dtrain)
                time_distributed = time.time() - start_time

            # --- VISUAL GRAPHS ---
            st.success("Analysis Complete! Generating Live Graphs...")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("⏱️ Speed Performance Graph")
                chart_data = pd.DataFrame({
                    "Execution Method": ["Non-Distributed (Serial)", "Distributed (Dask)"],
                    "Time (Seconds)": [time_serial, time_distributed]
                })
                st.bar_chart(chart_data.set_index("Execution Method"))
                
            with col2:
                st.subheader("⚡ System Efficiency")
                if time_serial > time_distributed:
                    speedup = (time_serial / time_distributed)
                    st.metric(label="Distributed Speedup Ratio", value=f"{speedup:.2f}x Faster")
                    st.info("Distributed processing utilized multiple CPU cores simultaneously via parallel threads.")
                else:
                    speedup = (time_serial / time_distributed)
                    st.metric(label="Distributed Speedup Ratio", value=f"{speedup:.2f}x")
                    st.warning("Small datasets usually face network/sharding communication overhead. For optimal performance gains, please try uploading a larger CSV file!")

    else:
        st.info("💡 Please upload a CSV file from your laptop to start processing.")

    # --- LIVE WORKER TELEMETRY ---
    st.header("🖥️ Active Core Resource Utilization")
    worker_metrics = []
    for addr, info in workers_dict.items():
        metrics = info.get('metrics', {})
        cpu_usage = metrics.get('cpu', 0.0)
        memory_bytes = metrics.get('memory', 0)
        tasks_active = metrics.get('executing', len(info.get('processing', {})))
        
        worker_metrics.append({
            "Core Address/ID": addr,
            "CPU Usage (%)": cpu_usage,
            "Memory Allocated (MB)": round(memory_bytes / (1024**2), 2),
            "Tasks Active": tasks_active
        })
        
    if worker_metrics:
        st.table(pd.DataFrame(worker_metrics))

if __name__ == '__main__':
    main()
