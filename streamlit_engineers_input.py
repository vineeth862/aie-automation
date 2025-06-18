import streamlit as st
import os
from pathlib import Path
import traceback
import sys

# Must be the very first Streamlit command
st.set_page_config(
    page_title="Anomaly Markup Tool", 
    page_icon="📊", 
    layout="wide"
)

# Debug information
# st.sidebar.write("Debug Info:")
# st.sidebar.write(f"Python version: {sys.version}")
# st.sidebar.write(f"Streamlit version: {st.__version__}")
# st.sidebar.write(f"Current directory: {os.getcwd()}")

# Import your main function
try:
    from mainApp import execute_markup
    # st.sidebar.success("✅ mainApp imported successfully")
except ImportError as e:
    # st.sidebar.error(f"❌ Import error: {e}")
    st.error("Could not import 'execute_markup' from 'mainApp'. Please ensure mainApp.py is in the same directory.")
    st.write("Files in current directory:")
    for file in os.listdir("."):
        st.write(f"- {file}")
    st.stop()
except RuntimeError as e:
    if "torch::class_" in str(e):
        # st.sidebar.error("❌ PyTorch compatibility error detected")
        st.error("🔧 PyTorch Compatibility Issue Detected")
        st.markdown("""
        **This error is typically caused by PyTorch version incompatibility. Try these solutions:**
        
        ### Solution 1: Update PyTorch
        ```bash
        pip uninstall torch torchvision torchaudio
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        ```
        
        ### Solution 2: Create Fresh Virtual Environment
        ```bash
        python -m venv new_env
        new_env\\Scripts\\activate  # Windows
        pip install streamlit torch torchvision torchaudio
        ```
        
        ### Solution 3: Check Your Dependencies
        ```bash
        pip list | grep torch
        ```
        
        ### Solution 4: If using CUDA
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ```
        """)
        
        with st.expander("Show full error details"):
            st.code(str(e))
        st.stop()
    else:
        # st.sidebar.error(f"❌ Runtime error: {e}")
        st.error(f"Runtime error occurred: {e}")
        st.stop()
except Exception as e:
    # st.sidebar.error(f"❌ Unexpected error: {e}")
    st.error(f"Unexpected error loading mainApp: {e}")
    with st.expander("Show error details"):
        st.code(traceback.format_exc())
    st.stop()

def main():
    
    st.title("🔍 Anomaly Markup Tool")
    st.markdown("Upload your files and configure parameters to execute anomaly markup analysis.")
    
    # Create two columns for better layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 File Upload")
        
        # File uploaders
        uploaded_excel = st.file_uploader(
            "Upload Excel File",
            type=['xlsx', 'xls'],
            help="Upload your Excel file containing the data"
        )
        
        uploaded_pdf = st.file_uploader(
            "Upload PDF File",
            type=['pdf'],
            help="Upload your PDF file"
        )
        
        # Output directory input
        output_path = "./output"
        
        # Handle uploaded files
        input_file = None
        pdf_path = None
        
        if uploaded_excel is not None:
            # Save uploaded Excel file temporarily
            temp_excel_path = f"temp_{uploaded_excel.name}"
            with open(temp_excel_path, "wb") as f:
                f.write(uploaded_excel.getbuffer())
            input_file = temp_excel_path
            st.success(f"✅ Excel file uploaded: {uploaded_excel.name}")
        
        if uploaded_pdf is not None:
            # Save uploaded PDF file temporarily
            temp_pdf_path = f"temp_{uploaded_pdf.name}"
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_pdf.getbuffer())
            pdf_path = temp_pdf_path
            st.success(f"✅ PDF file uploaded: {uploaded_pdf.name}")
    
    with col2:
        st.header("⚙️ Configuration Parameters")
        
        # Anomaly threshold
        anomaly_threshold = st.number_input(
            "Anomaly Threshold",
            min_value=0.0,
            max_value=10.0,
            value=3.0,
            step=0.1,
            help="Threshold value for anomaly detection"
        )
        
        # Column names
        line_number_column = st.text_input(
            "Line Number Column",
            value="Piping tag/line number",
            help="Name of the column containing line numbers"
        )
        
        feature_id_column = st.text_input(
            "Feature ID Column",
            value="Feature ID",
            help="Name of the column containing feature IDs"
        )
        
        anomaly_column = st.text_input(
            "Anomaly Column",
            value="Anomaly Summary",
            help="Name of the column containing anomaly summaries"
        )
        
        anomaly_LTCR = st.text_input(
            "Anomaly LTCR Column",
            value="RL @ LTCR",
            help="Name of the column for LTCR anomalies"
        )
        
        anomaly_STCR = st.text_input(
            "Anomaly STCR Column",
            value="RL @ STCR",
            help="Name of the column for STCR anomalies"
        )
    
    # File validation section
    st.subheader("📋 File Status")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        input_exists = uploaded_excel is not None
        st.write(f"Excel file: {'✅ Ready' if input_exists else '❌ Not uploaded'}")
    
    with col4:
        pdf_exists = uploaded_pdf is not None
        st.write(f"PDF file: {'✅ Ready' if pdf_exists else '❌ Not uploaded'}")
    
    # with col5:
    #     output_dir_ready = bool(output_path.strip())
    #     st.write(f"Output path: {'✅ Set' if output_dir_ready else '❌ Not set'}")
    
    # Execute button
    st.markdown("---")
    
    if st.button("🚀 Execute Markup Analysis", type="primary", use_container_width=True):
        # Validate inputs
        if uploaded_excel is None or uploaded_pdf is None:
            st.error("Please upload both Excel and PDF files.")
            return
        if not output_path.strip():
            st.error("Please specify an output directory path.")
            return
        # Execute the function
        with st.spinner("Executing markup analysis... This may take a few minutes."):
            try:
                # Call the main function
                result = execute_markup(
                    input_file,
                    pdf_path,
                    output_path,
                    anomaly_threshold,
                    line_number_column,
                    feature_id_column,
                    anomaly_column,
                    anomaly_LTCR,
                    anomaly_STCR
                )
                
                st.success("✅ Markup analysis completed successfully!")
                # Clean up temporary files
                try:
                    if input_file and os.path.exists(input_file):
                        os.remove(input_file)
                    if pdf_path and os.path.exists(pdf_path):
                        os.remove(pdf_path)
                except:
                    pass  # Ignore cleanup errors
                
            except Exception as e:
                st.error(f"❌ An error occurred during execution:")
                st.code(str(e))
                
                # Show detailed error in expander
                with st.expander("Show detailed error trace"):
                    st.code(traceback.format_exc())
                
                # Clean up temporary files even on error
                try:
                    if input_file and os.path.exists(input_file):
                        os.remove(input_file)
                    if pdf_path and os.path.exists(pdf_path):
                        os.remove(pdf_path)
                except:
                    pass
    
    # # Information section
    # st.markdown("---")
    # st.subheader("ℹ️ Instructions")
    # st.markdown("""
    # ### 📋 How to Use:
    # 1. **Upload Files**: Drag & drop or browse to upload your Excel and PDF files
    # 2. **Set Output Path**: Specify where results should be saved (e.g., `./output` or `C:/output`)
    # 3. **Configure Parameters**: Adjust anomaly threshold and column names if needed
    # 4. **Execute**: Click the button to run the analysis
    
    # **Note**: Make sure the `mainApp.py` file is in the same directory as this app.
    # """)

if __name__ == "__main__":
    main()
