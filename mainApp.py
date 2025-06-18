# Import Neccessary Packages
import os
import json
import time
import dotenv
import base64
import io
from typing import Annotated, List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import pymupdf
import fitz
from PIL import Image
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageEnhance, ImageFilter
import re
from scipy.ndimage import rotate
from tqdm import tqdm
from langchain_anthropic import ChatAnthropic
import zipfile
import streamlit as st

## Streamlit configuration
os.environ["LANGCHAIN_API_KEY"]=st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=st.secrets["LANGCHAIN_PROJECT"]
os.environ['anthropic_api_key']=st.secrets["CLAUDE_API_KEY"]

import streamlit as st
from modules.anomaly_detection import AnomalyDetection
from modules.line_number_extractor import PDFLineExtractor
from modules.feature_detection import FeatureTextDetection
from modules.pdf_annotator import PDFAnnotator

def load_yolo_model_safely(model_name, description="model"):
    """Safely load YOLO models for Streamlit Cloud"""
    try:
        from pathlib import Path
        
        # Get the script directory
        script_dir = Path(__file__).parent
        model_path = script_dir / model_name
        
        # Debug: Check if file exists
        st.sidebar.write(f"🔍 Looking for {description}: {model_path}")
        
        if not model_path.exists():
            st.sidebar.error(f"❌ {description} not found: {model_path}")
            st.error(f"Model file not found: {model_name}")
            return None
            
        # Load the YOLO model
        model = YOLO(str(model_path))
        st.sidebar.success(f"✅ Loaded {description}: {model_name}")
        return model
        
    except ImportError as e:
        st.sidebar.error(f"❌ Import error for {description}")
        st.error("YOLO (ultralytics) package not found. Please install it: pip install ultralytics")
        return None
    except Exception as e:
        st.sidebar.error(f"❌ Error loading {description}: {str(e)}")
        st.error(f"Error loading {description} ({model_name}): {str(e)}")
        return None

def execute_markup(input_file,
            pdf_path,
            output_path,
            anomaly_threshold,
            line_number_column,
            feature_id_column,
            anomaly_column,
            anomaly_LTCR,
            anomaly_STCR):
    
    start_time = time.time()
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load models safely
    st.info("🔄 Loading AI models...")
    
    # Load feature extractor model
    feature_model = load_yolo_model_safely("best.pt", "Feature Extractor Model")
    if feature_model is None:
        return {
            'return_code': 0,
            'status': 'Failed to load feature extractor model (best.pt). Please check if the file exists in the root directory.'
        }
    
    # Load text extractor model  
    text_model = load_yolo_model_safely("best2.pt", "Text Extractor Model")
    if text_model is None:
        return {
            'return_code': 0,
            'status': 'Failed to load text extractor model (best2.pt). Please check if the file exists in the root directory.'
        }
    
    # Convert models back to paths for the existing code structure
    from pathlib import Path
    script_dir = Path(__file__).parent
    featureExtractorModelPath = str(script_dir / "best.pt")
    textExtractorModelPath = str(script_dir / "best2.pt")
    
    st.success("✅ All models loaded successfully!")
    
    # Process anomaly detection
    st.info("🔄 Processing anomaly detection...")
    anomalyDetectionObj = AnomalyDetection(input_file = input_file,
                                        anomaly_threshold=anomaly_threshold,
                                        line_number_column=line_number_column,
                                        feature_id_column=feature_id_column,
                                        anomaly_column=anomaly_column,
                                        anomaly_LTCR=anomaly_LTCR,
                                        anomaly_STCR=anomaly_STCR
                                        )
    anomalyResult = anomalyDetectionObj.process_anomalies()
    if anomalyResult['return_code'] != 1:
        st.error(f"Anomaly detection failed: {anomalyResult['status']}")
        return anomalyResult

    filtered_anomaly_data = anomalyResult['filter_feature_result']
    # filtered_anomaly_data.to_excel(f"{output_path}/AnomalyData.xlsx",index=False)
    st.success("✅ Anomaly detection completed!")

    # Extract Line Numbers
    st.info("🔄 Extracting line numbers from PDF...")
    extractor = PDFLineExtractor(pdf_path,batch_size=5)
    lineNumberOnEachPage = extractor.extract_line_numbers()
    if lineNumberOnEachPage['return_code'] == 1:
        lineNumbers = (lineNumberOnEachPage['line_numbers'])
        st.success("✅ Line number extraction completed!")
    else:
        st.error(f"Line number extraction failed: {lineNumberOnEachPage.get('status', 'Unknown error')}")
        return lineNumberOnEachPage
    
    uniqueLineNumbers = list(filtered_anomaly_data[line_number_column].unique())
    
    # Extract Feature and Process
    st.info("🔄 Processing feature and text detection...")
    featureTextDetector = FeatureTextDetection(pdf_path=pdf_path,
                                             featureExtractorModelPath=featureExtractorModelPath,
                                             textExtractorModelPath=textExtractorModelPath)
    featureTextOnEachPage = featureTextDetector.extractFeatureAndProcess(lineNumbers,uniqueLineNumbers)
    if featureTextOnEachPage['return_code'] != 1:
        st.error(f"Feature detection failed: {featureTextOnEachPage.get('status', 'Unknown error')}")
        return featureTextOnEachPage
    featureTextOnEachPage = featureTextOnEachPage['final_df']
    # featureTextOnEachPage.to_excel(f"{output_path}/FeatureData.xlsx",index=False)
    st.success("✅ Feature and text detection completed!")

    # Annotation/Markup
    st.info("🔄 Creating PDF annotations...")
    annotator = PDFAnnotator(pdf_path, filtered_anomaly_data,featureTextOnEachPage,lineNumberColumn=line_number_column)
    merged_data = annotator.mergeAnomalyDataAndFeatureLocation()
    merged_data['isAnoted'] = merged_data['text'].apply(lambda x: False if pd.isna(x) else True)
    # merged_data.to_excel(f"{output_path}/AnotationData.xlsx",index=False)

    # Annotate each line number
    progress_bar = st.progress(0)
    unique_lines = filtered_anomaly_data[line_number_column].unique()
    for i, lineNumber in enumerate(unique_lines):
        annotator.annotate_feature(lineNumber,line_number_column,anomaly_column)
        progress_bar.progress((i + 1) / len(unique_lines))
    
    # annotator.save_pdf(f'{output_path}/AnomalyMarked.pdf')
    st.success("✅ PDF annotation completed!")
    
    end_time = time.time()
    execution_time = end_time - start_time
    def create_zip_download(filtered_anomaly_data, featureTextOnEachPage, merged_data, pdf_annotator):
        """Create a ZIP file containing all 4 files"""
        zip_buffer = io.BytesIO()
    
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Convert DataFrames to Excel bytes and add to ZIP
            excel_files = [
                (filtered_anomaly_data, "AnomalyData.xlsx"),
                (featureTextOnEachPage, "FeatureData.xlsx"),
                (merged_data, "AnnotationData.xlsx")
            ]
            
            for df, filename in excel_files:
                excel_buffer = io.BytesIO()
                df.to_excel(excel_buffer, index=False)
                zip_file.writestr(filename, excel_buffer.getvalue())
            
            # Get PDF bytes from PDFAnnotator object and add to ZIP
            # Method 1: If PDFAnnotator has a save() method that returns bytes
            try:
                pdf_bytes = pdf_annotator.save()  # or pdf_annotator.output()
                zip_file.writestr("AnomalyMarked.pdf", pdf_bytes)
            except AttributeError:
                # Method 2: If PDFAnnotator has a different method to get bytes
                try:
                    pdf_buffer = io.BytesIO()
                    pdf_annotator.save(pdf_buffer)  # Save to buffer
                    zip_file.writestr("AnomalyMarked.pdf", pdf_buffer.getvalue())
                except:
                    # Method 3: If you need to save to a temporary file first
                    import tempfile
                    with tempfile.NamedTemporaryFile() as tmp_file:
                        pdf_annotator.save(tmp_file.name)
                        tmp_file.seek(0)
                        zip_file.writestr("AnomalyMarked.pdf", tmp_file.read())
        
        return zip_buffer.getvalue()
    st.success(f"🎉 All processing completed in {execution_time:.2f} seconds!")
    zip_data = create_zip_download(
        filtered_anomaly_data=filtered_anomaly_data,
        featureTextOnEachPage=featureTextOnEachPage, 
        merged_data=merged_data,
        pdf_annotator=annotator
    )
    
    # Single download button in center
    st.markdown("<br>", unsafe_allow_html=True)  # Add some space
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.download_button(
            label="📦 Download All Results (ZIP)",
            data=zip_data,
            file_name="anomaly_analysis_results.zip",
            mime="application/zip",
            help="Downloads all 4 files: AnomalyData.xlsx, FeatureData.xlsx, AnnotationData.xlsx, AnomalyMarked.pdf"
        )
    
    # Optional: Show what's included
    st.info("📁 ZIP contains: AnomalyData.xlsx • FeatureData.xlsx • AnnotationData.xlsx • AnomalyMarked.pdf")
    return {
        'return_code': 1,
        'status': 'Success', 
        "execution_time": execution_time
    }
