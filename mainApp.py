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
# from langchain_openai import ChatOpenAI
# from langchain_ollama import ChatOllama
# from langchain_ollama.llms import OllamaLLM
from langchain_core.messages import HumanMessage, SystemMessage
import cv2
import numpy as np
# import pytesseract
from ultralytics import YOLO
from PIL import Image, ImageEnhance, ImageFilter
import re
from scipy.ndimage import rotate
from tqdm import tqdm
from langchain_anthropic import ChatAnthropic

import streamlit as st
# from dotenv import load_dotenv
# load_dotenv(override=True)

# # Import keys 
# os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
# ## Langsmith Tracking
# # print(os.getenv("OPENAI_API_KEY"))
# os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"]="true"
# os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
# os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API")
# os.environ['anthropic_api_key']=os.getenv("CLAUDE_API_KEY")

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
    
    os.makedirs(output_path, exist_ok=True)
    # featureExtractorModelPath=r"models/feature_identication_trained_model/weights/best.pt"
    # textExtractorModelPath=r"models/text_identication_trained_model/weights/best.pt"
    from pathlib import Path

    # Get absolute paths relative to your script location
    script_dir = Path(__file__).parent
    # featureExtractorModelPath = str(script_dir / "models" / "feature_identication_trained_model" / "weights" / "best.pt")
    # textExtractorModelPath = str(script_dir / "models" / "text_identication_trained_model" / "weights" / "best.pt")
    featureExtractorModelPath=r"./models\feature_identication_trained_model\weights\best.pt"
    textExtractorModelPath=r"./models\text_identification_trained_model\weights\best.pt"
    # C:\Users\Vineeth\AIE Dropbox\Vineeth S\Code\DataScience\rumailaOperatingOrganization\anomalyMarkupAutomation\finalDelivery\feature_identication_trained_model\weights\best.pt
    # Ask input from engineers for all the parameters listed below
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
        print(anomalyResult['status'])
        return anomalyResult

    filtered_anomaly_data = anomalyResult['filter_feature_result']
    filtered_anomaly_data.to_excel(f"{output_path}/AnomalyData.xlsx",index=False)

    # Extract Line Numbers
    extractor = PDFLineExtractor(pdf_path,batch_size=5)
    lineNumberOnEachPage = extractor.extract_line_numbers()
    if lineNumberOnEachPage['return_code'] == 1:
        lineNumbers = (lineNumberOnEachPage['line_numbers'])
        
    else:
        return lineNumberOnEachPage
    # lineNumbers = ['8"-VH-SHAM-52-1038-01AX', '8"-VH-SHAM-52-1042-01AX', '8"-VL-SHAM-53-1004-01AX', '8"-VL-SHAM-53-1008-01AX', '8"-VL-SHAM-53-1012-01AX', '8"-VL-SHAM-53-1016-01AX', '10"-VL-SHAM-53-1023-01AX', '10"-VL-SHAM-53-1018-01AX', '10"-VL-SHAM-53-1022-01AX', '10"-VL-SHAM-53-1017-01AX', '10"-VL-SHAM-53-1034-01AX', '10"-VL-SHAM-53-1044-01AX', '10"-VL-SHAM-53-1033-01AX', '10"-VL-SHAM-53-1043-01AX', '8"-VH-SHAM-52-2048-01AX', '8"-VH-SHAM-52-2052-01AX', '8"-VL-SHAM-53-2003-01AX', '8"-VL-SHAM-53-2004-01AX', '8"-VL-SHAM-53-2007-01AX', '8"-VL-SHAM-53-2008-01AX', '8"-VL-SHAM-53-2011-01AX', '8"-VL-SHAM-53-2012-01AX', '8"-VL-SHAM-53-2015-01AX', '8"-VL-SHAM-53-2016-01AX', '12"-VL-SHAM-53-2041-01AX', '12"-VL-SHAM-53-2022-01AX', '12"-VL-SHAM-53-2055-01AX', '6"-VL-SHAM-53-2028-01AX', '12"-VH-SHAM-52-3021-01AX', '20"-VL-SHAM-51-0085-01AX', '20"-VL-SHAM-51-0086-01AX', '20"-VL-SHAM-51-0087-01AX', '20"-VL-SHAM-51-0087-01AX', '20"-VL-SHAM-51-0088-01AX', '4"-VL-SHAM-53-0058-01AX', '4"-VL-SHAM-53-0059-01AX', '4"-VL-SHAM-53-0057-01AX', '8"-PG-SHAM-87-0001-01AX', '8"-VL-SHAM-53-0063-01AX', '4"-VL-SHAM-53-0008-01AX', '24"-SHAM-221-PJ1020-A01E2B-NI', '10"-SHAM-221-PJ1032-A03E2B-NI', '16"-SHAM-221-PJ1035-A03E2B-NI', '24"-SHAM-503-PJ1031-A01E2B-NI', '24"-SHAM-221-PJ2020-A01E2B-NI', '10"-SHAM-221-PJ2032-A03E2B-NI', '16"-SHAM-221-PJ2035-A03E2B-NI', '24"-SHAM-503-PJ2031-A01E2B-NI', '24"-SHAM-221-PJ3020-A01E2B-NI', '10"-SHAM-221-PJ3032-A03E2B-NI', '16"-SHAM-221-PJ3035-A03E2B-NI', '24"-SHAM-503-PJ3031-A01E2B-NI', '24"-SHAM-221-PJ3120-A01E2B-NI', '10"-SHAM-221-PJ3132-A03E2B-NI', '16"-SHAM-221-PJ3135-A03E2B-NI', '24"-SHAM-503-PJ3131-A01E2B-NI', '0000RO-S-DG-DG10-PI-ISO-1005-001']
    
    uniqueLineNumbers = list(filtered_anomaly_data[line_number_column].unique())
    # Extract Feature and Process
    featureTextDetector = FeatureTextDetection(pdf_path=pdf_path,featureExtractorModelPath=featureExtractorModelPath,textExtractorModelPath=textExtractorModelPath)
    featureTextOnEachPage = featureTextDetector.extractFeatureAndProcess(lineNumbers,uniqueLineNumbers)
    if featureTextOnEachPage['return_code'] != 1:
        return featureTextOnEachPage
    featureTextOnEachPage = featureTextOnEachPage['final_df']

    featureTextOnEachPage.to_excel(f"{output_path}/FeatureData.xlsx",index=False)

    # Annotation/Markup
    # filtered_anomaly_data = pd.read_excel(f"{output_path}/AnomalyData.xlsx")
    # featureTextOnEachPage = pd.read_excel(f"{output_path}/FeatureData.xlsx")
    annotator = PDFAnnotator(pdf_path, filtered_anomaly_data,featureTextOnEachPage,lineNumberColumn=line_number_column)
    merged_data = annotator.mergeAnomalyDataAndFeatureLocation()
    merged_data['isAnoted'] = merged_data['text'].apply(lambda x: False if pd.isna(x) else True)
    merged_data.to_excel(f"{output_path}/AnotationData.xlsx",index=False)

    for lineNumber in filtered_anomaly_data[line_number_column].unique():
        annotator.annotate_feature(lineNumber,line_number_column,anomaly_column)
    
    annotator.save_pdf(f'{output_path}/AnomalyMarked.pdf')
    end_time = time.time()
    return {'status':'1', "execution_time":end_time-start_time}
    # print(f"Execution time: {end_time - start_time:.2f} seconds")




