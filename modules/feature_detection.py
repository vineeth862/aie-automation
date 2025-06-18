import streamlit as st
import pandas as pd
from ultralytics import YOLO
from PIL import Image
import fitz
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import time

class FeatureTextDetection:
    def __init__(self, pdf_path, featureExtractorModelPath, textExtractorModelPath):
        self.pdf_path = pdf_path
        self.featureExtractorModelPath = featureExtractorModelPath
        self.textExtractorModelPath = textExtractorModelPath

    def extractText(self, img, c, output_folder=""):
        try:
            model = YOLO(self.textExtractorModelPath)
        except Exception as e:
            return {'return_code': "FTD_01", 'status': f'Failed to load the text extraction model,Error:{e}'}

        image = img.resize((220, 440), Image.LANCZOS) 
        results = model.predict(image, conf=0.7, verbose=False)
        detected_chars = []
        
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            class_id = int(box.cls[0])
            character = model.names[class_id]
            detected_chars.append((character, x1))
            
        sorted_chars = sorted(detected_chars, key=lambda x: x[1])
        text_output = "".join([char[0] for char in sorted_chars])
        return {"return_code": 1, "status": "success", "text_output": text_output}
    
    def extractFeatureAndProcess(self, all_responses, uniqueAnomalyLineNumbers):
        all_responses = [responseValue.replace('"', "") for responseValue in all_responses]
        
        common_anomalies = set(uniqueAnomalyLineNumbers) & set(all_responses)
        if len(common_anomalies) == 0:
            return {'return_code': "FTD_03", 'status': f'No common Pipeline line numbers are present, Recheck the excel and PDF'}

        matches = {anomaly: [i for i, resp in enumerate(all_responses) if anomaly == resp] 
                  for anomaly in common_anomalies}
        
        try:
            model = YOLO(self.featureExtractorModelPath)
        except Exception as e:
            return {'return_code': "FTD_02", 'status': f'Failed to load the feature extraction, Error:{e}'}

        doc = fitz.open(self.pdf_path)
        
        # Extract page images
        pageImages = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pageImages.append(img)
        
        final_df = pd.DataFrame()
        
        # Streamlit progress bar setup
        total_items = sum(len(pages) for pages in matches.values())
        progress_bar = st.progress(0)
        status_text = st.empty()
        current_item = 0
        
        for lineNumber, pages in matches.items():
            for page in pages:
                current_item += 1
                
                # Update progress bar
                progress = current_item / total_items
                progress_bar.progress(progress)
                status_text.text(f'Processing line {lineNumber}, page {page+1}... ({current_item}/{total_items})')
                
                image = pageImages[page]    
                img_width, img_height = image.size

                window_size = 640
                stride = 300
                detected_texts = []
                c = 1

                for y in range(0, img_height - window_size + 1, stride):
                    for x in range(0, img_width - window_size + 1, stride):
                        cropped_img = image.crop((x, y, x + window_size, y + window_size))
                        cropped_img_np = np.array(cropped_img)

                        results = model.predict(cropped_img_np, conf=0.5, verbose=False)
                        
                        for result in results:
                            for box in result.boxes.xyxy:
                                x1, y1, x2, y2 = map(int, box)
                                detected_region = cropped_img.crop((x1, y1, x2, y2))
                                detected_region = detected_region.convert("L")
                                detected_region = detected_region.filter(ImageFilter.SHARPEN)
                                enhancer = ImageEnhance.Contrast(detected_region)
                                detected_region = enhancer.enhance(3)

                                text = self.extractText(detected_region, c)
                           
                                if text['return_code'] == 1:
                                    text = text['text_output']
                                else:  
                                    return text
                                detected_texts.append((x1 + x, y1 + y, x2 + x, y2 + y, text))
                                c += 1

                detected_df = pd.DataFrame(detected_texts, columns=["x1", "y1", "x2", "y2", "text"])
        
                uniqueFeature = set()
                featureCoordinates = []
                for _, row in detected_df.iterrows():
                    feature = row['text'].strip()
                    if len(feature) > 2 and feature.startswith("F"):
                        short_feature = feature[:5]

                        if short_feature not in uniqueFeature:
                            uniqueFeature.add(short_feature)
                            featureCoordinates.append({
                                "x1": row["x1"],
                                "y1": row["y1"],
                                "x2": row["x2"],
                                "y2": row["y2"],
                                "text": short_feature,
                                "firstLetterReplaced": False
                            })
                            
                feature_df = pd.DataFrame(featureCoordinates)
                feature_df['page'] = page
                feature_df['lineNumber'] = lineNumber
                feature_df['imageWidth'], feature_df['imageHeight'] = image.size
                final_df = pd.concat([final_df, feature_df], ignore_index=True)
        
        # Complete the progress bar
        progress_bar.progress(1.0)
        status_text.text('Feature extraction completed!')
        
        return {"return_code": 1, "status": "success", "final_df": final_df}
