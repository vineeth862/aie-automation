import pandas as pd
from ultralytics import YOLO
from PIL import Image
import fitz
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
class FeatureTextDetection:

    def __init__(self,pdf_path,featureExtractorModelPath,textExtractorModelPath):
        self.pdf_path = pdf_path
        self.featureExtractorModelPath = featureExtractorModelPath
        self.textExtractorModelPath = textExtractorModelPath


    def extractText(self,img,c,output_folder=""):
        # model = YOLO(r"yolo Model/runs/detect/train4/weights/best.pt") 
        try:
            model = YOLO(self.textExtractorModelPath)
        except Exception as e:
            return {'return_code': "FTD_01", 'status': f'Failed to load the text extraction model,Error:{e}'}

        image = img.resize(
                                (220,440), Image.LANCZOS
                            ) 
        results  =model.predict(image, conf=0.7,verbose=False)
        detected_chars = []  # Store (character, x1) for sorting
        # cropped_image_path = os.path.join(output_folder, f"crop_{c}.png")
        # results[0].save(filename=cropped_image_path)
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # Bounding box
            conf = box.conf[0].item()  # Confidence score
            class_id = int(box.cls[0])  # Class ID (character)
            character = model.names[class_id]  # Convert ID to character

            detected_chars.append((character, x1))  # Store character with x1 position
        sorted_chars = sorted(detected_chars, key=lambda x: x[1])
        text_output = "".join([char[0] for char in sorted_chars])
        return {"return_code": 1, "status": "success", "text_output": text_output}
    
    
    def extractFeatureAndProcess(self,all_responses,uniqueAnomalyLineNumbers):
        all_responses=[responseValue.replace('"',"") for responseValue in all_responses]
        # Fix this
        common_anomalies = set(uniqueAnomalyLineNumbers) & set(all_responses)
        if len(common_anomalies)==0:
            return {'return_code': "FTD_03", 'status': f'No common Pipeline line numbers are present, Recheck the excel and PDF'}

        matches = {anomaly: [i for i, resp in enumerate(all_responses) if anomaly == resp] 
           for anomaly in common_anomalies}
        try:
            model = YOLO(self.featureExtractorModelPath)
        except Exception as e:
            return {'return_code': "FTD_02", 'status': f'Failed to load the feature extraction, Error:{e}'}

        doc = fitz.open(self.pdf_path)
        # Create an output folder
        
        # output_folder = r"C:\Users\Vineeth\AIE Dropbox\Vineeth S\ROO/actual_images_2"
        # os.makedirs(output_folder, exist_ok=True)

        # Iterate through pages and save each image into dpi=300 resolution
        pageImages = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=300)  # Render page at high resolution
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pageImages.append(img)
            # img.save(os.path.join(output_folder, f"page_{page_num+1}.png"))
        final_df = pd.DataFrame()
        # output_folder = r"C:\Users\Vineeth\AIE Dropbox\Vineeth S\ROO/03 Experiment Outputs/croppedImages_24020225_3_textExtractionResult"
        # os.makedirs(output_folder, exist_ok=True)
        
        for lineNumber, pages in tqdm(matches.items(),desc="Feature Extraction Progress"):  # Use .items() to iterate dictionary correctly
            for page in pages:
                image = pageImages[page]    
                img_width, img_height = image.size

                # Sliding window parameters
                window_size = 640  # Adjust based on your dataset
                stride = 300  # Move by window size

                detected_texts = []
                c = 1

                # Iterate over the image using a sliding window
                for y in range(0, img_height - window_size + 1, stride):
                    for x in range(0, img_width - window_size + 1, stride):
                        cropped_img = image.crop((x, y, x + window_size, y + window_size))
                        cropped_img_np = np.array(cropped_img)

                        results = model.predict(cropped_img_np, conf=0.5,verbose=False)
                        
                        for result in results:
                            for box in result.boxes.xyxy:
                                x1, y1, x2, y2 = map(int, box)

                                # Crop detected object
                                detected_region = cropped_img.crop((x1, y1, x2, y2))
                                detected_region = detected_region.convert("L")  # Convert to grayscale
                                detected_region = detected_region.filter(ImageFilter.SHARPEN)  # Sharpen
                                enhancer = ImageEnhance.Contrast(detected_region)
                                detected_region = enhancer.enhance(3)  # Increase contrast

                                text = self.extractText(detected_region,c)
                           
                                if text['return_code'] == 1:
                                    text = text['text_output']
                                else:  
                                    return text
                                detected_texts.append((x1 + x, y1 + y, x2 + x, y2 + y, text))
                                c += 1

                # Create DataFrame for detected text
                detected_df = pd.DataFrame(detected_texts, columns=["x1", "y1", "x2", "y2", "text"])
        
                uniqueFeature = set()
                featureCoordinates = []
                for _, row in detected_df.iterrows():
                    feature = row['text'].strip()
                    if len(feature) > 2 and feature.startswith("F"):
                        short_feature = feature[:5]  # Keep only the first 5 characters max

                        if short_feature not in uniqueFeature:
                            uniqueFeature.add(short_feature)

                            # Append processed row
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
        
               
        return {"return_code": 1, "status": "success", "final_df": final_df}
