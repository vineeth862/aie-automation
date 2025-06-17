import pandas as pd
import fitz
import numpy as np
import cv2
class PDFAnnotator:
    def __init__(self, pdf_path,anomalyData,featureLocationAndText,lineNumberColumn):
        self.pdf_path = pdf_path
        # self.image_width, self.image_height = image_size
        self.merged_Anomalyfeature_LocationAndText = pd.DataFrame()
        self.doc = fitz.open(pdf_path)
        self.white_spaces = []
        self.occupied_spaces = []
        self.anomalyData = anomalyData 
        self.featureLocationAndText = featureLocationAndText
        self.line_number_column = lineNumberColumn
    
    
    def mergeAnomalyDataAndFeatureLocation(self):
        merged_Anomalyfeature_LocationAndText = pd.merge(self.anomalyData,self.featureLocationAndText, right_on=['lineNumber','text'], left_on=[self.line_number_column,'FeatureGroup'], how='left')
        self.merged_Anomalyfeature_LocationAndText = merged_Anomalyfeature_LocationAndText
        return merged_Anomalyfeature_LocationAndText

    def scale_coordinates(self, x1, y1, x2, y2, pdf_width, pdf_height,image_width,image_height):
        """Scales image-based coordinates to PDF-based coordinates."""
        scale_x = pdf_width / image_width
        scale_y = pdf_height / image_height
        return x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y

    def correct_rotation(self, x1, y1, x2, y2, pdf_width, pdf_height, rotation):
        """Adjusts coordinates based on PDF rotation."""
        if rotation == 90:
            return y1, pdf_width - x2, y2, pdf_width - x1
        elif rotation == 180:
            return pdf_width - x2, pdf_height - y2, pdf_width - x1, pdf_height - y1
        elif rotation == 270:
            return pdf_height - y2, x1, pdf_height - y1, x2
        return x1, y1, x2, y2
    
    
    def identify_white_spaces(self,pageNum):
        self.occupied_spaces = [] # Every time when new page is loaded, occupied spaces should be reset
        white_space_windows = []
        white_space_threshold = 0.95  # 95% whitespace
        window_size = 200
        min_valid_size = 190 # A valid window should be atleast 190x190
        overlap_ratio = 0.5
        # output_folder = r"C:\Users\Vineeth\AIE Dropbox\Vineeth S\ROO\03 Experiment Outputs/25022025_WhiteSpaces_3"
        # Create output directory if it doesn't exist
        # if not os.path.exists(output_folder):
        #     os.makedirs(output_folder)

        doc = self.doc
        page = doc[pageNum]
        pix = page.get_pixmap()
        
        # Convert PDF to numpy image
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.shape[-1] == 3 else image

        # Adaptive threshold to detect black text/characters
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10
        )

        height, width = binary.shape[:2]

         # Counter for saved images
        step_size = int(window_size * (1 - overlap_ratio))  # Step size with overlap
        window_index = 0 
        for y in range(0, height, step_size):
            for x in range(0, width, step_size):
                x1 = x
                x2 = x + window_size
                y1 = y
                y2 = y + window_size
                
                
                window = binary[y1:y2, x1:x2]
                if window.shape[0] < min_valid_size or window.shape[1] < min_valid_size:
                    continue

                # Count black pixels (text, characters, lines)
                black_pixels = cv2.countNonZero(window)
                
                total_pixels = window.size
                white_pixels = total_pixels - black_pixels
                # print(window_index,white_pixels,total_pixels,(white_pixels)/total_pixels)
                # Store if whitespace is more than 95%
                whiteSpace = (white_pixels / total_pixels)
                if (white_pixels / total_pixels) > (white_space_threshold):

                    white_space_windows.append((x, y, x + window_size, y + window_size,whiteSpace))

                    # Save window as an image
                    # output_path = os.path.join(output_folder, f"window_{window_index}_{x1}_{y1}_{x2}_{y2}_{whiteSpace}_pageNum_{pageNum}.png")
                    # cv2.imwrite(output_path, window)
                    window_index += 1
        self.white_spaces = white_space_windows
       
        # print(f"Saved {window_index} whitespace window images in '{output_folder}'.")
        # return white_space_windows

    def find_best_white_space(self, page, bbox):

        if not hasattr(self, 'white_spaces') or not self.white_spaces:
            print("No whitespace data available. Run identify_white_spaces first.")
            return bbox  # Return the same bounding box if no whitespace found

        if not hasattr(self, 'occupied_spaces'):
            self.occupied_spaces = []  # Initialize if not present

        x1, y1, x2, y2 = bbox
        text_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])  # Compute text center

        # Hyperparameters
        alpha = 60  # Probability scaling factor (high priority)
        beta = 0.005  # Distance scaling factor (lower priority)

        best_white_space = None
        max_score = -1

        for wx1, wy1, wx2, wy2, whiteSpaceProb in self.white_spaces:
            
            
            # Skip already occupied spaces
            if (wx1, wy1, wx2, wy2) in self.occupied_spaces:
                continue
            
            
            overlap = False
            for ox1, oy1, ox2, oy2 in self.occupied_spaces:
                if not (wx2 <= ox1 or wx1 >= ox2 or wy2 <= oy1 or wy1 >= oy2):  # Check for overlap
                    overlap = True
                    break

            if overlap:
                continue
            # Compute center of whitespace
            white_center = np.array([(wx1 + wx2) / 2, (wy1 + wy2) / 2])
            
            # Compute Euclidean distance
            distance = np.linalg.norm(text_center - white_center)
            
            # Apply scoring function
            score = (whiteSpaceProb ** alpha) * np.exp(-beta * distance)

            # Ensure the whitespace does not contain the text center
            if wx1 <= text_center[0] <= wx2 and wy1 <= text_center[1] <= wy2:
                continue  # Skip this whitespace as it overlaps the text center

            # Update best whitespace if score is higher
            if score > max_score:
                max_score = score
                best_white_space = (wx1, wy1, wx2, wy2)

        if best_white_space:
            self.occupied_spaces.append(best_white_space)  # Mark as occupied
            return best_white_space
        else:
            print("No suitable whitespace found. Using original bounding box.")
            return bbox  # Return original bbox if no whitespace is found

    


    def annotate_feature(self, lineNumber,line_number_column,anomaly_column):
        """Annotates only the specified lineNumber in the PDF where a single line has multiple features."""
        feature_data = self.merged_Anomalyfeature_LocationAndText[self.merged_Anomalyfeature_LocationAndText[line_number_column] == lineNumber]
        if feature_data.empty:
            print(f"Feature '{lineNumber}' not found in dataframe.")
            return
        # print(feature_data)
        feature_data = feature_data.dropna(subset=['text'])
        processed_pages = set()
        for index, row in feature_data.iterrows():
            numberOfFeatures  = len(feature_data['text'].unique())
            feature_name = row['text']

            page_number = int(row['page'])

            x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
            page = self.doc[page_number]
            pdf_width, pdf_height = page.rect.width, page.rect.height
         
            x1, y1, x2, y2 = self.scale_coordinates(x1, y1, x2, y2, pdf_width, pdf_height,row['imageWidth'],row['imageHeight'])
            x1, y1, x2, y2 = self.correct_rotation(x1, y1, x2, y2, pdf_width, pdf_height, page.rotation)
            bbox = (x1, y1, x2, y2)
            if page_number not in processed_pages:
                
                self.identify_white_spaces(page_number)
                processed_pages.add(page_number)
          # Identify whitespace on this page
            new_bbox = self.find_best_white_space(page, bbox)  # Find best whitespace

            new_bbox =self.correct_rotation(new_bbox[0],new_bbox[1],new_bbox[2],new_bbox[3],pdf_width, pdf_height, page.rotation)
            
            # Dynamically adjust annotation size based on text length
            text_length = len(row[anomaly_column])
          # If text is short (≤ 350 characters), assume new_bbox is enough
            if numberOfFeatures <=3:
                if text_length <= 350:
                    annot_rect = fitz.Rect(new_bbox)
                    
                else:
                    
                    ref_width, ref_height = 200, 200  
                    ref_char_count = 335 

                    # Calculate required area based on text length
                    text_length = len(row[anomaly_column])
                    val = (text_length * (ref_width * ref_height)) / ref_char_count  # Scales proportionally

                    # Maintain aspect ratio: sqrt(val) gives a nearly square dimension
                    side_length = (val ** 0.5)
                    # Update new_bbox dynamically
                    new_bbox = (new_bbox[0], new_bbox[1], new_bbox[0] + side_length, new_bbox[1] + side_length)

                    # Create annotation rectangle with updated size
                    annot_rect = fitz.Rect(new_bbox)
            else:
                # if text_length <= 600:
                #     annot_rect = fitz.Rect(new_bbox)
                    
                # else:
                    
                ref_width, ref_height = 200, 200  
                ref_char_count = 450 

                # Calculate required area based on text length
                text_length = len(row[anomaly_column])
                val = (text_length * (ref_width * ref_height)) / ref_char_count  # Scales proportionally

                # Maintain aspect ratio: sqrt(val) gives a nearly square dimension
                side_length = (val ** 0.5)
                # Update new_bbox dynamically
                new_bbox = (new_bbox[0], new_bbox[1], new_bbox[0] + side_length, new_bbox[1] + side_length)

                # Create annotation rectangle with updated size
                annot_rect = fitz.Rect(new_bbox)

            # annot_rect = fitz.Rect(new_bbox[0], new_bbox[1], new_bbox[0] + box_width, new_bbox[1] + box_height)
            # annot_rect = fitz.Rect(new_bbox)
            
            callout_coords = [[x1, y1],[new_bbox[0] + 10, new_bbox[1] + 10]] # End pint in first array, start point in second array
                       
            if numberOfFeatures >3:
                opacity=0.7
                fontsize=10
            else:
                opacity=1 
                fontsize=12          
            # Add annotation to PDF
            
            border_color = (1, 0, 0) if row['isAnomaly'] else (44/255, 168/255, 48/255)  # Red for anomaly, Green for normal
            text_color = border_color
            annot = page.add_freetext_annot(
            annot_rect,
            row[anomaly_column],  # Text content
            fontsize=fontsize,
            fontname="helv",
            border_width=1,
            border_color=border_color,
            text_color=text_color,
            fill_color=(1, 1, 1),  # White background
            rotate=page.rotation,  # Set text rotation to match page
            callout=callout_coords,
            line_end=4,
            opacity = opacity
            )
            annot.update()
            
            print(f"Annotated '{feature_name}' on page {page_number} at {new_bbox}")

    def save_pdf(self, output_path):
        """Saves the modified PDF."""
        self.doc.save(output_path)
        self.doc.close()

