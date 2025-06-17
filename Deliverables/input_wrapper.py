import os
from mainApp import execute_markup
input_file= r"C:\Users\Vineeth\AIE Dropbox\Vineeth S\Data Science Projects (DSP)\02 Clients\02 Rumaila Operating Organization (ROO)\07 Anomaly Markup on Isometric\01 Data\Sample 3\SHA-0-VL-3.xlsx"
pdf_path = r"C:\Users\Vineeth\AIE Dropbox\Vineeth S\Data Science Projects (DSP)\02 Clients\02 Rumaila Operating Organization (ROO)\07 Anomaly Markup on Isometric\01 Data\Sample 3\SHA-0-VL-3.pdf"
output_path = r"C:\Users\Vineeth\AIE Dropbox\Vineeth S\Data Science Projects (DSP)\02 Clients\02 Rumaila Operating Organization (ROO)\07 Anomaly Markup on Isometric\03 Sample Output/160625"

#Anomaly
anomaly_threshold=3
line_number_column='Piping tag/line number'
feature_id_column='Feature ID'
anomaly_column='Anomaly Summary'
anomaly_LTCR='RL @ LTCR'
anomaly_STCR='RL @ STCR'

status = execute_markup(input_file,
               pdf_path,
               output_path,
               anomaly_threshold,
              line_number_column,
              feature_id_column,
              anomaly_column,
              anomaly_LTCR,
              anomaly_STCR,
              )
print(status)