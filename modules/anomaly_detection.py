import pandas as pd
class AnomalyDetection:
    def __init__(self,input_file: str, anomaly_threshold: float, line_number_column: str ,feature_id_column :str,anomaly_LTCR:str,anomaly_STCR:str,anomaly_column:str):
        self.input_file = input_file
        self.anomaly_threshold = anomaly_threshold
        self.line_number_column = line_number_column
        self.feature_id_column = feature_id_column
        self.anomaly_LTCR = anomaly_LTCR
        self.anomaly_STCR = anomaly_STCR
        self.anomaly_column = anomaly_column

    def read_excel(self):
        """ Read the input excel file and verify whether the column provided by engineers are present in data or not"""
        try:
            # Skip rows should be removed and we need to ask engineers to upload excel where columns should be prsent on first row
            data = pd.read_excel(self.input_file,skiprows=4)
   
            column_check = self.pre_process_columns(data)
            if column_check['return_code'] != 1:
                return column_check
            return {"return_code": 1, "status": "success", "data": data}
        except Exception as e:
            return {"return_code": "AND_01", "status": f"Unable to read the input file:Error-{e}"}
        

    def pre_process_columns(self, data):
        """Function to verify whether the mandatory columns are present in data or not
        Args:
            data (pd.Dataframe): Input excel data

        Returns:
            dict: Verfied input data, status code and status
        """

        required_columns = [
            self.line_number_column,
            self.feature_id_column,
            self.anomaly_LTCR,
            self.anomaly_STCR,
            self.anomaly_column
        ]

        missing_columns = [col for col in required_columns if col not in data.columns]
  
        if missing_columns:
            return {
                'return_code': 'AND_02',
                'status': f"Missing columns: {', '.join(missing_columns)}",
                'missing_columns': missing_columns
            }
        return {'return_code': 1, 'status': 'success'}

    def filter_anomaly_line_number(self, anomaly_file: pd.DataFrame):
        """ Filter all the unique anomaly line number present in given data
        Args:
            anomaly_file (pd.DataFrame) : Verified input data

        Returns:
            dict: return_code, status , unique_anomaly_lines and filtered_anomaly_data
        """
        try:
            anomaly_file[self.anomaly_LTCR] = anomaly_file[self.anomaly_LTCR].astype(float)
            anomaly_file[self.anomaly_STCR] = anomaly_file[self.anomaly_STCR].astype(float)
        except Exception as e:
            return {'return_code': "AND_03", 'status': "Either anomaly_LTCR or anomaly_STCR has non-numeric values"}
        
        unique_anomaly_lines = set(anomaly_file[anomaly_file[self.anomaly_LTCR] <= self.anomaly_threshold][self.line_number_column].values)
        unique_anomaly_lines.update(anomaly_file[anomaly_file[self.anomaly_STCR] <= self.anomaly_threshold][self.line_number_column].values)
        
        filtered_anomaly_data = anomaly_file[anomaly_file[self.line_number_column].isin(unique_anomaly_lines)]
        return {'return_code': 1, 'status': 'success', 'uniqueAnomalyLineNumber': unique_anomaly_lines, 'data': filtered_anomaly_data}

    def filter_anomaly_feature(self, unique_anomaly_lines: set, filtered_anomaly_data: pd.DataFrame):
        """ Filter all the featureId from unique anomaly line number
        Args:
            filtered_anomaly_data (pd.DataFrame) : Filtered anomaly data where only anomaly line numbers are present
            unique_anomaly_lines (list) : List of unique anomaly line numbers

        Returns:
            dict: return_code, status , final_anomaly_df
        """
        filtered_rows = []

        for anomaly_line in unique_anomaly_lines:
            anomaly_subset = filtered_anomaly_data[filtered_anomaly_data[self.line_number_column] == anomaly_line].copy()
            anomaly_subset['FeatureGroup'] = anomaly_subset[self.feature_id_column].str.split('/').str[0]
            
            for feature_id, group in anomaly_subset.groupby('FeatureGroup'):
                has_low_anomaly = ((group[self.anomaly_LTCR] < self.anomaly_threshold) | (group[self.anomaly_STCR] < self.anomaly_threshold)).any()
                
                if has_low_anomaly:
                    valid_rows = group.dropna(subset=[self.anomaly_LTCR, self.anomaly_STCR], how='all')
                    if not valid_rows.empty:
                        lowest_row = valid_rows.loc[valid_rows[[self.anomaly_LTCR, self.anomaly_STCR]].min(axis=1).idxmin()]
                        lowest_row['isAnomaly'] = True
                        filtered_rows.append(lowest_row.to_dict())
                else:
                    condition_1 = group[[self.anomaly_LTCR, self.anomaly_STCR]].notna().any(axis=1)
                    if condition_1.any():
                        filtered_condition_1_rows = group[condition_1]
                        if not filtered_condition_1_rows.empty:
                            lowest_row = filtered_condition_1_rows.loc[filtered_condition_1_rows[[self.anomaly_LTCR, self.anomaly_STCR]].min(axis=1).idxmin()]
                            lowest_row['isAnomaly'] = False
                            filtered_rows.append(lowest_row.to_dict())
        
        final_anomaly_df = pd.DataFrame(filtered_rows)
        return {'return_code': 1, 'status': 'success', 'data': final_anomaly_df}
    

    def process_anomalies(self):
        """ Perform all the operations present in AnomalyDetection class and returns the final output """
        # Read excel and check use given column are present
        read_result = self.read_excel()
        if read_result["return_code"] != 1:
            return read_result
        anomaly_data = read_result["data"]

        #FIlter anomaly line number
        filter_line_result = self.filter_anomaly_line_number(anomaly_data)
        if filter_line_result["return_code"] != 1:
            return filter_line_result
        
        # Filter Anomaly Feature
        filter_feature_result = self.filter_anomaly_feature(
            filter_line_result["uniqueAnomalyLineNumber"], filter_line_result["data"])
        
        return {"return_code": 1, "status": "success", "filter_feature_result": filter_feature_result['data']}
