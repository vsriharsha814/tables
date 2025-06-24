"""Data extraction agent for processing Excel files after format detection."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime

from agents.base_agent import BaseAgent
from utils.preprocessing import DataPreprocessor

logger = logging.getLogger(__name__)


class DataExtractionAgent(BaseAgent):
    """Extract and structure data from Excel files based on detected format."""
    
    def __init__(self):
        """Initialize the data extraction agent."""
        self.preprocessor = DataPreprocessor()
        
    def process(self, file_path: str, format_info: Dict[str, Any], 
                header_row: Optional[int] = None) -> Dict[str, Any]:
        """Extract data from file based on detected format.
        
        Args:
            file_path: Path to the Excel file
            format_info: Format detection results
            header_row: Optional header row index
            
        Returns:
            Dict with extracted data and metadata
        """
        logger.info(f"Extracting data from {file_path} with format: {format_info.get('format_name')}")
        
        try:
            # Read the Excel file
            df = self._read_excel_file(file_path, header_row)
            
            # Preprocess the data
            processed_df = self._preprocess_dataframe(df, format_info)
            
            # Extract structured data
            extracted_data = self._extract_structured_data(processed_df, format_info)
            
            # Validate extracted data
            validation_results = self._validate_data(extracted_data, format_info)
            
            return {
                'success': True,
                'extracted_data': extracted_data,
                'metadata': {
                    'total_rows': len(processed_df),
                    'total_columns': len(processed_df.columns),
                    'data_types': self.preprocessor.detect_data_types(processed_df),
                    'missing_values': self._count_missing_values(processed_df),
                    'validation_results': validation_results
                },
                'sample_data': self.preprocessor.extract_sample_data(processed_df, 3)
            }
            
        except Exception as e:
            logger.error(f"Data extraction failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'extracted_data': None
            }
    
    def _read_excel_file(self, file_path: str, header_row: Optional[int] = None) -> pd.DataFrame:
        """Read Excel file with proper header detection."""
        file_path = Path(file_path)
        
        if header_row is not None:
            # Use specified header row
            df = pd.read_excel(file_path, header=header_row)
        else:
            # Auto-detect header row
            df = pd.read_excel(file_path, header=None)
            detected_header = self.preprocessor.find_header_row(df)
            df = pd.read_excel(file_path, header=detected_header)
        
        logger.info(f"Read Excel file with {len(df)} rows and {len(df.columns)} columns")
        return df
    
    def _preprocess_dataframe(self, df: pd.DataFrame, format_info: Dict[str, Any]) -> pd.DataFrame:
        """Preprocess the dataframe based on format requirements."""
        # Clean column names
        df.columns = self.preprocessor.clean_column_names(df.columns)
        
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Handle missing values
        df = self._handle_missing_values(df, format_info)
        
        # Convert data types based on format expectations
        df = self._convert_data_types(df, format_info)
        
        logger.info(f"Preprocessed dataframe: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame, format_info: Dict[str, Any]) -> pd.DataFrame:
        """Handle missing values based on format requirements."""
        format_name = format_info.get('format_name', '')
        
        # Different strategies for different formats
        if 'employee' in format_name.lower():
            # For employee data, fill missing with appropriate defaults
            df = df.fillna({
                'Employee ID': 'UNKNOWN',
                'Name': 'Unknown',
                'Email': '',
                'Department': 'Unknown'
            })
        elif 'financial' in format_name.lower():
            # For financial data, fill numeric columns with 0
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(0)
        else:
            # Default: fill with empty string for strings, 0 for numbers
            df = df.fillna('')
        
        return df
    
    def _convert_data_types(self, df: pd.DataFrame, format_info: Dict[str, Any]) -> pd.DataFrame:
        """Convert data types based on format expectations."""
        format_name = format_info.get('format_name', '')
        
        # Common date columns to convert
        date_columns = ['Date', 'Created Date', 'Updated Date', 'Start Date', 'End Date']
        
        for col in df.columns:
            if any(date_col.lower() in col.lower() for date_col in date_columns):
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass
            
            # Convert numeric columns
            elif any(num_col in col.lower() for num_col in ['amount', 'salary', 'cost', 'price', 'quantity']):
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
        
        return df
    
    def _extract_structured_data(self, df: pd.DataFrame, format_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract structured data from the dataframe."""
        format_name = format_info.get('format_name', '')
        
        # Convert dataframe to list of dictionaries
        data = []
        for _, row in df.iterrows():
            # Convert row to dict, handling datetime objects
            row_dict = {}
            for col, val in row.items():
                if pd.isna(val):
                    row_dict[col] = None
                elif isinstance(val, datetime):
                    row_dict[col] = val.isoformat()
                elif isinstance(val, (np.integer, np.floating)):
                    row_dict[col] = float(val) if isinstance(val, np.floating) else int(val)
                else:
                    row_dict[col] = str(val)
            
            data.append(row_dict)
        
        logger.info(f"Extracted {len(data)} structured records")
        return data
    
    def _validate_data(self, data: List[Dict[str, Any]], format_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate extracted data against format requirements."""
        format_name = format_info.get('format_name', '')
        validation_results = {
            'total_records': len(data),
            'valid_records': 0,
            'invalid_records': 0,
            'errors': []
        }
        
        if not data:
            return validation_results
        
        # Get column names from first record
        columns = list(data[0].keys()) if data else []
        
        for i, record in enumerate(data):
            is_valid = True
            record_errors = []
            
            # Basic validation based on format
            if 'employee' in format_name.lower():
                # Employee data validation
                if not record.get('Employee ID') or record.get('Employee ID') == 'UNKNOWN':
                    is_valid = False
                    record_errors.append("Missing or invalid Employee ID")
                
                if not record.get('Name') or record.get('Name') == 'Unknown':
                    is_valid = False
                    record_errors.append("Missing or invalid Name")
            
            elif 'financial' in format_name.lower():
                # Financial data validation
                amount_cols = [col for col in columns if 'amount' in col.lower()]
                for col in amount_cols:
                    try:
                        amount = float(record.get(col, 0))
                        if amount < 0:
                            record_errors.append(f"Negative amount in {col}")
                    except:
                        record_errors.append(f"Invalid amount format in {col}")
            
            if record_errors:
                is_valid = False
                validation_results['errors'].append({
                    'record_index': i,
                    'errors': record_errors
                })
            
            if is_valid:
                validation_results['valid_records'] += 1
            else:
                validation_results['invalid_records'] += 1
        
        logger.info(f"Validation complete: {validation_results['valid_records']} valid, "
                   f"{validation_results['invalid_records']} invalid records")
        return validation_results
    
    def _count_missing_values(self, df: pd.DataFrame) -> Dict[str, int]:
        """Count missing values per column."""
        missing_counts = {}
        for col in df.columns:
            missing_counts[col] = df[col].isna().sum()
        return missing_counts 