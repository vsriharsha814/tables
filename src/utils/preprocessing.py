"""Data preprocessing utilities."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import re
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Utilities for preprocessing data before format detection."""
    
    @staticmethod
    def clean_column_names(columns: List[str]) -> List[str]:
        """Clean and standardize column names."""
        cleaned = []
        for col in columns:
            # Convert to string
            col_str = str(col).strip()
            
            # Remove extra whitespace
            col_str = ' '.join(col_str.split())
            
            # Remove special characters but keep spaces and basic punctuation
            col_str = re.sub(r'[^\w\s\-_.]', '', col_str)
            
            # Handle empty or unnamed columns
            if not col_str or col_str.lower() in ['unnamed', 'nan', 'none', '']:
                col_str = f'Column_{len(cleaned) + 1}'
            
            cleaned.append(col_str)
        
        return cleaned
    
    @staticmethod
    def detect_data_types(df: pd.DataFrame) -> Dict[str, str]:
        """Detect data types for each column."""
        type_info = {}
        
        for col in df.columns:
            # Skip if column is empty
            if df[col].isna().all():
                type_info[col] = 'empty'
                continue
            
            # Sample non-null values
            sample = df[col].dropna().head(100)
            
            # Try to infer type
            if pd.api.types.is_numeric_dtype(sample):
                if pd.api.types.is_integer_dtype(sample):
                    type_info[col] = 'integer'
                else:
                    type_info[col] = 'float'
            elif pd.api.types.is_datetime64_any_dtype(sample):
                type_info[col] = 'datetime'
            elif pd.api.types.is_bool_dtype(sample):
                type_info[col] = 'boolean'
            else:
                # Check if it might be dates in string format
                try:
                    pd.to_datetime(sample, errors='coerce')
                    if sample.notna().sum() > len(sample) * 0.5:
                        type_info[col] = 'date_string'
                    else:
                        type_info[col] = 'string'
                except:
                    type_info[col] = 'string'
        
        return type_info
    
    @staticmethod
    def find_header_row(df: pd.DataFrame, max_rows: int = 10) -> int:
        """Find the most likely header row in a dataframe."""
        scores = []
        
        for i in range(min(max_rows, len(df))):
            row = df.iloc[i]
            score = 0
            
            # Count non-null values
            non_null = row.notna().sum()
            score += non_null * 10
            
            # Check if values are string-like (headers usually are)
            string_count = sum(1 for val in row if isinstance(val, str) and val.strip())
            score += string_count * 5
            
            # Check for numeric values (headers usually aren't numeric)
            numeric_count = sum(1 for val in row if isinstance(val, (int, float)) and not pd.isna(val))
            score -= numeric_count * 3
            
            # Check for diversity (headers usually have unique values)
            unique_ratio = len(row.unique()) / len(row) if len(row) > 0 else 0
            score += unique_ratio * 20
            
            scores.append((i, score))
        
        # Return row with highest score
        best_row = max(scores, key=lambda x: x[1])[0]
        logger.info(f"Detected header row at index {best_row}")
        
        return best_row
    
    @staticmethod
    def extract_sample_data(df: pd.DataFrame, n_rows: int = 5) -> List[Dict[str, Any]]:
        """Extract sample data rows for analysis."""
        sample_rows = []
        
        # Get up to n_rows of non-empty rows
        for _, row in df.iterrows():
            if row.notna().sum() > len(row) * 0.3:  # At least 30% non-null
                sample_rows.append(row.to_dict())
            
            if len(sample_rows) >= n_rows:
                break
        
        return sample_rows

