"""File type detection agent for the format detection pipeline."""

import os
import pandas as pd
from typing import Literal


class FileTypeAgent:
    """Detect the file type of an input file.
    
    This agent is used as the first step in the orchestration pipeline to quickly
    identify supported file types and reject unsupported formats.
    """
    
    def __init__(self, file_path: str):
        """Initialize the agent with a file path.
        
        Args:
            file_path: Path to the file to analyze
        """
        self.file_path = file_path
        self.extension = os.path.splitext(file_path)[1].lower()
    
    def detect(self) -> Literal["Excel", "CSV", "Reject"]:
        """Detect and return the file type.
        
        Returns:
            One of "Excel", "CSV", or "Reject"
        """
        try:
            # Check Excel files
            if self.extension in [".xlsx", ".xls"]:
                # Try to read first row to validate it's a real Excel file
                pd.read_excel(self.file_path, nrows=1)
                return "Excel"
            
            # Check CSV files
            if self.extension in [".csv", ".tsv"]:
                sep = "\t" if self.extension == ".tsv" else ","
                # Try to read first row to validate it's a real CSV file
                pd.read_csv(self.file_path, sep=sep, nrows=1)
                return "CSV"
            
            # All other file types are rejected
            return "Reject"
            
        except Exception:
            # If we can't read the file, reject it
            return "Reject"