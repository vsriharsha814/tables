"""File type detection agent."""

import os
import pandas as pd
from typing import Dict, Any, Literal
from agents.base_agent import BaseAgent
from agents.agent_factory import AgentFactory


class FileTypeAgent(BaseAgent):
    """Detect the file type of an input file."""
    
    def process(self, file_path: str) -> Dict[str, Any]:
        """Process file and detect its type.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Dict with 'file_type' and 'valid' keys
        """
        extension = os.path.splitext(file_path)[1].lower()
        
        try:
            # Check Excel files
            if extension in [".xlsx", ".xls"]:
                pd.read_excel(file_path, nrows=1)
                return {
                    "file_type": "Excel",
                    "valid": True,
                    "extension": extension
                }
            
            # Check CSV files
            if extension in [".csv", ".tsv"]:
                sep = "\t" if extension == ".tsv" else ","
                pd.read_csv(file_path, sep=sep, nrows=1)
                return {
                    "file_type": "CSV",
                    "valid": True,
                    "extension": extension,
                    "separator": sep
                }
            
            # Unsupported file type
            return {
                "file_type": "Unknown",
                "valid": False,
                "extension": extension,
                "reason": "Unsupported file type"
            }
            
        except Exception as e:
            return {
                "file_type": "Unknown",
                "valid": False,
                "extension": extension,
                "reason": f"Failed to read file: {str(e)}"
            }


# Register with factory
AgentFactory.register("FileTypeAgent", FileTypeAgent)
