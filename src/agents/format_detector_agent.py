"""Format detection agent using semantic similarity."""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from sentence_transformers import SentenceTransformer
import logging

from agents.base_agent import BaseAgent
from agents.agent_factory import AgentFactory
from models.manager import ModelManager

logger = logging.getLogger(__name__)


class FormatDetectorAgent(BaseAgent):
    """Detect document format using semantic similarity."""
    
    def __init__(self):
        """Initialize with model from ModelManager."""
        self.model_manager = ModelManager()
        self.model = self.model_manager.get_sentence_transformer()
        self.known_formats = {}
        self.training_data = {}  # Additional training data from JSON
    
    def load_formats(self, excel_path: Path) -> None:
        """Load known formats from Excel file."""
        df = pd.read_excel(excel_path)
        
        for index, row in df.iterrows():
            format_label = row.get('Format Label', '')
            if not format_label:
                continue
            
            # Parse sample columns
            sample_cols = []
            layout = row.get('Sample Layout', '')
            
            if isinstance(layout, str) and layout.strip():
                try:
                    # Guard against non-string-literal inputs
                    if not (layout.startswith('[') and layout.endswith(']')):
                        raise ValueError("Layout is not a valid list representation.")
                    
                    import ast
                    parsed = ast.literal_eval(layout)
                    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                        sample_cols = list(parsed[0].keys())
                    else:
                        logger.warning(
                            f"Could not parse a valid column list from layout for format '{format_label}' in row {int(str(index)) + 2}. Layout: '{layout}'"
                        )

                except (ValueError, SyntaxError, TypeError) as e:
                    logger.error(
                        f"Failed to parse 'Sample Layout' for format '{format_label}' in row {int(str(index)) + 2}. Error: {e}. Layout: '{layout}'"
                    )
            
            if sample_cols:
                sample_cols = [str(col).strip() for col in sample_cols]
                
                self.known_formats[format_label] = {
                    'columns': sample_cols,
                    'description': row.get('Description', ''),
                    'category': row.get('Format Type', 'Unknown'),
                    'embeddings': None  # Will be computed on demand
                }
            else:
                logger.warning(f"No sample columns loaded for format '{format_label}' in row {int(str(index)) + 2}. This format will not be used for matching.")

        if not self.known_formats:
            logger.critical("CRITICAL: No formats were loaded from the master sheet. Format detection will fail.")
        else:
            logger.info(f"Successfully loaded {len(self.known_formats)} formats from Excel.")
    
    def load_training_data(self, json_path: Path) -> None:
        """Load additional training data from JSON file."""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            formats = data.get('document_formats', [])
            
            for format_data in formats:
                format_name = format_data.get('format_name', '')
                if not format_name:
                    continue
                
                # Extract semantic concepts from output_schema
                semantic_concepts = self._extract_semantic_concepts_from_schema(
                    format_data.get('output_schema', '')
                )
                
                if semantic_concepts:
                    self.training_data[format_name] = {
                        'semantic_concepts': semantic_concepts,
                        'description': format_data.get('description', ''),
                        'category': format_data.get('category', 'Unknown'),
                        'format_code': format_data.get('format_code', ''),
                        'confidence_threshold': format_data.get('confidence_threshold', 0.7),
                        'output_schema': format_data.get('output_schema', '')
                    }
                    logger.info(f"Loaded training data for format: {format_name}")
                else:
                    logger.warning(f"No semantic concepts found for training format '{format_name}'.")
            
            logger.info(f"Successfully loaded training data for {len(self.training_data)} formats from JSON.")
                
        except Exception as e:
            logger.error(f"Failed to load training data from JSON file: {e}")
            raise
    
    def _extract_semantic_concepts_from_schema(self, schema_str: str) -> List[str]:
        """Extract semantic concept names from the output schema."""
        try:
            # Parse the schema string as JSON
            schema = json.loads(schema_str)
            semantic_concepts = schema.get('semantic_concepts', {})
            
            # Return the concept names as column names
            return list(semantic_concepts.keys())
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse output schema: {e}")
            return []
    
    def get_training_data(self, format_name: str) -> Dict[str, Any]:
        """Get training data for a specific format."""
        return self.training_data.get(format_name, {})
    
    def get_all_training_data(self) -> Dict[str, Any]:
        """Get all training data."""
        return self.training_data
    
    def process(self, file_path: str, header_row: int = 0) -> Dict[str, Any]:
        """Process file and detect its format.
        
        Args:
            file_path: Path to the file
            header_row: Optional header row index (default: 0)
            
        Returns:
            Dict with format detection results
        """
        if len(self.known_formats) == 0:
            logger.error("No known formats loaded, cannot perform detection.")
            return {
                'format_name': None,
                'confidence_score': 0.0,
                'top_matches': [],
                'file_columns': [],
                'status': 'error',
                'error': 'No known formats are loaded into the system.'
            }

        file_path_obj = Path(file_path)
        
        # Read file based on type
        if file_path_obj.suffix.lower() in {'.xlsx', '.xls'}:
            df = pd.read_excel(file_path, header=header_row)
        else:
            df = pd.read_csv(file_path, header=header_row)
        
        # Get file columns
        file_columns = [str(col).strip() for col in df.columns]
        
        # Compute embeddings for file columns
        file_embeddings = self.model.encode(file_columns)
        
        # Compare against known formats
        best_match = None
        best_score = 0.0
        all_scores = []
        
        for format_name, format_info in self.known_formats.items():
            # Get or compute format embeddings
            if format_info['embeddings'] is None:
                format_info['embeddings'] = self.model.encode(format_info['columns'])
            
            # Compute similarity
            score = self._compute_similarity(
                file_embeddings, 
                format_info['embeddings'],
                file_columns,
                format_info['columns']
            )
            
            all_scores.append({
                'format': format_name,
                'score': score,
                'category': format_info['category']
            })
            
            if score > best_score:
                best_score = score
                best_match = format_name
        
        # Sort by score
        all_scores.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'format_name': best_match,
            'confidence_score': best_score,
            'top_matches': all_scores[:5],
            'file_columns': file_columns,
            'status': 'success'
        }
    
    def _compute_similarity(self, file_emb, format_emb, file_cols, format_cols) -> float:
        """Compute similarity between file and format columns."""
        if len(format_cols) == 0:
            return 0.0
        
        # Find best match for each format column
        matches = 0
        total_similarity = 0.0
        
        for i, format_col in enumerate(format_cols):
            best_sim = 0.0
            for j, file_col in enumerate(file_cols):
                # Compute cosine similarity
                # Add a check for zero norm vectors
                format_norm = np.linalg.norm(format_emb[i])
                file_norm = np.linalg.norm(file_emb[j])
                if format_norm == 0 or file_norm == 0:
                    sim = 0.0
                else:
                    sim = float(np.dot(format_emb[i], file_emb[j]) / (format_norm * file_norm))
                best_sim = max(best_sim, sim)
            
            if best_sim > 0.5:  # Threshold for considering a match
                matches += 1
                total_similarity += best_sim
        
        # Combined score
        coverage = matches / len(format_cols)
        avg_similarity = total_similarity / matches if matches > 0 else 0
        
        return 0.7 * coverage + 0.3 * avg_similarity


# Register with factory
AgentFactory.register("FormatDetectorAgent", FormatDetectorAgent)

