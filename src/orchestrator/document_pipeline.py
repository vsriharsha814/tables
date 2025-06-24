"""Simple document processing pipeline for testing."""

import logging
from pathlib import Path
from typing import Dict, Any, cast

from agents.agent_factory import AgentFactory
from agents.format_detector_agent import FormatDetectorAgent

logger = logging.getLogger(__name__)


class SimpleDocumentPipeline:
    """Simplified pipeline for testing individual components."""
    
    def __init__(self, formats_db_path: str):
        """Initialize the simple pipeline."""
        self.formats_db_path = Path(formats_db_path)
        
        # Initialize agents
        self.file_type_agent = AgentFactory.get_agent("FileTypeAgent")
        self.format_detector: FormatDetectorAgent = cast(FormatDetectorAgent, AgentFactory.get_agent("FormatDetectorAgent"))
        
        # Load formats
        self.format_detector.load_formats(self.formats_db_path)
        
        logger.info("Simple pipeline initialized")
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """Process a single file through the pipeline."""
        logger.info(f"Processing file: {file_path}")
        
        # Step 1: Check file type
        file_type_result = self.file_type_agent.process(file_path)
        logger.info(f"File type: {file_type_result}")
        
        if not file_type_result['valid']:
            return {
                'success': False,
                'error': 'Invalid file type',
                'details': file_type_result
            }
        
        # Step 2: Detect format
        format_result = self.format_detector.process(file_path)
        logger.info(f"Format detection: {format_result['format_name']} "
                   f"(confidence: {format_result['confidence_score']:.3f})")
        
        return {
            'success': True,
            'file_type': file_type_result['file_type'],
            'format': format_result['format_name'],
            'confidence': format_result['confidence_score'],
            'top_matches': format_result.get('top_matches', [])
        }
