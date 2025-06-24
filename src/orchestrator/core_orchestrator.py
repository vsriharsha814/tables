"""Core orchestrator using LangChain for the processing pipeline."""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, cast

from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field
from pydantic.v1 import SecretStr

from agents.agent_factory import AgentFactory
from agents.format_detector_agent import FormatDetectorAgent
from agents.file_type_agent import FileTypeAgent
from agents.data_extraction_agent import DataExtractionAgent

logger = logging.getLogger(__name__)


class FormatDetectionResult(BaseModel):
    """Result of format detection."""
    format_name: str = Field(description="Detected format name")
    confidence_score: float = Field(description="Confidence score (0-1)")
    source: str = Field(description="Detection source: 'semantic' or 'llm'")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CoreOrchestrator:
    """Orchestrates the document processing pipeline."""
    
    def __init__(
        self, 
        formats_db_path: str = "src/models/master_formats.xlsx",
        training_data_path: Optional[str] = None,
        confidence_threshold: float = 0.7,
        use_llm_fallback: bool = True,
        enable_data_extraction: bool = True
    ):
        """Initialize orchestrator.
        
        Args:
            formats_db_path: Path to master formats Excel file
            training_data_path: Optional path to JSON training data file
            confidence_threshold: Minimum confidence for semantic matching
            use_llm_fallback: Whether to use LLM for low-confidence matches
            enable_data_extraction: Whether to extract data after format detection
        """
        self.formats_db_path = Path(formats_db_path)
        self.training_data_path = Path(training_data_path) if training_data_path else None
        self.confidence_threshold = confidence_threshold
        self.use_llm_fallback = use_llm_fallback
        self.enable_data_extraction = enable_data_extraction
        
        # Initialize agents
        self.file_type_agent = AgentFactory.get_agent("FileTypeAgent")
        self.format_detector = cast(FormatDetectorAgent, AgentFactory.get_agent("FormatDetectorAgent"))
        
        # Initialize data extraction agent if enabled
        self.data_extractor = None
        if enable_data_extraction:
            self.data_extractor = cast(DataExtractionAgent, AgentFactory.get_agent("DataExtractionAgent"))
        
        # Load formats into detector
        self.format_detector.load_formats(self.formats_db_path)
        
        # Load training data if provided
        if self.training_data_path and self.training_data_path.exists():
            self.format_detector.load_training_data(self.training_data_path)
            logger.info(f"Loaded training data from: {self.training_data_path}")
        
        # Initialize LLM if needed
        self.llm = None
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if use_llm_fallback and api_key:
            self.llm = ChatAnthropic(
                model_name="claude-3-haiku-20240307",
                api_key=SecretStr(api_key),
                timeout=30,
                temperature=0.1
            )
        
        logger.info(f"Orchestrator initialized with {len(self.format_detector.known_formats)} formats")
        logger.info(f"Training data loaded: {len(self.format_detector.training_data)} formats")
        logger.info(f"Data extraction: {'enabled' if enable_data_extraction else 'disabled'}")
    
    def process(self, file_path: str, header_row: Optional[int] = None) -> Dict[str, Any]:
        """Process a file through the detection pipeline.
        
        Args:
            file_path: Path to the file to process
            header_row: Optional header row index
            
        Returns:
            Detection results as dictionary
        """
        logger.info(f"Starting pipeline for: {file_path}")
        
        # Step 1: File type validation
        file_type_result = self.file_type_agent.process(file_path)
        
        if not file_type_result['valid']:
            return {
                'success': False,
                'error': file_type_result.get('reason', 'Invalid file type'),
                'file_type': file_type_result.get('file_type', 'Unknown')
            }
        
        logger.info(f"File type validated: {file_type_result['file_type']}")
        
        # Step 2: Format detection
        detection_result = self.format_detector.process(file_path, header_row or 0)
        
        format_name = detection_result['format_name']
        confidence = detection_result['confidence_score']
        
        logger.info(f"Semantic detection: {format_name} (confidence: {confidence:.3f})")
        
        # Step 3: Confidence-based branching
        if confidence >= self.confidence_threshold:
            # High confidence - use semantic match
            response = self._create_response(
                format_name=format_name,
                confidence=confidence,
                source='semantic',
                detection_result=detection_result,
                file_type=file_type_result['file_type']
            )
        else:
            # Low confidence - try LLM fallback if available
            if self.use_llm_fallback and self.llm:
                logger.info("Low confidence - attempting LLM fallback")
                
                llm_result = self._llm_fallback(
                    file_columns=detection_result['file_columns'],
                    top_matches=detection_result['top_matches']
                )
                
                if llm_result['success']:
                    response = self._create_response(
                        format_name=llm_result['format_name'],
                        confidence=llm_result['confidence'],
                        source='llm',
                        detection_result=detection_result,
                        file_type=file_type_result['file_type'],
                        llm_reasoning=llm_result.get('reasoning')
                    )
                else:
                    # No good match found
                    response = self._create_response(
                        format_name=format_name if confidence > 0.3 else None,
                        confidence=confidence,
                        source='semantic',
                        detection_result=detection_result,
                        file_type=file_type_result['file_type'],
                        low_confidence=True
                    )
            else:
                # No good match found
                response = self._create_response(
                    format_name=format_name if confidence > 0.3 else None,
                    confidence=confidence,
                    source='semantic',
                    detection_result=detection_result,
                    file_type=file_type_result['file_type'],
                    low_confidence=True
                )
        
        # Step 4: Data extraction (if enabled and format detected)
        if (self.enable_data_extraction and 
            self.data_extractor and 
            response['success'] and 
            response['result']['format_name']):
            
            logger.info("Starting data extraction...")
            extraction_result = self.data_extractor.process(
                file_path=file_path,
                format_info=response['result'],
                header_row=header_row
            )
            
            if extraction_result['success']:
                response['extraction'] = extraction_result
                logger.info("Data extraction completed successfully")
            else:
                response['extraction'] = {
                    'success': False,
                    'error': extraction_result.get('error', 'Data extraction failed')
                }
                logger.warning(f"Data extraction failed: {extraction_result.get('error')}")
        
        return response
    
    def _llm_fallback(self, file_columns: list, top_matches: list) -> Dict[str, Any]:
        """Use LLM to determine format when semantic matching has low confidence."""
        try:
            # Prepare context
            known_formats = "\n".join([
                f"- {fmt}: {info['description']}"
                for fmt, info in self.format_detector.known_formats.items()
            ])
            
            # Create prompt
            prompt = f"""You are a document format expert. Analyze these file columns and determine which format they best match.

File columns:
{', '.join(file_columns)}

Top semantic matches (with scores):
{json.dumps(top_matches[:3], indent=2)}

Known formats:
{known_formats}

Based on the column names and their semantic meaning, which format does this file most likely represent?
Respond with a JSON object containing:
- format_name: the exact format name from the list
- confidence: your confidence score (0.0 to 1.0)
- reasoning: brief explanation of your choice
"""
            
            messages = [
                SystemMessage(content="You are a precise document format classifier."),
                HumanMessage(content=prompt)
            ]
            
            if self.llm:
                response = self.llm.invoke(messages)
                
                # Parse response
                result = json.loads(str(response.content))
                
                # Validate format name
                if result['format_name'] in self.format_detector.known_formats:
                    return {
                        'success': True,
                        'format_name': result['format_name'],
                        'confidence': float(result['confidence']),
                        'reasoning': result.get('reasoning', '')
                    }
            
        except Exception as e:
            logger.error(f"LLM fallback failed: {e}")
        
        return {'success': False}
    
    def _create_response(
        self, 
        format_name: Optional[str],
        confidence: float,
        source: str,
        detection_result: Dict[str, Any],
        file_type: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Create standardized response."""
        response = {
            'success': True,
            'result': {
                'format_name': format_name,
                'confidence_score': round(confidence, 3),
                'confidence_level': self._get_confidence_level(confidence),
                'detection_source': source,
                'file_type': file_type
            },
            'metadata': {
                'top_matches': detection_result.get('top_matches', []),
                'total_columns': len(detection_result.get('file_columns', [])),
                'threshold_used': self.confidence_threshold
            }
        }
        
        # Add any additional kwargs
        for key, value in kwargs.items():
            if value is not None:
                response['metadata'][key] = value
        
        return response
    
    def _get_confidence_level(self, score: float) -> str:
        """Convert numeric score to confidence level."""
        if score >= 0.8:
            return 'HIGH'
        elif score >= 0.6:
            return 'MEDIUM'
        elif score >= 0.3:
            return 'LOW'
        else:
            return 'VERY_LOW'
