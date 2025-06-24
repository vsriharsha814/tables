"""Main FastAPI application with orchestrator integration."""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
from pathlib import Path
from typing import Optional
import logging
import sys
import pandas as pd
import json

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from orchestrator.core_orchestrator import CoreOrchestrator
from agents.agent_factory import AgentFactory
from api.feedback import router as feedback_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Format Detection API",
    description="Document format detection using semantic similarity and LLM fallback",
    version="2.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include feedback router
app.include_router(feedback_router)

# Global orchestrator instance
orchestrator: Optional[CoreOrchestrator] = None


@app.on_event("startup")
async def startup_event():
    """Initialize orchestrator on startup."""
    global orchestrator
    
    # Get configuration from environment or defaults
    formats_db = os.getenv("FORMATS_DB_PATH", "src/models/master_formats.xlsx")
    training_data = os.getenv("TRAINING_DATA_PATH", "src/models/document_formats.json")
    confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
    use_llm = os.getenv("USE_LLM_FALLBACK", "true").lower() == "true"
    
    logger.info(f"Initializing orchestrator...")
    logger.info(f"  Formats DB: {formats_db}")
    logger.info(f"  Training Data: {training_data}")
    logger.info(f"  Confidence threshold: {confidence_threshold}")
    logger.info(f"  LLM fallback: {use_llm}")
    
    try:
        orchestrator = CoreOrchestrator(
            formats_db_path=formats_db,
            training_data_path=training_data,
            confidence_threshold=confidence_threshold,
            use_llm_fallback=use_llm
        )
        logger.info("✓ Orchestrator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {e}")
        raise


@app.get("/")
async def root():
    """API information."""
    return {
        "name": "Format Detection API",
        "version": "2.0.0",
        "orchestrator": "LangChain-based pipeline",
        "endpoints": {
            "POST /detect": "Upload file for format detection and data extraction",
            "POST /extract": "Extract data from file with specified format",
            "GET /health": "Health check",
            "GET /formats": "List available formats",
            "GET /training-data": "Get training data for formats",
            "GET /training-data/{format_name}/schema": "Get the output schema for a specific format"
        },
        "features": {
            "format_detection": "Semantic similarity + LLM fallback",
            "data_extraction": "Structured data extraction with validation",
            "preprocessing": "Automatic header detection and data cleaning"
        }
    }


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy" if orchestrator else "unhealthy",
        "orchestrator_loaded": orchestrator is not None,
        "agents": AgentFactory.list_agents() if orchestrator else []
    }


@app.get("/formats")
async def list_formats():
    """List available formats."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        formats = []
        loaded_formats = set(orchestrator.format_detector.known_formats.keys())
        training_formats = set(orchestrator.format_detector.training_data.keys())
        
        # Read from Excel file for detection formats
        df = pd.read_excel(orchestrator.formats_db_path)
        
        for _, row in df.iterrows():
            format_label = row.get('Format Label', '')
            if not format_label:
                continue
                
            # Check if this format was successfully loaded
            is_loaded = format_label in loaded_formats
            has_training_data = format_label in training_formats
            
            # Get column count from loaded format if available
            column_count = 0
            if is_loaded:
                column_count = len(orchestrator.format_detector.known_formats[format_label]['columns'])
            
            formats.append({
                "name": format_label,
                "category": row.get('Format Type', 'Unknown'),
                "description": row.get('Description', ''),
                "column_count": column_count,
                "is_loaded": is_loaded,
                "has_training_data": has_training_data,
                "sample_layout": row.get('Sample Layout', ''),
                "format_type": row.get('Format Type', 'Unknown')
            })
        
        # Add training-only formats (formats that exist in JSON but not in Excel)
        for format_name in training_formats - loaded_formats:
            training_info = orchestrator.format_detector.training_data[format_name]
            formats.append({
                "name": format_name,
                "category": training_info.get('category', 'Unknown'),
                "description": training_info.get('description', ''),
                "column_count": len(training_info.get('semantic_concepts', [])),
                "is_loaded": False,
                "has_training_data": True,
                "format_code": training_info.get('format_code', ''),
                "confidence_threshold": training_info.get('confidence_threshold', 0.7),
                "training_only": True
            })
        
        return {
            "formats": formats, 
            "total": len(formats),
            "loaded_count": len(loaded_formats),
            "training_count": len(training_formats),
            "source_file": str(orchestrator.formats_db_path),
            "training_file": str(orchestrator.training_data_path) if orchestrator.training_data_path else None
        }
        
    except Exception as e:
        logger.error(f"Error reading formats: {e}")
        # Fallback to loaded formats only
        formats = []
        for name, info in orchestrator.format_detector.known_formats.items():
            formats.append({
                "name": name,
                "category": info.get('category', 'Unknown'),
                "description": info.get('description', ''),
                "column_count": len(info.get('columns', [])),
                "is_loaded": True,
                "has_training_data": name in orchestrator.format_detector.training_data
            })
        
        return {"formats": formats, "total": len(formats), "error": str(e)}


@app.post("/detect")
async def detect_format(
    file: UploadFile = File(...),
    header_row: Optional[int] = Form(None),
    extract_data: Optional[bool] = Form(True)
):
    """Detect format of uploaded file and optionally extract data."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Update orchestrator settings for this request
        orchestrator.enable_data_extraction = extract_data if extract_data is not None else True
        
        # Process through orchestrator
        result = orchestrator.process(tmp_path, header_row)
        
        # Add file info
        result['file_info'] = {
            'filename': file.filename,
            'size_bytes': len(content)
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Cleanup
        try:
            os.unlink(tmp_path)
        except:
            pass


@app.post("/extract")
async def extract_data(
    file: UploadFile = File(...),
    format_name: str = Form(...),
    header_row: Optional[int] = Form(None)
):
    """Extract data from file with specified format."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Validate format
    if format_name not in orchestrator.format_detector.known_formats:
        raise HTTPException(status_code=400, detail=f"Unknown format: {format_name}")
    
    # Check if data extraction is enabled
    if not orchestrator.data_extractor:
        raise HTTPException(status_code=503, detail="Data extraction is not enabled")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Create format info for extraction
        format_info = {
            'format_name': format_name,
            'confidence_score': 1.0,  # User specified format
            'detection_source': 'user_specified'
        }
        
        # Extract data
        extraction_result = orchestrator.data_extractor.process(
            file_path=tmp_path,
            format_info=format_info,
            header_row=header_row
        )
        
        # Add file info
        extraction_result['file_info'] = {
            'filename': file.filename,
            'size_bytes': len(content),
            'format_name': format_name
        }
        
        return extraction_result
        
    except Exception as e:
        logger.error(f"Extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Cleanup
        try:
            os.unlink(tmp_path)
        except:
            pass


@app.get("/training-data")
async def get_training_data(format_name: Optional[str] = None):
    """Get training data for formats."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    if format_name:
        # Get training data for specific format
        training_info = orchestrator.format_detector.get_training_data(format_name)
        if not training_info:
            raise HTTPException(status_code=404, detail=f"No training data found for format: {format_name}")
        
        return {
            "format_name": format_name,
            "training_data": training_info
        }
    else:
        # Get all training data
        all_training = orchestrator.format_detector.get_all_training_data()
        
        # Format the response
        training_formats = []
        for name, info in all_training.items():
            training_formats.append({
                "format_name": name,
                "category": info.get('category', 'Unknown'),
                "format_code": info.get('format_code', ''),
                "confidence_threshold": info.get('confidence_threshold', 0.7),
                "semantic_concepts_count": len(info.get('semantic_concepts', [])),
                "description": info.get('description', '')[:200] + "..." if len(info.get('description', '')) > 200 else info.get('description', '')
            })
        
        return {
            "training_formats": training_formats,
            "total": len(training_formats),
            "source_file": str(orchestrator.training_data_path) if orchestrator.training_data_path else None
        }


@app.get("/training-data/{format_name}/schema")
async def get_format_schema(format_name: str):
    """Get the output schema for a specific format."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    training_info = orchestrator.format_detector.get_training_data(format_name)
    if not training_info:
        raise HTTPException(status_code=404, detail=f"No training data found for format: {format_name}")
    
    schema_str = training_info.get('output_schema', '')
    if not schema_str:
        raise HTTPException(status_code=404, detail=f"No schema found for format: {format_name}")
    
    try:
        # Parse and return the schema
        schema = json.loads(schema_str)
        return {
            "format_name": format_name,
            "schema": schema
        }
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid schema format: {e}")


if __name__ == "__main__":
    import uvicorn
    
    # Run with: python main.py
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5001,
        reload=True,
        log_level="info"
    )