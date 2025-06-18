from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import tempfile
from pathlib import Path
import logging
from datetime import datetime

# Import your detector (assuming it's in the same directory)
from agents.format_detector_agent import SimpleFormatDetector

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global detector instance (initialized once to avoid reloading model)
detector = None
FORMATS_DB_PATH = None

# Allowed file extensions
ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_detector(formats_db_path):
    """Initialize the detector with formats database."""
    global detector, FORMATS_DB_PATH
    try:
        detector = SimpleFormatDetector()
        detector.load_formats(Path(formats_db_path))
        FORMATS_DB_PATH = formats_db_path
        logger.info(f"Detector initialized with formats from {formats_db_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        return False

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy' if detector else 'not_initialized',
        'detector_loaded': detector is not None,
        'formats_db': FORMATS_DB_PATH,
        'total_formats': len(detector.known_formats) if detector else 0,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/formats', methods=['GET'])
def list_formats():
    """List all available formats."""
    if not detector:
        return jsonify({'error': 'Detector not initialized'}), 500
    
    formats_info = []
    for format_name, format_info in detector.known_formats.items():
        formats_info.append({
            'name': format_name,
            'category': format_info['category'],
            'description': format_info['description'],
            'column_count': len(format_info['columns']),
            'sample_columns': format_info['columns'][:5]  # First 5 columns as sample
        })
    
    return jsonify({
        'total_formats': len(formats_info),
        'formats': formats_info
    })

@app.route('/detect', methods=['POST'])
def detect_format():
    """Main endpoint for format detection.
    
    Expected form data:
    - file: The Excel/CSV file to analyze
    - header_row (optional): Row number containing headers (0-based)
    """
    if not detector:
        return jsonify({
            'error': 'Detector not initialized. Please check server configuration.'
        }), 500
    
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded. Use form-data with key "file"'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            'error': f'File type not allowed. Supported: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400
    
    # Get optional header row parameter
    header_row = request.form.get('header_row')
    if header_row:
        try:
            header_row = int(header_row)
        except ValueError:
            return jsonify({'error': 'header_row must be an integer'}), 400
    else:
        header_row = None
    
    # Save uploaded file temporarily
    filename = secure_filename(file.filename)
    temp_path = None
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name
        
        logger.info(f"Processing file: {filename} (temp: {temp_path})")
        
        # Run detection
        result = detector.detect_format(Path(temp_path), header_row)
        
        # Clean up the result for JSON serialization
        clean_result = clean_detection_result(result)
        
        # Add metadata
        clean_result['metadata'] = {
            'original_filename': filename,
            'processing_timestamp': datetime.now().isoformat(),
            'header_row_specified': header_row,
            'file_size_bytes': os.path.getsize(temp_path)
        }
        
        # Add summary for easier consumption
        if clean_result.get('best_match'):
            best = clean_result['best_match']
            clean_result['summary'] = {
                'best_format': best['format'],
                'confidence_score': best['scores']['composite_score'],
                'confidence_level': get_confidence_level(best['scores']['composite_score']),
                'sheet': best['sheet'],
                'category': best['category'],
                'matches_found': best['scores']['total_matches'],
                'total_expected_columns': best['scores']['total_format_cols'],
                'exact_matches': best['scores']['exact_matches'],
                'semantic_matches': best['scores']['semantic_matches']
            }
        else:
            clean_result['summary'] = {
                'best_format': None,
                'confidence_score': 0,
                'confidence_level': 'NONE',
                'message': 'No format matches found'
            }
        
        logger.info(f"Detection completed for {filename}")
        return jsonify(clean_result)
        
    except Exception as e:
        logger.error(f"Error processing file {filename}: {e}")
        return jsonify({
            'error': f'Failed to process file: {str(e)}'
        }), 500
        
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                logger.debug(f"Cleaned up temporary file: {temp_path}")
            except OSError:
                logger.warning(f"Failed to delete temporary file: {temp_path}")

def get_confidence_level(score):
    """Convert numeric score to confidence level."""
    if score >= 0.8:
        return 'HIGH'
    elif score >= 0.6:
        return 'MEDIUM'
    elif score >= 0.3:
        return 'LOW'
    else:
        return 'VERY_LOW'

def clean_detection_result(result):
    """Clean the detection result for JSON serialization."""
    def clean_object(obj):
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: clean_object(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_object(item) for item in obj]
        else:
            return obj
    
    return clean_object(result)

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found. Available: /health, /formats, /detect'}), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Flask Format Detection API Server')
    parser.add_argument(
        '--formats',
        type=str,
        required=True,
        help='Path to Excel file with format definitions'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Host to bind to (default: 127.0.0.1)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port to bind to (default: 5000)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run in debug mode'
    )
    
    args = parser.parse_args()
    
    # Initialize detector
    print("Initializing format detector...")
    if not init_detector(args.formats):
        print("❌ Failed to initialize detector. Exiting.")
        exit(1)
    
    print(f"✅ Detector initialized with {len(detector.known_formats)} formats")
    print(f"🚀 Starting Flask API server on {args.host}:{args.port}")
    print(f"📊 Formats database: {args.formats}")
    print(f"\n📡 API Endpoints:")
    print(f"   GET  http://{args.host}:{args.port}/health")
    print(f"   GET  http://{args.host}:{args.port}/formats") 
    print(f"   POST http://{args.host}:{args.port}/detect")
    print(f"\n💡 Use Postman to test the /detect endpoint with form-data file upload")
    
    app.run(host=args.host, port=args.port, debug=args.debug)
