"""
Additional API routes for advanced functionality
"""

from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename
import pandas as pd
import json

from datetime import datetime

from session_manager import SessionManager
from config_web import WebConfig

api_bp = Blueprint('api', __name__, url_prefix='/api')

@api_bp.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })

@api_bp.route('/sessions/<session_id>/status')
def get_session_status(session_id):
    """Get detailed session status"""
    session = SessionManager.get_session(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    
    return jsonify({
        'session_id': session_id,
        'status': session.status,
        'created_at': session.created_at.isoformat(),
        'last_activity': session.last_activity.isoformat(),
        'has_results': session.results is not None
    })

@api_bp.route('/ground-truth/upload', methods=['POST'])
def upload_ground_truth():
    """Upload ground truth data for validation"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        filepath = WebConfig.UPLOAD_FOLDER / filename
        file.save(filepath)
        
        # Validate CSV format
        try:
            df = pd.read_csv(filepath)
            required_columns = ['gt_id', 'n_manufacturers', 'periods', 'disruption_magnitude']
            
            if not all(col in df.columns for col in required_columns):
                return jsonify({'error': f'Missing required columns: {required_columns}'}), 400
            
            return jsonify({
                'message': 'File uploaded successfully',
                'filename': filename,
                'rows': len(df),
                'columns': list(df.columns)
            })
            
        except Exception as e:
            return jsonify({'error': f'Invalid CSV file: {str(e)}'}), 400
    
    return jsonify({'error': 'Invalid file format. Only CSV files are supported.'}), 400

@api_bp.route('/ground-truth/run', methods=['POST'])
def run_ground_truth_experiment():
    """Run simulation against ground truth data"""
    data = request.get_json()
    filename = data.get('filename')
    
    if not filename:
        return jsonify({'error': 'Filename required'}), 400
    
    filepath = WebConfig.UPLOAD_FOLDER / filename
    if not filepath.exists():
        return jsonify({'error': 'File not found'}), 404
    
    # This would integrate with your existing ground truth experiment code
    # from main.py run_gt_experiments function
    
    return jsonify({'message': 'Ground truth experiment started', 'filename': filename})

@api_bp.route('/results/<session_id>/download/<format>')
def download_results(session_id, format):
    """Download simulation results in various formats"""
    session = SessionManager.get_session(session_id)
    if not session or not session.results:
        return jsonify({'error': 'No results available'}), 404
    
    if format not in ['json', 'csv', 'xlsx']:
        return jsonify({'error': 'Unsupported format'}), 400
    
    # Generate file based on format
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if format == 'json':
        filename = f'shortagesim_results_{session_id}_{timestamp}.json'
        filepath = WebConfig.RESULTS_FOLDER / filename
        
        with open(filepath, 'w') as f:
            json.dump(session.results, f, indent=2)
        
        return send_file(filepath, as_attachment=True)
    
    elif format == 'csv':
        filename = f'shortagesim_results_{session_id}_{timestamp}.csv'
        filepath = WebConfig.RESULTS_FOLDER / filename
        
        # Convert results to DataFrame and save as CSV
        # This would need to be implemented based on your results structure
        
        return send_file(filepath, as_attachment=True)
    
    return jsonify({'error': 'Format not implemented yet'}), 501