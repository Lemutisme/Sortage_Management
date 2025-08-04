# File: startup.py
"""
Application startup and initialization
"""

import sys
from pathlib import Path

def setup_python_path():
    """Add src directory to Python path"""
    src_path = Path(__file__).parent / 'src'
    if src_path.exists():
        sys.path.insert(0, str(src_path))

def create_directory_structure():
    """Create necessary directories"""
    directories = [
        'templates',
        'static/js',
        'static/css',
        'uploads',
        'results',
        'web_logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def validate_dependencies():
    """Validate that all required modules are available"""
    required_modules = [
        'flask',
        'flask_socketio',
        'flask_cors',
        'eventlet'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"Missing required modules: {missing_modules}")
        print("Install them with: pip install -r requirements_web.txt")
        return False
    
    return True

def initialize_app():
    """Initialize the application"""
    setup_python_path()
    create_directory_structure()
    
    if not validate_dependencies():
        sys.exit(1)
    
    print("✅ ShortageSim web application initialized successfully")
    return True

if __name__ == '__main__':
    initialize_app()