import os
from pathlib import Path

class WebConfig:
    """Configuration for the web server and simulation integration"""
    
    def __init__(self):
        self.HOST = os.getenv('SHORTAGESIM_HOST', '0.0.0.0')
        self.PORT = int(os.getenv('SHORTAGESIM_PORT', 5000))
        self.DEBUG = os.getenv('SHORTAGESIM_DEBUG', 'True').lower() == 'true'
        self.SECRET_KEY = os.getenv('SHORTAGESIM_SECRET_KEY', 'dev-key-change-in-production')
        
        # File storage
        self.UPLOAD_FOLDER = Path('uploads')
        self.RESULTS_FOLDER = Path('results')
        self.LOGS_FOLDER = Path('web_logs')
        
        # Create directories
        self.UPLOAD_FOLDER.mkdir(exist_ok=True)
        self.RESULTS_FOLDER.mkdir(exist_ok=True)
        self.LOGS_FOLDER.mkdir(exist_ok=True)
        
        # Session management
        self.MAX_CONCURRENT_SIMULATIONS = int(os.getenv('MAX_SIMULATIONS', 10))
        self.SESSION_TIMEOUT = int(os.getenv('SESSION_TIMEOUT', 3600))  # 1 hour
        
        # LLM settings
        self.DEFAULT_API_KEY = os.getenv('OPENAI_API_KEY')
        self.ALLOW_API_KEY_OVERRIDE = os.getenv('ALLOW_API_KEY_OVERRIDE', 'True').lower() == 'true'