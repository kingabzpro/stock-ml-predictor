import os
import sys
from flask import Flask, jsonify
from flask_cors import CORS

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import Config
from src.utils.logger import get_logger
from src.api.endpoints import create_endpoints

def create_app(config_path: str = 'config.yaml') -> Flask:
    # Initialize Flask app
    app = Flask(__name__)
    
    # Load configuration
    config = Config(config_path)
    api_config = config.get_api_config()
    
    # Setup logging
    logger = get_logger(__name__, config.get_logging_config())
    
    # Enable CORS if configured
    if api_config.get('cors_enabled', True):
        CORS(app)
    
    # Register endpoints
    create_endpoints(app, config)
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Endpoint not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {str(error)}")
        return jsonify({'error': 'Internal server error'}), 500
    
    @app.errorhandler(Exception)
    def handle_exception(error):
        logger.error(f"Unhandled exception: {str(error)}")
        return jsonify({'error': str(error)}), 500
    
    # Health check endpoint
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({
            'status': 'healthy',
            'service': 'Stock ML API',
            'version': '1.0.0'
        })
    
    logger.info("Flask application created successfully")
    return app

def main():
    # Load configuration
    config = Config('config.yaml')
    api_config = config.get_api_config()
    
    # Create app
    app = create_app()
    
    # Run server
    app.run(
        host=api_config.get('host', '0.0.0.0'),
        port=api_config.get('port', 5000),
        debug=api_config.get('debug', False)
    )

if __name__ == '__main__':
    main()