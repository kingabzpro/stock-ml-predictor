import unittest
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.app import create_app

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()
        
    def test_health_check(self):
        # Test health check endpoint
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('service', data)
        
    def test_predict_endpoint_without_model(self):
        # Test prediction without a loaded model
        response = self.client.post('/predict',
                                  json={'symbol': 'AAPL'},
                                  content_type='application/json')
        
        # Should return error if no model is loaded
        self.assertIn(response.status_code, [500, 400])
        
    def test_predict_missing_symbol(self):
        # Test prediction with missing symbol
        response = self.client.post('/predict',
                                  json={'features': {}},
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        
    def test_model_info_endpoint(self):
        # Test model info endpoint
        response = self.client.get('/model/info')
        
        # Should return error if no model is loaded
        self.assertEqual(response.status_code, 500)
        
    def test_model_list_endpoint(self):
        # Test model list endpoint
        response = self.client.get('/model/list')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('models', data)
        self.assertIsInstance(data['models'], list)
        
    def test_data_fetch_endpoint(self):
        # Test data fetch endpoint
        response = self.client.post('/data/fetch',
                                  json={
                                      'symbol': 'AAPL',
                                      'start_date': '2023-01-01',
                                      'end_date': '2023-01-31'
                                  },
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('symbol', data)
        self.assertIn('data', data)
        self.assertIn('count', data)
        
    def test_data_fetch_missing_symbol(self):
        # Test data fetch with missing symbol
        response = self.client.post('/data/fetch',
                                  json={'start_date': '2023-01-01'},
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        
    def test_batch_predict_endpoint(self):
        # Test batch prediction endpoint
        response = self.client.post('/predict/batch',
                                  json={
                                      'predictions': [
                                          {
                                              'symbol': 'AAPL',
                                              'features': {'close': 150, 'volume': 1000000}
                                          },
                                          {
                                              'symbol': 'GOOGL',
                                              'features': {'close': 100, 'volume': 2000000}
                                          }
                                      ]
                                  },
                                  content_type='application/json')
        
        # Should return error if no model is loaded
        self.assertEqual(response.status_code, 500)
        
    def test_404_error(self):
        # Test 404 error handler
        response = self.client.get('/nonexistent')
        self.assertEqual(response.status_code, 404)
        
        data = json.loads(response.data)
        self.assertIn('error', data)

if __name__ == '__main__':
    unittest.main()