"""
Unit tests for FastAPI endpoints
"""

import unittest
from fastapi.testclient import TestClient
from src.api.main import app
from src.common.dependency_injection import configure_services


class TestAPI(unittest.TestCase):
    """Test cases for API endpoints"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        configure_services()
        cls.client = TestClient(app)
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('timestamp', data)
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation"""
        response = self.client.post(
            "/api/v1/data/generate_synthetic",
            params={"num_ticks": 100, "initial_price": 100.0}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['records'], 100)
        self.assertIn('data', data)
    
    def test_get_statistics(self):
        """Test statistics endpoint"""
        response = self.client.get("/api/v1/stats")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('event_subscribers', data)
        self.assertIn('timestamp', data)


if __name__ == '__main__':
    unittest.main()
