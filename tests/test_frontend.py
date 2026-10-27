"""Static frontend delivery checks; no image upload or inference requests."""
import re
import unittest
from fastapi.testclient import TestClient
from backend import app, frontend_dist


@unittest.skipUnless((frontend_dist / 'index.html').is_file(), 'Build frontend with npm run build first')
class FrontendDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_home_and_javascript_are_served(self):
        home = self.client.get('/')
        self.assertEqual(home.status_code, 200)
        self.assertIn('LungLens', home.text)
        script = re.search(r'src="(/assets/[^\"]+\.js)"', home.text)
        self.assertIsNotNone(script)
        self.assertEqual(self.client.get(script.group(1)).status_code, 200)
        self.assertEqual(self.client.get('/favicon.svg').status_code, 200)

    def test_api_and_static_files_remain_separate(self):
        self.assertEqual(self.client.get('/health').json()['project'], 'LungLens')
        self.assertEqual(self.client.get('/.env').status_code, 404)
        self.assertEqual(self.client.get('/backend.py').status_code, 404)
        schema = self.client.get('/openapi.json').json()
        self.assertIn('/predict', schema['paths'])


if __name__ == '__main__':
    unittest.main()
