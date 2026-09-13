import io
import unittest
from unittest.mock import patch

from PIL import Image
import app as application


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.client = application.app.test_client()

    def test_page_and_assets(self):
        page = self.client.get('/')
        self.assertEqual(page.status_code, 200)
        self.assertIn(b'id="seedInput"', page.data)
        self.assertIn(b'id="historyGrid"', page.data)
        for asset in ('app.js', 'style.css'):
            with self.client.get('/static/' + asset) as response:
                self.assertEqual(response.status_code, 200)

    def test_seed_reproduces_png(self):
        first = self.client.post('/generate?seed=42')
        second = self.client.post('/generate?seed=42')
        other = self.client.post('/generate?seed=43')
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.mimetype, 'image/png')
        self.assertEqual(first.headers['X-Generation-Seed'], '42')
        self.assertIn('no-store', first.headers['Cache-Control'])
        self.assertEqual(first.data, second.data)
        self.assertNotEqual(first.data, other.data)
        with Image.open(io.BytesIO(first.data)) as image:
            self.assertEqual(image.size, (64, 64))
            self.assertEqual(image.mode, 'RGB')

    def test_random_seed_can_be_replayed(self):
        first = self.client.post('/generate')
        seed = first.headers['X-Generation-Seed']
        self.assertEqual(first.data, self.client.post('/generate?seed=' + seed).data)

    def test_seed_boundaries(self):
        for seed in ('0', '4294967295'):
            with self.subTest(seed=seed):
                self.assertEqual(self.client.post('/generate?seed=' + seed).status_code, 200)

    def test_invalid_seed_never_runs_model(self):
        with patch.object(application, 'generator') as model:
            for seed in ('', '-1', '1.5', '4294967296', 'abc', '9' * 100, '１２'):
                with self.subTest(seed=seed):
                    result = self.client.post('/generate', query_string={'seed': seed})
                    self.assertEqual(result.status_code, 400)
                    self.assertIn('error', result.json)
            model.assert_not_called()

    def test_failure_is_safe_and_lock_recovers(self):
        with patch.object(application, 'generator', side_effect=RuntimeError('private detail')):
            with self.assertLogs(application.app.logger, level='ERROR'):
                result = self.client.post('/generate?seed=1')
            self.assertEqual(result.status_code, 500)
            self.assertNotIn(b'private detail', result.data)
        self.assertEqual(self.client.post('/generate?seed=1').status_code, 200)


if __name__ == '__main__':
    unittest.main()
