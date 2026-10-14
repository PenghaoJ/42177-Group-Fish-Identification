import base64
import io
import json
import threading
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from PIL import Image
from backend import server, adapters, config
from unittest.mock import patch

class APITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = server.ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = 'http://127.0.0.1:' + str(cls.server.server_port)
        image = io.BytesIO(); Image.new('RGB', (30, 20), 'blue').save(image, 'PNG')
        cls.encoded = base64.b64encode(image.getvalue()).decode()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join()

    def post(self, route, payload):
        request = Request(self.url + route, data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
        try:
            with urlopen(request) as response:
                return response.status, json.load(response)
        except HTTPError as error:
            return error.code, json.load(error)

    def test_both_demo_tasks_and_explanations(self):
        for task in adapters.LABELS:
            status, result = self.post('/api/predict/'+task, {'image_base64':self.encoded})
            self.assertEqual(status, 200); self.assertIsNone(result['score'])
            self.assertIsNone(result['model'])
            status, explanation = self.post('/api/explain', {'result_id':result['id']})
            self.assertEqual(status, 200); self.assertIn('DEMO ONLY', explanation['text'])

    def test_final_model_does_not_fall_back_to_demo(self):
        with patch.dict(config.FINAL_MODELS, {'species':'resnet18'}):
            status, _ = self.post('/api/predict/species', {'image_base64':self.encoded})
            self.assertEqual(status, 503)

    def test_invalid_input_and_untrusted_explanation(self):
        for payload in [[], {'image_base64':'not base64'}, {'image_base64':base64.b64encode(b'not an image').decode()}, {'image_base64':self.encoded,'model':'fake'}]:
            self.assertEqual(self.post('/api/predict/species', payload)[0],400)
        self.assertEqual(self.post('/api/explain', {'result_id':'invented','label':'safe to eat'})[0],404)
        self.assertEqual(self.post('/api/explain', {'result_id':[]})[0],400)

    def test_registered_model_and_output_validation(self):
        key = ('species','svm')
        saved = config.FINAL_MODELS['species']
        config.FINAL_MODELS['species'] = 'svm'
        try:
            adapters.register(*key, lambda image: {'label':'cod','score':.8,'model_version':'test-v1'})
            status, result = self.post('/api/predict/species', {'image_base64':self.encoded})
            self.assertEqual(status,200); self.assertEqual(result['scientific_name'],'Gadus morhua')
            adapters.register(*key, lambda image: {'label':'cod','score':float('nan'),'model_version':'test'})
            self.assertEqual(self.post('/api/predict/species', {'image_base64':self.encoded})[0],500)
        finally:
            config.FINAL_MODELS['species'] = saved
            adapters.PREDICTORS.pop(key, None)

    def test_client_cannot_choose_model_or_mode(self):
        for extra in [{'model':'resnet18'}, {'mode':'trained'}]:
            self.assertEqual(self.post('/api/predict/species', {'image_base64':self.encoded, **extra})[0],400)

    def test_static_page_and_missing_route(self):
        with urlopen(self.url) as response:
            self.assertIn(b'FishLens', response.read())
        self.assertEqual(self.post('/api/missing', {})[0],404)

if __name__ == '__main__':
    unittest.main()
