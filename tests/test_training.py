import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from PIL import Image
import torch

from models import Generator, load_generator
from train import FaceDataset, parser, train


class TrainingTests(unittest.TestCase):
    def test_finetune_resume_and_export(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root/'faces'
            data.mkdir()
            for i in range(4):
                Image.new('RGB', (178, 218), (i*60, 100, 160)).save(data/f'{i}.png')
            original = root/'original.pth'
            model = Generator(feature_maps=4)
            torch.save(model.state_dict(), original)
            digest = hashlib.sha256(original.read_bytes()).hexdigest()
            common = ['--data', str(data), '--checkpoint', str(original), '--output', str(root/'run'),
                      '--device', 'cpu', '--workers', '0', '--batch-size', '2', '--critic-width', '4',
                      '--warmup-steps', '2', '--max-batches', '2', '--r1-interval', '1']
            train(parser().parse_args(common+['--epochs', '1']))
            # Warmup must not mutate G parameters OR BatchNorm running statistics.
            warmup = load_generator(root/'run/generator_candidate.pth')
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, warmup.state_dict()[key]), key)
            train(parser().parse_args(common+['--epochs', '2', '--resume', str(root/'run/latest.pt')]))
            exported = load_generator(root/'run/generator_candidate.pth')
            self.assertFalse(torch.equal(model.main[0].weight, exported.main[0].weight))
            self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(), digest)
            state = torch.load(root/'run/latest.pt', weights_only=True)
            self.assertEqual((state['epoch'], state['step']), (2, 4))
            with torch.inference_mode():
                images = exported(torch.randn(2,100,1,1))
            self.assertEqual(tuple(images.shape), (2,3,64,64))
            self.assertTrue(torch.isfinite(images).all())
            self.assertTrue((root/'run/before.png').exists())
            self.assertTrue((root/'run/epoch-002.png').exists())
            with self.assertRaises(ValueError):
                train(parser().parse_args(common+['--epochs', '1']))

    def test_dataset_range_and_empty_error(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                FaceDataset(folder)
            Image.new('RGB', (178,218), (0,255,0)).save(Path(folder)/'image.jpg')
            for crop in (0,108):
                sample = FaceDataset(folder,crop)[0]
                self.assertEqual(tuple(sample.shape), (3,64,64))
                self.assertGreaterEqual(sample.min(), -1)
                self.assertLessEqual(sample.max(), 1)


if __name__ == '__main__':
    unittest.main()
