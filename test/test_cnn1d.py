import unittest
import torch

from models.cnn1d import get_mask

class MaskingCNNTest(unittest.TestCase):
    def test_mask(self):
        # data is assumed to be of size (batch, 1, seq_len)
        # in this case we generate seq_len == 5
        input_data = torch.tensor([[[1, 2, 3, 0, 0]], [[3, 0, 1, 0, 0]], [[0, 0, 1, 2, 0]]])
        print(input_data.shape)
        self.assertTrue(input_data.shape[0] > 1)
        self.assertTrue(input_data.shape[1] == 1)
        self.assertTrue(input_data.shape[2] == 5)

        result = get_mask(input_data)
        self.assertEquals(result.shape, input_data.shape)

        result = result.squeeze()
        self.assertEqual(result[0, :].tolist(), [1, 1, 1, 0, 0])
        self.assertEqual(result[1, :].tolist(), [1, 1, 1, 0, 0])
        self.assertEqual(result[2, :].tolist(), [1, 1, 1, 1, 0])