import unittest

from line_data_generation.generate_training_sample import add_rectangle, generate_training_sample


class Test(unittest.TestCase):

    def test_add_rectangle(self):
        rectangle = add_rectangle()
        self.assertIsNotNone(rectangle)

    def test_generate_training_sample(self):
        training_sample = generate_training_sample()
        self.assertIsNotNone(training_sample)


if __name__ == '__main__':
    unittest.main()
