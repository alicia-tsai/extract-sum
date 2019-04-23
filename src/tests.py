import unittest
import data
import utils

class TestExtractSum(unittest.TestCase):

    def test_get_outlook_data(self):
        title, ref, text = data.get_outlook_data()
        num_title, num_ref, num_text = len(title), len(ref), len(text)
        self.assertEqual(True, num_title > 0)
        self.assertEqual(True, num_title == num_ref == num_text)

    def 


if __name__ == '__main__':
    unittest.main()
