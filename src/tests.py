import unittest
import data
import utils

class TestExtractSum(unittest.TestCase):

    def test_get_outlook_data(self):
        outlook_data = data.get_outlook_data()
        self.assertEqual(True, (len(outlook_data) > 0))

if __name__ == '__main__':
    unittest.main()
