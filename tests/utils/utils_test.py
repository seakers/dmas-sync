
import unittest

from dmas.core.utils import *

class TestUtils(unittest.TestCase):
    def test_banner(self):
        # just ensure it runs without error
        print_scenario_banner('Test Banner')

if __name__ == '__main__':
    
    # run tests
    unittest.main()
