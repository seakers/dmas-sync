
import unittest

from dmas.utils.tools import *

class TestTools(unittest.TestCase):
    def test_banner(self):
        # just ensure it runs without error
        print_scenario_banner('Test Banner')

    # TODO test interval data class object
    
    # TODO test time-indexed data class object

if __name__ == '__main__':
    
    # run tests
    unittest.main()
