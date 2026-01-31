import unittest

from tests.planners.tester import PlannerTester

class TestEarliest(PlannerTester, unittest.TestCase):
    def setUp(self):
        super().setUp()

        self.single_sat_toy : bool = False
        self.multiple_sat_toy : bool = False
        self.single_sat_lakes : bool = True
        self.multiple_sat_lakes : bool = False
    
    def planner_name(self) -> str:
        return "earliest"

    def toy_planner_config(self) -> dict:
        return {
            "preplanner": {
                "@type": "earliest",
                "debug": "False",
                # "horizon": 1000,
                "period" : 500,
            }
        }
    
    def lakes_planner_config(self) -> dict:
        return {
            "preplanner": {
                "@type": "earliest",
                "debug": "False",
                # "horizon": 1000,
                "period" : 500,
            }
        }

if __name__ == '__main__':

    # run tests
    unittest.main()