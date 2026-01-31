import unittest

from tests.planners.tester import PlannerTester

class TestNadir(PlannerTester, unittest.TestCase):
    def setUp(self):
        super().setUp()

        self.single_sat_toy : bool = True
        self.multiple_sat_toy : bool = False
        self.single_sat_lakes : bool = False
        self.multiple_sat_lakes : bool = False
    
    def planner_name(self) -> str:
        return "nadir"

    def toy_planner_config(self) -> dict:
        return {
            "preplanner": {
                "@type": "nadir",
                "debug": "False",
                "horizon": 500,
                "period" : 500,
            }
        }

    def lakes_planner_config(self) -> dict:
        return {
            "preplanner": {
                "@type": "nadir",
                "debug": "False",
                "horizon": 500,
                "period" : 500,
            }
        }

if __name__ == '__main__':

    # run tests
    unittest.main()