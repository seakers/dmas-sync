import unittest

from tests.planners.tester import PlannerTester

class TestDynamicProgramming(PlannerTester, unittest.TestCase):
    def setUp(self):
        super().setUp()

        self.single_sat_toy : bool = True
        self.multiple_sat_toy : bool = False
        self.single_sat_lakes : bool = True
        self.multiple_sat_lakes : bool = False
    
    def planner_name(self) -> str:
        return "dynamic-programming"

    def toy_planner_config(self) -> dict:
        return {
            "preplanner": {
                "@type": "dynamic",
                "debug": "False",
                "model" : "earliest",
                "sharing": "periodic",
                # "horizon": 250,
                "period" : 250,
            }
        }
    
    def lakes_planner_config(self) -> dict:
        return {
            "preplanner": {
                "@type": "dynamic",
                "debug": "False",
                "model" : "earliest",
                "sharing": "periodic",
                # "horizon": 250,
                "period" : 250,
            }
        }

if __name__ == '__main__':

    # run tests
    unittest.main()