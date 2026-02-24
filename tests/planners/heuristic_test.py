import unittest

from tests.planners.tester import PlannerTester

class TestHeuristic(PlannerTester, unittest.TestCase):
    def setUp(self):
        super().setUp()

        self.single_sat_toy : bool = False
        self.multiple_sat_toy : bool = False
        self.single_sat_lakes : bool = False
        self.multiple_sat_lakes : bool = False

        ## toy cases
        self.toy_1 = False  # single sat    default mission     single target, no events
        self.toy_2 = True  # single sat    no default mission  one event
        self.toy_3 = False  # two sats      no default mission  one event
    
    def planner_name(self) -> str:
        return "heuristic"

    def toy_planner_config(self) -> dict:
        return {
            # "preplanner": {
            #     "@type": "heuristic",
            #     "debug": "False",
            #     # "horizon": 1000,
            #     "period" : 200,
            # },
            "replanner": {
                "@type": "heuristic",
                "debug": "False"
            }
        }
    
    def lakes_planner_config(self) -> dict:
        return {
            "preplanner": {
                "@type": "heuristic",
                "debug": "False",
                # "horizon": 1000,
                "period" : 100,
            }
        }

if __name__ == '__main__':

    # run tests
    unittest.main()