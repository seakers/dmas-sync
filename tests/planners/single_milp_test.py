import unittest

from tests.planners.tester import PlannerTester

class TestSingleSatMILP(PlannerTester, unittest.TestCase):
    def setUp(self):
        super().setUp()

        self.single_sat_toy : bool = True
        self.multiple_sat_toy : bool = False
        self.single_sat_lakes : bool = True
        self.multiple_sat_lakes : bool = False
        
    def planner_name(self) -> str:
        return "single-sat-milp"

    def toy_planner_config(self) -> dict:
        return {
            "preplanner": {
                "@type": "milp",
                "model": "earliest",
                "licensePath": "./gurobi.lic",
                # "horizon": 500,
                "period" : 250,
                "maxTasks": 100,
                "debug" : "False"
            }
        }
    
    def lakes_planner_config(self) -> dict:
        return {
            "preplanner": {
                "@type": "milp",
                "model": "earliest",
                "licensePath": "./gurobi.lic",
                # "horizon": 500,
                "period" : 250,
                "maxTasks": 100,
                "debug" : "False"
            }
        }
    
if __name__ == '__main__':
    # run tests
    unittest.main()
    