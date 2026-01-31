import unittest

from tests.planners.tester import PlannerTester

class TestHeuristic(PlannerTester, unittest.TestCase):
    def setUp(self):
        super().setUp()

        self.single_sat_toy : bool = True
        self.multiple_sat_toy : bool = False
        self.single_sat_lakes : bool = False
        self.multiple_sat_lakes : bool = False
    
    def planner_name(self) -> str:
        return "announcer"

    def toy_planner_config(self) -> dict:
        return {
            "preplanner": {
                "@type": "eventAnnouncer",
                "debug": "False",
                "eventsPath" : "./tests/planners/resources/events/toy_events.csv"

            }
        }

    def lakes_planner_config(self) -> dict:
        return {
            "preplanner": {
                "@type": "eventAnnouncer",
                "debug": "False",
                "eventsPath" : "./tests/planners/resources/events/lake_events_seed-1000.csv"
            }
        }

if __name__ == '__main__':

    # run tests
    unittest.main()