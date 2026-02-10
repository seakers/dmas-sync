
import os
from typing import Dict
import unittest

from dmas.models.trackers import LatestObservationTracker
from dmas.utils.orbitdata import OrbitData
from dmas.utils.tools import print_scenario_banner


class TestLatestObservationTracker(unittest.TestCase):
    def setUp(self):
        self.sat_name = 'sat1'

    def test_from_orbitdata(self):
        # define orbitdata path
        orbitdata_path = os.path.join('tests', 'agents', 'orbit_data', 'obs_tracker')

        # load orbitdata
        orbitdata : Dict[str, OrbitData] = OrbitData.from_directory(orbitdata_path, simulation_duration=2.0/24.0)

        # ensure the right data was loaded
        self.assertEqual(len(orbitdata), 1, f"Expected 1 agent in orbitdata, got {len(orbitdata)}")
        self.assertIn(self.sat_name, orbitdata, f"Expected agent name '{self.sat_name}' in orbitdata, got {list(orbitdata.keys())}")

        # iterate trough loaded orbitdata
        for agent_name, agent_orbitdata in orbitdata.items():
            # construct tracker from orbitdata
            tracker = LatestObservationTracker.from_orbitdata(agent_orbitdata, agent_name)

            # check if tracker is instance of LatestObservationTracker
            self.assertIsInstance(tracker, LatestObservationTracker)

            # check if tracker has correct information
            self.assertEqual(len(tracker.key_to_k), 9)
            self.assertEqual(len(tracker.k_to_key), 9)
            self.assertEqual(tracker.N, 9)

            
if __name__ == '__main__':
    # print banner
    print_scenario_banner("`LatestObservationTracker` Unit Tests")
    
    # run tests
    unittest.main()