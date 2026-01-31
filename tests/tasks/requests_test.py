import unittest

from execsatm.tasks import DefaultMissionTask, GenericObservationTask
from execsatm.utils import Interval

from dmas.models.science.requests import TaskRequest
from dmas.utils.tools import print_scenario_banner

class TestTaskRequests(unittest.TestCase):
    def setUp(self):
        super().setUp()

        self.default_task = DefaultMissionTask(
            parameter="test_parameter",
            location=(45.0, 90.0, 1, 2),
            mission_duration=3600.0,            
            id="test_task_001"
        )
        self.default_request = TaskRequest(
            task=self.default_task,
            requester="test_requester",
            mission_name="test_mission",
            t_req=0.0
        )
                

    def test_task_request_initializer(self):        
        self.assertIsInstance(self.default_request, TaskRequest)
        self.assertEqual(self.default_request.task, self.default_task)
        self.assertEqual(self.default_request.requester, "test_requester")
        self.assertEqual(self.default_request.mission_name, "test_mission")
        self.assertEqual(self.default_request.t_req, 0.0)
    
    def test_representation(self):
        self.assertEqual(repr(self.default_request), f"TaskRequest_{self.default_request.id.split('-')[0]}")

    def test_to_dict(self):
        request_dict = self.default_request.to_dict()
        self.assertIsInstance(request_dict, dict)
        self.assertIsInstance(request_dict['task'], dict)
        self.assertEqual(request_dict['requester'], "test_requester")
        self.assertEqual(request_dict['task']['task_type'], GenericObservationTask.DEFAULT)
        self.assertEqual(request_dict['task']['parameter'], "test_parameter")
        self.assertEqual(request_dict['task']['location'], [(45.0, 90.0, 1, 2)])
        self.assertEqual(request_dict['task']['availability'], Interval(0, 3600.0).to_dict())
        self.assertEqual(request_dict['task']['priority'], 1.0)
        self.assertEqual(request_dict['task']['objective'], None)
        self.assertEqual(request_dict['task']['id'], "test_task_001")
        self.assertEqual(request_dict['mission_name'], "test_mission")
        self.assertEqual(request_dict['t_req'], 0.0)
        self.assertEqual(request_dict['id'], self.default_request.id)

    def test_from_dict(self):
        request_dict = self.default_request.to_dict()
        new_request = TaskRequest.from_dict(request_dict)
        self.assertEqual(new_request.id, self.default_request.id)
        self.assertEqual(new_request.requester, self.default_request.requester)
        self.assertEqual(new_request.mission_name, self.default_request.mission_name)
        self.assertEqual(new_request.t_req, self.default_request.t_req)
        self.assertEqual(new_request.task.id, self.default_request.task.id)


    # def test_event_request_comparison(self):
    #     event_objective = EventDrivenObjective(
    #         event_type="earthquake",
    #         parameter="test_parameter",
    #         weight=1.0,
    #         requirements=[
    #             # Define any specific requirements for the objective here
    #             PointTargetSpatialRequirement((45.0, 90.0, 0, 1)),
    #             AvailabilityRequirement(0, 3600.0),
    #         ],
    #     )
    #     event_1 = GeophysicalEvent(
    #         event_type="earthquake",
    #         severity=5.0,
    #         location=[(45.0, 90.0, 0, 1)],
    #         t_detect=1000.0,
    #         d_exp=3600.0
    #     )
    #     event_2 = GeophysicalEvent(
    #         event_type="earthquake",
    #         severity=1.0,
    #         location=[(90.0, 45.0, 0, 1)],
    #         t_detect=500.0,
    #         d_exp=1000.0
    #     )
    #     event_none_task = EventObservationTask(
    #         parameter="test_parameter",
    #         priority=1.0,
    #         objective=event_objective,
    #     )
    #     event_1_opp_1 = EventObservationTask(
    #         parameter="test_parameter",
    #         priority=1.0,
    #         objective=event_objective,
    #         event=event_1
    #     )
    #     event_2_task = EventObservationTask(
    #         parameter="test_parameter",
    #         priority=1.0,
    #         objective=event_objective,
    #         event=event_2
    #     )
    #     event_request = TaskRequest(
    #         task=event_none_task,
    #         requester="event_requester",
    #         mission_name="event_mission",
    #         t_req=0.0
    #     )

    #     self.assertRaises(ValueError, event_request.__eq__, other_req="invalid_request")
    #     self.assertRaises(ValueError, event_request.__eq__, other_req=self.default_request)
    #     self.assertTrue(event_request == event_request)


if __name__ == '__main__':
    # terminal welcome message
    print_scenario_banner('Task Request Definitions Test')
    
    # run tests
    unittest.main()