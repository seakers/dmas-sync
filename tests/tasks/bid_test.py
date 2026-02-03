import unittest

import numpy as np

from dmas.utils.tools import print_scenario_banner
from dmas.models.planning.decentralized.consensus.bids import Bid, BidComparisonResults
from execsatm.tasks import DefaultMissionTask


class TestBids(unittest.TestCase):
    """
    Test suite for bids used in consensus planners.
    """
    SENDER = 'sender'
    RECEIVER = 'receiver'
    THIRD_PARTY = 'third_party'
    FOURTH_PARTY = 'fourth_party'

    def setUp(self):
        # Common task used across many tests
        self.task_a = DefaultMissionTask(parameter='test_param',
                                       location=(0.0, 0.0, 0, 0),
                                       mission_duration=100.0,
                                       priority=1)
        
        self.task_b = DefaultMissionTask(parameter='test_param',
                                       location=(0.0, 0.0, 1, 2),
                                       mission_duration=50.0,
                                       priority=0.5)
        
    # ----------------------------------------------------------------------
    # Helper to build bids quickly
    # ----------------------------------------------------------------------
    def make_bid(self,
                 task=None,
                 bidder="agentA",
                 n_obs=0,
                 owner_bid=None,
                 winning_bid=0.0,
                 winner=Bid.NONE,
                 t_img=0.0,
                 t_bid=0.0,
                 t_stamps=None,
                 performed=False,
                 main_measurement="VNIR",
                ) -> Bid:
        # set default values if missing
        task = task or self.task_a
        owner_bid = winning_bid if owner_bid is None else owner_bid

        # return bid
        return Bid(task,
                   bidder,
                   n_obs,
                   owner_bid,
                   winning_bid,
                   winner,
                   t_img,
                   t_bid,
                   t_stamps,
                   main_measurement,
                   performed
                   )

    def make_bid_dict(self,
                 task=None,
                 bidder="agentA",
                 n_obs=0,
                 owner_bid=None,
                 winning_bid=0.0,
                 winner=Bid.NONE,
                 t_img=0.0,
                 t_bid=0.0,
                 t_stamps=None,
                 performed=False,
                 main_measurement="VNIR",
                ) -> dict:
        bid = self.make_bid(task,
                            bidder,
                            n_obs,
                            owner_bid,
                            winning_bid,
                            winner,
                            t_img,
                            t_bid,
                            t_stamps,
                            main_measurement,
                            performed)
        return bid.to_dict()

    # ----------------------------------------------------------------------
    # Basic construction / serialization
    # ----------------------------------------------------------------------
    def test_to_dict_and_from_dict_roundtrip(self):
        bid = self.make_bid(
            bidder="agentA",
            winner="agentA",
            winning_bid=10.0,
            t_img=10.0,
            n_obs=1,
            performed=True,
        )

        bid_dict = bid.to_dict()
        reconstructed = Bid.from_dict(bid_dict)

        self.assertEqual(bid.task.id, reconstructed.task.id)
        self.assertEqual(bid.main_measurement, reconstructed.main_measurement)
        self.assertEqual(bid.owner, reconstructed.owner)
        self.assertAlmostEqual(bid.owner_bid, reconstructed.owner_bid)
        self.assertEqual(bid.winner, reconstructed.winner)
        self.assertAlmostEqual(bid.winning_bid, reconstructed.winning_bid)
        self.assertAlmostEqual(bid.t_img, reconstructed.t_img)
        self.assertEqual(bid.n_obs, reconstructed.n_obs)
        self.assertAlmostEqual(bid.t_stamps, reconstructed.t_stamps)
        self.assertTrue(bid.t_stamps is not reconstructed.t_stamps)
        self.assertEqual(bid.performed, reconstructed.performed)

    def test_copy_creates_independent_object(self):
        bid = self.make_bid(
            bidder="agentA",
            winner="agentA",
            winning_bid=10.0,
            t_img=10.0,
            n_obs=1,
            performed=True,
        )
        bid_copy = bid.copy()

        self.assertIsNot(bid, bid_copy)
        self.assertEqual(bid, bid_copy)
        # mutate copy to ensure original is unchanged
        # bid_copy.set(main_instrument='VNIR', bid_value=20.0, t_img=15.0, n_img=2, t_update=6.0)
        bid_copy.set('VNIR', 20.0, 15.0, 1.0)
        self.assertNotEqual(bid.winning_bid, bid_copy.winning_bid)

    # ----------------------------------------------------------------------
    # Comparison operators (value + tie-breaker)
    # ----------------------------------------------------------------------
    def test_comparison_operators_basic(self):
        # b_high has higher bid_value than b_low
        b_low = self.make_bid(
            bidder="agentA",
            winner="agentA",
            winning_bid=5.0,
            owner_bid=5.0,
        )
        b_high = self.make_bid(
            bidder="agentA",
            winner="agentA",
            winning_bid=10.0,
            owner_bid=10.0
        )

        self.assertTrue(b_high > b_low)
        self.assertTrue(b_low < b_high)
        self.assertTrue(b_high >= b_low)
        self.assertTrue(b_low <= b_high)
        self.assertFalse(b_high == b_low)
        self.assertTrue(b_high != b_low)

    def test_comparison_tie_breaker_on_bidder_name(self):
        # Same winning_bid and bid_value; tie broken on bidder name (alphabetical)
        bA = self.make_bid(
            bidder="agentA",
            winner="agentA",
            winning_bid=10.0,
            owner_bid=10.0,
        )
        bB = self.make_bid(
            bidder="agentB",
            winner="agentB",
            winning_bid=10.0,
            owner_bid=10.0,
        )

        # __tie_breaker returns the lexicographically smallest bidder as "greater"
        # because it uses min(...) and __gt__/__lt__ interpret it accordingly.
        self.assertTrue(bA > bB)
        self.assertTrue(bB < bA)

    def test_comparison_input_checks(self):
        b1 = self.make_bid(
            bidder="agentA",
            winner="agentA",
            winning_bid=10.0,
            owner_bid=10.0,
            task=self.task_a,
        )
        b2 = self.make_bid(
            bidder="agentB",
            winner="agentB",
            winning_bid=5.0,
            owner_bid=5.0,
            task=self.task_b,  # different task
        )
        b3 = self.make_bid(
            bidder="agentA",
            winner="agentA",
            winning_bid=10.0,
            owner_bid=10.0,
            task=self.task_a,
            n_obs=10 # different n_obs
        )

        # compare to b2 (different task) should raise AssertionError
        with self.assertRaises(AssertionError): b1 > "b2"
        with self.assertRaises(AssertionError): b1 > b2
        with self.assertRaises(AssertionError): b1 < b2
        with self.assertRaises(AssertionError): b1 >= b2
        with self.assertRaises(AssertionError): b1 <= b2

        # compare to b2 `dict` (different task) should raise AssertionError
        b2 = b2.to_dict()
        with self.assertRaises(AssertionError): b1 > "b2"
        with self.assertRaises(AssertionError): b1 > b2
        with self.assertRaises(AssertionError): b1 < b2
        with self.assertRaises(AssertionError): b1 >= b2
        with self.assertRaises(AssertionError): b1 <= b2

        # compare to b3 (different n_img) should raise AssertionError
        with self.assertRaises(AssertionError): "b1" > b3
        with self.assertRaises(AssertionError): b1 > b3
        with self.assertRaises(AssertionError): b1 < b3
        with self.assertRaises(AssertionError): b1 >= b3
        with self.assertRaises(AssertionError): b1 <= b3

        # compare to b3 `dict` (different n_img) should raise AssertionError
        b3 = b3.to_dict()
        with self.assertRaises(TypeError): "b1" > b3
        with self.assertRaises(AssertionError): b1 > b3
        with self.assertRaises(AssertionError): b1 < b3
        with self.assertRaises(AssertionError): b1 >= b3
        with self.assertRaises(AssertionError): b1 <= b3

    def test_has_different_winner_values(self):
        # define bids
        b1 = self.make_bid(
            bidder="agentA",
            winner="agentA",
            winning_bid=10.0,
            owner_bid=10.0,
            task=self.task_a,
            t_img=10
        )
        b2 = self.make_bid(
            bidder="agentA",
            winner="agentB", # different winner
            winning_bid=10.0, 
            owner_bid=10.0,
            task=self.task_a,
            t_img=10
        )
        b3 = self.make_bid(
            bidder="agentA",
            winner="agentA",
            winning_bid=11.0, # different winning bid
            owner_bid=10.0,
            task=self.task_a,
            t_img=10
        )
        b4 = self.make_bid(
            bidder="agentA",
            winner="agentA",
            winning_bid=10.0,
            owner_bid=10.0,
            task=self.task_a,
            t_img=12          # different imaging time
        )
        b5 = self.make_bid(
            bidder="agentB", 
            winner="agentA",
            winning_bid=10.0,
            owner_bid=5.0,
            task=self.task_a,
            t_img=10
        )

        self.assertFalse(b1.has_different_winner_values(b1) )
        self.assertTrue(b1.has_different_winner_values(b2))
        self.assertTrue(b1.has_different_winner_values(b3))
        self.assertTrue(b1.has_different_winner_values(b4))
        self.assertFalse(b1.has_different_winner_values(b5))

    def test_has_different_winner_values_input_checks(self):
        # define bids
        b1 = self.make_bid(
            bidder="agentA",
            winner="agentA",
            winning_bid=10.0,
            owner_bid=10.0,
            task=self.task_a,
        )
        b2 = self.make_bid(
            bidder="agentB",
            winner="agentB",
            winning_bid=5.0,
            owner_bid=5.0,
            task=self.task_b,  # different task
        )
        b3 = self.make_bid(
            bidder="agentA",
            winner="agentA",
            winning_bid=10.0,
            owner_bid=10.0,
            task=self.task_a,
            n_obs=10    # different n_obs
        )
        b4 = self.make_bid(
            bidder="agentB",
            winner="agentB",
            winning_bid=11.0,
            owner_bid=11.0,
            task=self.task_a,
        )
        # other is the wrong type
        with self.assertRaises(AssertionError): b1.has_different_winner_values('b2')

        # other is dict but...
        # does not have the appropriate fields
        with self.assertRaises(AssertionError): b1.has_different_winner_values(b2.to_dict())

        # is for a different task
        with self.assertRaises(AssertionError): b1.has_different_winner_values(b3.to_dict())

        # is for different number of observations

        # other is a Bid but...
        # is for a different task
        with self.assertRaises(AssertionError): b1.has_different_winner_values(b2.to_dict())

        # is for different number of observations
        with self.assertRaises(AssertionError): b1.has_different_winner_values(b3.to_dict())

    # ----------------------------------------------------------------------
    # compare() behavior in representative scenarios
    # ----------------------------------------------------------------------
    def test_compare_sender_thinks_itself_winner(self):
        """
        CASE 1: Sender thinks itself winner
        """
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.SENDER,
            winning_bid=10.0,
            t_img=5.0
        )
        # CASE 1.1: Receiver thinks itself is the winner 
        # CASE 1.1.1: Sender has higher bid and an earlier observation time
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.RECEIVER,
            winning_bid=1.0,
            t_img=10.0,
        )   

        # Expect: UPDATE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.UPDATE)
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.UPDATE)        

        # CASE 1.1.2: Sender has higher bid and a later observation time
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.RECEIVER,
            winning_bid=1.0,
            t_img=1.0,
        )   

        # Expect: UPDATE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.UPDATE)
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.UPDATE)

        # CASE 1.1.2: Sender has lower bid and an earlier observation time
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.RECEIVER,
            winning_bid=15.0,
            t_img=10.0,
        ) 

        # Expect: UPDATE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.UPDATE)
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.UPDATE)

        # CASE 1.1.4: Sender has lower bid and a later observation time
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.RECEIVER,
            winning_bid=15.0,
            t_img=1.0,
        ) 
        
        # Expect: LEAVE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.LEAVE)  
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.LEAVE)  

        # CASE 1.2: Receiver thinks other is the winner 
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.SENDER,
            winning_bid=1.0
        )   

        # Expect: UPDATE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.UPDATE)
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.UPDATE)

        # CASE 1.3: Receiver thinks a 3rd party is the winner 
        # CASE 1.3.1: Sender has a more recent communication with 3rd party, higher bid, and earlier observation time
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.SENDER,
            t_stamps={self.SENDER : 1.0, self.THIRD_PARTY : 10.0},
            winning_bid=20.0,
            t_img=5.0,
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.THIRD_PARTY,
            t_stamps={self.THIRD_PARTY : 5.0},
            winning_bid=15.0,
            t_img=10.0
        )
        # Expect: UPDATE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.UPDATE)
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.UPDATE)

        # CASE 1.3.2: Sender has a more recent communication with 3rd party, higher bid, and later observation time
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.SENDER,
            t_stamps={self.SENDER : 1.0, self.THIRD_PARTY : 10.0},
            winning_bid=20.0,
            t_img=15.0,
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.THIRD_PARTY,
            t_stamps={self.THIRD_PARTY : 5.0},
            winning_bid=15.0,
            t_img=10.0
        )
        # Expect: UPDATE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.UPDATE)        
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.UPDATE)

        # CASE 1.3.3: Sender has a more recent communication with 3rd party, lower bid, and earlier observation time
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.SENDER,
            t_stamps={self.SENDER : 1.0, self.THIRD_PARTY : 10.0},
            winning_bid=10.0,
            t_img=5.0,
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.THIRD_PARTY,
            t_stamps={self.THIRD_PARTY : 5.0},
            winning_bid=15.0,
            t_img=10.0
        )
        # Expect: UPDATE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.UPDATE) 
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.UPDATE)
        
        # CASE 1.3.4: Sender has a more recent communication with 3rd party, lower bid, and later observation time
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.SENDER,
            t_stamps={self.SENDER : 1.0, self.THIRD_PARTY : 10.0},
            winning_bid=10.0,
            t_img=15.0,
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.THIRD_PARTY,
            t_stamps={self.THIRD_PARTY : 5.0},
            winning_bid=15.0,
            t_img=10.0
        )
        # Expect: UPDATE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.UPDATE) 
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.UPDATE)

        # CASE 1.3.5: Sender has an older communication with 3rd party, higher bid, and earlier observation time
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.SENDER,
            t_stamps={self.SENDER : 1.0, self.THIRD_PARTY : 1.0},
            winning_bid=20.0,
            t_img=5.0,
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.THIRD_PARTY,
            t_stamps={self.THIRD_PARTY : 5.0},
            winning_bid=15.0,
            t_img=10.0
        )
        # Expect: UPDATE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.UPDATE) 
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.UPDATE)

        # CASE 1.3.6: Sender has an older communication with 3rd party, higher bid, and later observation time
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.SENDER,
            t_stamps={self.SENDER : 1.0, self.THIRD_PARTY : 1.0},
            winning_bid=20.0,
            t_img=15.0,
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.THIRD_PARTY,
            t_stamps={self.THIRD_PARTY : 5.0},
            winning_bid=15.0,
            t_img=10.0
        )
        # Expect: UPDATE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.UPDATE) 
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.UPDATE)

        # CASE 1.3.7: Sender has an older communication with 3rd party, lower bid, and earlier observation time
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.SENDER,
            t_stamps={self.SENDER : 1.0, self.THIRD_PARTY : 1.0},
            winning_bid=10.0,
            t_img=5.0,
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.THIRD_PARTY,
            t_stamps={self.THIRD_PARTY : 5.0},
            winning_bid=15.0,
            t_img=10.0
        )
        # Expect: UPDATE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.UPDATE) 
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.UPDATE)

        # CASE 1.3.8: Sender has an older communication with 3rd party, lower bid, and later observation time
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.SENDER,
            t_stamps={self.SENDER : 1.0, self.THIRD_PARTY : 1.0},
            winning_bid=10.0,
            t_img=15.0,
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.THIRD_PARTY,
            t_stamps={self.THIRD_PARTY : 5.0},
            winning_bid=15.0,
            t_img=10.0
        )
        # Expect: LEAVE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.LEAVE) 
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.LEAVE)

        # CASE 1.4: Receiver thinks no one is the winner 
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=Bid.NONE,
            winning_bid=1.0
        )   

        # Expect: UPDATE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.UPDATE)
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.UPDATE)

    def test_compare_sender_thinks_receiver_winner(self):
        """
        CASE 2: Sender thinks the receiving agent is the winner
        """
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.RECEIVER,
            winning_bid=10.0
        )
        # CASE 2.1: Receiver thinks itself winner 
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.RECEIVER,
            winning_bid=10.0,
        )   

        # Expect: LEAVE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.LEAVE)
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.LEAVE)

        # CASE 2.2: Receiver thinks sender is the winner 
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.SENDER,
            winning_bid=10.0,
        )   

        # Expect: RESET
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.RESET)
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.RESET)

        # CASE 2.3: Receiver thinks a third party is the winner 
        # CASE 2.3.1: Sender has a more recent communication with 3rd party
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.RECEIVER,
            t_stamps={self.SENDER : 1.0, self.THIRD_PARTY : 10.0},
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.THIRD_PARTY,
            t_stamps={self.THIRD_PARTY : 5.0},
        )
        
        # Expect: RESET
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.RESET)
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.RESET)
        
        # CASE 2.3.2: Sender has an updated communication with 3rd party
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.RECEIVER,
            t_stamps={self.SENDER : 1.0, self.THIRD_PARTY : 1.0},
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.THIRD_PARTY,
            t_stamps={self.THIRD_PARTY : 5.0},
        )
        
        # Expect: LEAVE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.LEAVE)
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.LEAVE)

        # CASE 2.4: Receiver has no winner
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.RECEIVER,
            winning_bid=10.0
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=Bid.NONE,
            winning_bid=10.0,
        )  

        # Expect: LEAVE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.LEAVE)
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.LEAVE)

    def test_compare_sender_thinks_third_party_winner(self):
        """
        CASE 3: Sender thinks a 3rd party agent is the winner
        """        
        # CASE 3.1: Receiver thinks itself winner
        # CASE 3.1.1: Sender has a more updated communication with 3rd party, a higher bid, and an earlier observation time
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.THIRD_PARTY,
            t_stamps={self.THIRD_PARTY : 10.0},
            winning_bid=15.0,
            t_img=5.0,
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.RECEIVER,
            t_stamps={self.THIRD_PARTY : 5.0},
            winning_bid=10.0,
            t_img=10.0,
        )  

        # Expect: UPDATE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.UPDATE)
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.UPDATE)

        # CASE 3.1.2: Sender has a more updated communication with 3rd party, a higher bid, and a later observation time
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.THIRD_PARTY,
            t_stamps={self.THIRD_PARTY : 10.0},
            winning_bid=15.0,
            t_img=15.0,
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.RECEIVER,
            t_stamps={self.THIRD_PARTY : 5.0},
            winning_bid=10.0,
            t_img=10.0,
        )  

        # Expect: UPDATE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.UPDATE)
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.UPDATE)

        # CASE 3.1.3: Sender has a more updated communication with 3rd party, a lower bid, and an earlier observation time
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.THIRD_PARTY,
            t_stamps={self.THIRD_PARTY : 10.0},
            winning_bid=5.0,
            t_img=5.0,
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.RECEIVER,
            t_stamps={self.THIRD_PARTY : 5.0},
            winning_bid=10.0,
            t_img=10.0,
        )  

        # Expect: UPDATE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.UPDATE)
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.UPDATE)

        # CASE 3.1.4: Sender has a more updated communication with 3rd party, a lower bid, and a later observation time
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.THIRD_PARTY,
            t_stamps={self.THIRD_PARTY : 10.0},
            winning_bid=5.0,
            t_img=15.0,
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.RECEIVER,
            t_stamps={self.THIRD_PARTY : 5.0},
            winning_bid=10.0,
            t_img=10.0,
        )  

        # Expect: LEAVE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.LEAVE)
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.LEAVE)

        # CASE 3.1.5: Sender has a more oudated communication with 3rd party, a higher bid, and an earlier observation time
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.THIRD_PARTY,
            t_stamps={self.THIRD_PARTY : 1.0},
            winning_bid=15.0,
            t_img=5.0,
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.RECEIVER,
            t_stamps={self.THIRD_PARTY : 5.0},
            winning_bid=10.0,
            t_img=10.0,
        )  

        # Expect: LEAVE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.LEAVE)
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.LEAVE)

        # CASE 3.1.6: Sender has a more oudated communication with 3rd party, a higher bid, and a later observation time
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.THIRD_PARTY,
            t_stamps={self.THIRD_PARTY : 1.0},
            winning_bid=15.0,
            t_img=15.0,
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.RECEIVER,
            t_stamps={self.THIRD_PARTY : 5.0},
            winning_bid=10.0,
            t_img=10.0,
        )  

        # Expect: LEAVE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.LEAVE)
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.LEAVE)

        # CASE 3.1.7: Sender has a more oudated communication with 3rd party, a lower bid, and an earlier observation time
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.THIRD_PARTY,
            t_stamps={self.THIRD_PARTY : 1.0},
            winning_bid=5.0,
            t_img=5.0,
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.RECEIVER,
            t_stamps={self.THIRD_PARTY : 5.0},
            winning_bid=10.0,
            t_img=10.0,
        )  

        # Expect: LEAVE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.LEAVE)
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.LEAVE)

        # CASE 3.1.8: Sender has a more oudated communication with 3rd party, a lower bid, and a later observation time
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.THIRD_PARTY,
            t_stamps={self.THIRD_PARTY : 1.0},
            winning_bid=5.0,
            t_img=15.0,
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.RECEIVER,
            t_stamps={self.THIRD_PARTY : 5.0},
            winning_bid=10.0,
            t_img=10.0,
        )  

        # Expect: LEAVE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.LEAVE)    
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.LEAVE)    

        # CASE 3.2: Receiver thinks sender is the winner
        # CASE 3.2.1: Sender has a more recent communication with 3rd party
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.THIRD_PARTY,
            t_stamps={self.THIRD_PARTY : 10.0},
            winning_bid=5.0,
            t_img=15.0,
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.SENDER,
            t_stamps={self.THIRD_PARTY : 5.0},
            winning_bid=10.0,
            t_img=10.0,
        ) 

        # Expect: UPDATE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.UPDATE) 
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.UPDATE) 

        # CASE 3.2.2: Sender has a more outdated communication with 3rd party
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.THIRD_PARTY,
            t_stamps={self.THIRD_PARTY : 1.0},
            winning_bid=5.0,
            t_img=15.0,
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.SENDER,
            t_stamps={self.THIRD_PARTY : 5.0},
            winning_bid=10.0,
            t_img=10.0,
        ) 

        # Expect: RESET
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.RESET) 
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.RESET) 

        # CASE 3.3: Receiver thinks the same third party is the winner 
        # CASE 3.3.1: Sender has a more recent communication with 3rd party
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.THIRD_PARTY,
            t_stamps={self.THIRD_PARTY : 10.0},
            winning_bid=5.0,
            t_img=15.0,
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.THIRD_PARTY,
            t_stamps={self.THIRD_PARTY : 5.0},
            winning_bid=10.0,
            t_img=10.0,
        ) 

        # Expect: UPDATE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.UPDATE) 
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.UPDATE) 

        # CASE 3.3.2: Sender has a more oudated communication with 3rd party
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.THIRD_PARTY,
            t_stamps={self.THIRD_PARTY : 1.0},
            winning_bid=5.0,
            t_img=15.0,
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.THIRD_PARTY,
            t_stamps={self.THIRD_PARTY : 5.0},
            winning_bid=10.0,
            t_img=10.0,
        ) 

        # Expect: LEAVE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.LEAVE) 
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.LEAVE)

        # CASE 3.4: Receiver thinks a different third party is the winner 
        # CASE 3.4.1:  Sender has a more recent communication with 3rd party, a more recent communication with 4th party, a higher bid, and an earlier observation time
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.THIRD_PARTY,
            t_stamps={self.THIRD_PARTY : 10.0, self.FOURTH_PARTY : 10.0},
            winning_bid=15.0,
            t_img=5.0,
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.FOURTH_PARTY,
            t_stamps={self.THIRD_PARTY : 5.0, self.FOURTH_PARTY : 5.0},
            winning_bid=10.0,
            t_img=10.0,
        ) 

        # Expect: UPDATE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.UPDATE)
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.UPDATE)

        # CASE 3.4.2:  Sender has a more recent communication with 3rd party, a more recent communication with 4th party, a higher bid, and a later observation time
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.THIRD_PARTY,
            t_stamps={self.THIRD_PARTY : 10.0, self.FOURTH_PARTY : 10.0},
            winning_bid=15.0,
            t_img=15.0,
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.FOURTH_PARTY,
            t_stamps={self.THIRD_PARTY : 5.0, self.FOURTH_PARTY : 5.0},
            winning_bid=10.0,
            t_img=10.0,
        ) 

        # Expect: UPDATE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.UPDATE)
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.UPDATE)

        # CASE 3.4.3:  Sender has a more recent communication with 3rd party, a more recent communication with 4th party, a lower bid, and an earlier observation time
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.THIRD_PARTY,
            t_stamps={self.THIRD_PARTY : 10.0, self.FOURTH_PARTY : 10.0},
            winning_bid=5.0,
            t_img=5.0,
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.FOURTH_PARTY,
            t_stamps={self.THIRD_PARTY : 5.0, self.FOURTH_PARTY : 5.0},
            winning_bid=10.0,
            t_img=10.0,
        ) 

        # Expect: UPDATE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.UPDATE)
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.UPDATE)

        # CASE 3.4.4:  Sender has a more recent communication with 3rd party, a more recent communication with 4th party, a lower bid, and a later observation time
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.THIRD_PARTY,
            t_stamps={self.THIRD_PARTY : 10.0, self.FOURTH_PARTY : 10.0},
            winning_bid=5.0,
            t_img=15.0,
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.FOURTH_PARTY,
            t_stamps={self.THIRD_PARTY : 5.0, self.FOURTH_PARTY : 5.0},
            winning_bid=10.0,
            t_img=10.0,
        ) 
        
        # Expect: UPDATE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.UPDATE)
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.UPDATE)

        # CASE 3.4.5:  Sender has a more recent communication with 3rd party, a more oudated communication with 4th party, a higher bid, and an earlier observation time
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.THIRD_PARTY,
            t_stamps={self.THIRD_PARTY : 10.0, self.FOURTH_PARTY : 1.0},
            winning_bid=15.0,
            t_img=5.0,
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.FOURTH_PARTY,
            t_stamps={self.THIRD_PARTY : 5.0, self.FOURTH_PARTY : 5.0},
            winning_bid=10.0,
            t_img=10.0,
        ) 

        # Expect: UPDATE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.UPDATE)
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.UPDATE)

        # CASE 3.4.6:  Sender has a more recent communication with 3rd party, a more oudated communication with 4th party, a higher bid, and a later observation time
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.THIRD_PARTY,
            t_stamps={self.THIRD_PARTY : 10.0, self.FOURTH_PARTY : 1.0},
            winning_bid=15.0,
            t_img=15.0,
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.FOURTH_PARTY,
            t_stamps={self.THIRD_PARTY : 5.0, self.FOURTH_PARTY : 5.0},
            winning_bid=10.0,
            t_img=10.0,
        ) 

        # Expect: UPDATE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.UPDATE)
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.UPDATE)

        # CASE 3.4.7:  Sender has a more recent communication with 3rd party, a more oudated communication with 4th party, a lower bid, and an earlier observation time
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.THIRD_PARTY,
            t_stamps={self.THIRD_PARTY : 10.0, self.FOURTH_PARTY : 1.0},
            winning_bid=5.0,
            t_img=5.0,
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.FOURTH_PARTY,
            t_stamps={self.THIRD_PARTY : 5.0, self.FOURTH_PARTY : 5.0},
            winning_bid=10.0,
            t_img=10.0,
        ) 
        
        # Expect: UPDATE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.UPDATE)
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.UPDATE)

        # CASE 3.4.8:  Sender has a more recent communication with 3rd party, a more oudated communication with 4th party, a lower bid, and a later observation time
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.THIRD_PARTY,
            t_stamps={self.THIRD_PARTY : 10.0, self.FOURTH_PARTY : 1.0},
            winning_bid=5.0,
            t_img=15.0,
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.FOURTH_PARTY,
            t_stamps={self.THIRD_PARTY : 5.0, self.FOURTH_PARTY : 5.0},
            winning_bid=10.0,
            t_img=10.0,
        ) 

        # Expect: LEAVE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.LEAVE)
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.LEAVE)

        # CASE 3.4.9:  Sender has a more oudated communication with 3rd party, a more recent communication with 4th party, a higher bid, and an earlier observation time
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.THIRD_PARTY,
            t_stamps={self.THIRD_PARTY : 10.0, self.FOURTH_PARTY : 10.0},
            winning_bid=15.0,
            t_img=5.0,
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.FOURTH_PARTY,
            t_stamps={self.THIRD_PARTY : 15.0, self.FOURTH_PARTY : 5.0},
            winning_bid=10.0,
            t_img=10.0,
        ) 

        # Expect: RESET
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.RESET)
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.RESET)

        # CASE 3.4.10: Sender has a more oudated communication with 3rd party, a more recent communication with 4th party, a higher bid, and a later observation time
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.THIRD_PARTY,
            t_stamps={self.THIRD_PARTY : 10.0, self.FOURTH_PARTY : 10.0},
            winning_bid=15.0,
            t_img=15.0,
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.FOURTH_PARTY,
            t_stamps={self.THIRD_PARTY : 15.0, self.FOURTH_PARTY : 5.0},
            winning_bid=10.0,
            t_img=10.0,
        ) 

        # Expect: RESET
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.RESET)
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.RESET)

        # CASE 3.4.11: Sender has a more oudated communication with 3rd party, a more recent communication with 4th party, a lower bid, and an earlier observation time
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.THIRD_PARTY,
            t_stamps={self.THIRD_PARTY : 10.0, self.FOURTH_PARTY : 10.0},
            winning_bid=5.0,
            t_img=5.0,
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.FOURTH_PARTY,
            t_stamps={self.THIRD_PARTY : 15.0, self.FOURTH_PARTY : 5.0},
            winning_bid=10.0,
            t_img=10.0,
        ) 

        # Expect: RESET
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.RESET)
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.RESET)

        # CASE 3.4.12: Sender has a more oudated communication with 3rd party, a more recent communication with 4th party, a lower bid, and a later observation time
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.THIRD_PARTY,
            t_stamps={self.THIRD_PARTY : 10.0, self.FOURTH_PARTY : 10.0},
            winning_bid=15.0,
            t_img=15.0,
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.FOURTH_PARTY,
            t_stamps={self.THIRD_PARTY : 15.0, self.FOURTH_PARTY : 5.0},
            winning_bid=10.0,
            t_img=10.0,
        ) 

        # Expect: RESET
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.RESET) 
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.RESET)

        # CASE 3.4.13: Sender has a more oudated communication with 3rd party, a more oudated communication with 4th party, a higher bid, and an earlier observation time
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.THIRD_PARTY,
            t_stamps={self.THIRD_PARTY : 10.0, self.FOURTH_PARTY : 1.0},
            winning_bid=15.0,
            t_img=5.0,
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.FOURTH_PARTY,
            t_stamps={self.THIRD_PARTY : 15.0, self.FOURTH_PARTY : 5.0},
            winning_bid=10.0,
            t_img=10.0,
        ) 

        # Expect: LEAVE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.LEAVE) 
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.LEAVE)

        # CASE 3.4.14: Sender has a more oudated communication with 3rd party, a more oudated communication with 4th party, a higher bid, and a later observation time
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.THIRD_PARTY,
            t_stamps={self.THIRD_PARTY : 10.0, self.FOURTH_PARTY : 1.0},
            winning_bid=15.0,
            t_img=15.0,
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.FOURTH_PARTY,
            t_stamps={self.THIRD_PARTY : 15.0, self.FOURTH_PARTY : 5.0},
            winning_bid=10.0,
            t_img=10.0,
        ) 

        # Expect: LEAVE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.LEAVE) 
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.LEAVE)

        # CASE 3.4.15: Sender has a more oudated communication with 3rd party, a more oudated communication with 4th party, a lower bid, and an earlier observation time
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.THIRD_PARTY,
            t_stamps={self.THIRD_PARTY : 10.0, self.FOURTH_PARTY : 1.0},
            winning_bid=5.0,
            t_img=5.0,
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.FOURTH_PARTY,
            t_stamps={self.THIRD_PARTY : 15.0, self.FOURTH_PARTY : 5.0},
            winning_bid=10.0,
            t_img=10.0,
        ) 

        # Expect: LEAVE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.LEAVE) 
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.LEAVE)

        # CASE 3.4.16: Sender has a more oudated communication with 3rd party, a more oudated communication with 4th party, a lower bid, and a later observation time
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.THIRD_PARTY,
            t_stamps={self.THIRD_PARTY : 10.0, self.FOURTH_PARTY : 1.0},
            winning_bid=5.0,
            t_img=15.0,
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.FOURTH_PARTY,
            t_stamps={self.THIRD_PARTY : 15.0, self.FOURTH_PARTY : 5.0},
            winning_bid=10.0,
            t_img=10.0,
        ) 
        
        # Expect: LEAVE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.LEAVE) 
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.LEAVE)

        # CASE 3.5: Receiver has no winner
        # CASE 3.5.1: Sender has a more recent communication with 3rd party
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.THIRD_PARTY,
            t_stamps={self.THIRD_PARTY : 10.0},
            winning_bid=5.0,
            t_img=15.0,
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=Bid.NONE,
            t_stamps={self.THIRD_PARTY : 5.0},
            winning_bid=10.0,
            t_img=10.0,
        )  

        # Expect: UPDATE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.UPDATE)   
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.UPDATE)   

        # CASE 3.5.2: Sender has a more outdated communication with 3rd party
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=self.THIRD_PARTY,
            t_stamps={self.THIRD_PARTY : 1.0},
            winning_bid=5.0,
            t_img=15.0,
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=Bid.NONE,
            t_stamps={self.THIRD_PARTY : 5.0},
            winning_bid=10.0,
            t_img=10.0,
        )  
        
        # Expect: LEAVE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.LEAVE) 
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.LEAVE)     

    def test_compare_sender_has_no_winner(self):
        """
        CASE 4: Sender thinks no one is winning this bid
        """
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=Bid.NONE,
            winning_bid=0.0,
        )

        # CASE 4.1: Receiver thinks itself winner
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.RECEIVER,
            winning_bid=0.0,
        )  

        # Expect: LEAVE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.LEAVE)
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.LEAVE)

        # CASE 4.2: Receiver thinks sender is the winner
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.SENDER,
            winning_bid=0.0,
        )  

        # Expect: UPDATE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.UPDATE)
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.UPDATE)

        # CASE 4.3: Receiver thinks a third party is the winner 
        # CASE 4.3.1: Sender has a more recent bid
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=Bid.NONE,
            t_bid=10.0,
            t_img=np.NINF
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.THIRD_PARTY,
            t_bid=1.0,
            t_img=20.0,
        )
        # Expect: UPDATE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.UPDATE)
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.UPDATE)

        # CASE 4.3.2: Receiver has a more recent bid
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=Bid.NONE,
            t_bid=10.0,
            t_img=np.NINF
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=self.THIRD_PARTY,
            t_img=12.0,
            t_bid=11.0
        )
        # Expect: LEAVE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.LEAVE)
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.LEAVE)
        
        # CASE 1.4: Receiver has no winner
        bid_sender = self.make_bid(
            bidder=self.SENDER,
            winner=Bid.NONE,
            t_stamps={self.SENDER : 1.0, self.THIRD_PARTY : 1.0},
        )
        bid_receiver = self.make_bid(
            bidder=self.RECEIVER,
            winner=Bid.NONE,
            t_stamps={self.THIRD_PARTY : 5.0},
        )
        # Expect: LEAVE
        action = bid_receiver._rule_comparison(bid_sender)
        self.assertEqual(action, BidComparisonResults.LEAVE)
        action = bid_receiver._rule_comparison(bid_sender.to_dict())
        self.assertEqual(action, BidComparisonResults.LEAVE)


    # ----------------------------------------------------------------------
    # Modifiers: update / set / set_performed / has_winner
    # ----------------------------------------------------------------------

    def test_update_inputs(self):
        """
        Scenario where compare() returns UPDATE and we apply update().
        """
        base_bid : Bid = self.make_bid(
            bidder="agentA",
            winner="agentA",
            winning_bid=5.0,
            t_img=10.0,
        )
        other_bid : Bid = self.make_bid(
            bidder="agentB",
            winner="agentB",
            winning_bid=10.0,
            t_img=20.0,
        )
        
        # update bid
        with self.assertRaises(AssertionError): base_bid.update("not a bid or dict", t_comp=3.0)
        with self.assertRaises(AssertionError): base_bid.update(other_bid, t_comp=-1.0)

    def test_update_applies_update_info(self):
        """
        Scenario where compare() returns UPDATE and we apply update().
        """
        base_bid = self.make_bid(
            bidder="agentA",
            winner="agentA",
            winning_bid=5.0,
            t_img=10.0,
        )
        other_bid = self.make_bid(
            bidder="agentB",
            winner="agentB",
            winning_bid=10.0,
            t_img=20.0,
        )

        # sanity check: compare says UPDATE
        action = base_bid._rule_comparison(other_bid)
        self.assertEqual(action, BidComparisonResults.UPDATE)
        
        # update bid
        updated : Bid = base_bid.update(other_bid, t_comp=3.0)

        # check updated values
        self.assertAlmostEqual(updated.winning_bid, other_bid.winning_bid)
        self.assertEqual(updated.winner, other_bid.winner)
        self.assertAlmostEqual(updated.t_img, other_bid.t_img)
        self.assertAlmostEqual(updated.t_stamps[other_bid.owner], 3.0)
        
        # check a new object was created
        self.assertIsNot(base_bid, updated)

        # ensure the base bid's winner values were not modified
        self.assertNotAlmostEqual(base_bid.winning_bid, updated.winning_bid) 
        self.assertNotEqual(base_bid.winner, updated.winner)  
        self.assertNotAlmostEqual(base_bid.t_img, updated.t_img)  

        # ensure the base bid's timestamp for other bidder was modified
        self.assertAlmostEqual(base_bid.t_stamps[other_bid.owner], 3.0)

        # ensure the bids tasks are the same
        self.assertEqual(base_bid.task, updated.task)
        self.assertIs(base_bid.task, updated.task)
        self.assertIs(base_bid.t_stamps, updated.t_stamps)

    def test_update_applies_update_info_dict(self):
        """
        Scenario where compare() returns UPDATE and we apply update().
        """
        base_bid = self.make_bid(
            bidder="agentA",
            winner="agentA",
            winning_bid=5.0,
            t_img=10.0,
        )
        other_bid = self.make_bid(
            bidder="agentB",
            winner="agentB",
            winning_bid=10.0,
            t_img=20.0,
        )
        other_bid_dict = other_bid.to_dict()  # to exercise dict path

        # sanity check: compare says UPDATE
        action = base_bid._rule_comparison(other_bid_dict)
        self.assertEqual(action, BidComparisonResults.UPDATE)
        
        # update bid
        updated : Bid = base_bid.update(other_bid_dict, t_comp=3.0)

        # check updated values
        self.assertAlmostEqual(updated.winning_bid, other_bid.winning_bid)
        self.assertEqual(updated.winner, other_bid.winner)
        self.assertAlmostEqual(updated.t_img, other_bid.t_img)
        self.assertAlmostEqual(updated.t_stamps[other_bid.owner], 3.0)
        
        # check a new object was created
        self.assertIsNot(base_bid, updated)

        # ensure the base bid's winner values were not modified
        self.assertNotAlmostEqual(base_bid.winning_bid, updated.winning_bid) 
        self.assertNotEqual(base_bid.winner, updated.winner)  
        self.assertNotAlmostEqual(base_bid.t_img, updated.t_img)  

        # ensure the base bid's timestamp for other bidder was modified
        self.assertAlmostEqual(base_bid.t_stamps[other_bid.owner], 3.0)

        # ensure the bids tasks are the same
        self.assertEqual(base_bid.task, updated.task)
        self.assertIs(base_bid.task, updated.task)
        self.assertIs(base_bid.t_stamps, updated.t_stamps)

    def test_set_and_has_winner(self):
        bid = self.make_bid(
            bidder="agentA",
            winner=Bid.NONE,
            winning_bid=0.0,
            t_img=np.NINF,
            n_obs=2,
        )

        self.assertFalse(bid.has_winner())

        bid.set(main_measurement='VNIR', bid_value=7.5, t_img=12.0, t_update=3.0)

        self.assertTrue(bid.has_winner())
        self.assertEqual(bid.winner, "agentA")
        self.assertEqual(bid.main_measurement, 'VNIR')
        self.assertAlmostEqual(bid.winning_bid, 7.5)
        self.assertAlmostEqual(bid.t_img, 12.0)
        self.assertEqual(bid.n_obs, 2)
        self.assertAlmostEqual(bid.t_stamps["agentA"], 3.0)

    # def test_set_performed_marks_performed_and_updates_time(self):
    #     bid = self.make_bid(
    #         bidder="agentA",
    #         winner=Bid.NONE,
    #         winning_bid=0.0,
    #         t_img=np.NINF,
    #         t_bid=0.0,
    #     )

    #     bid.set_performed(t=50.0, performed=True, performer="agentB")

    #     self.assertTrue(bid.performed)
    #     self.assertEqual(bid.winner, "agentB")
    #     self.assertAlmostEqual(bid.t_img, 50.0)
    #     self.assertAlmostEqual(bid.t_stamp, 50.0)

if __name__ == '__main__':
    # terminal welcome message
    print_scenario_banner('Task Bid Test')
    
    # run tests
    unittest.main()
