from chess3d.agents.planning.periodic import AbstractPeriodicPlanner


class BlankPlanner(AbstractPeriodicPlanner):
    """
    A blank planner that does nothing. Used for testing purposes.
    """
    def _schedule_observations(self, *_) -> list:
        """ No observations scheduled. """
        return []
    
    def _schedule_broadcasts(self, *_) -> list:
        """ No broadcasts scheduled. """
        return []