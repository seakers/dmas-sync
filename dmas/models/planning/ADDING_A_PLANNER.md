# Adding a Planner

A planner decides *what an agent does next*. DMAS calls it from `SimulationAgent.decide_action()` every time the agent
gets control (see ARCHITECTURE.md, diagrams 3 and 4).

## 1. Pick a base class

| You want to... | Subclass | Configured under | Typical use |
|---|---|---|---|
| Build a fresh plan every `period` seconds over a `horizon` | `AbstractPeriodicPlanner` (`models/planning/periodic.py`) | `planner.preplanner` | Greedy, DP, MILP, event announcers |
| Repair the current plan when something changes (new task, new bids, ...) | `AbstractReactivePlanner` (`models/planning/reactive.py`) | `planner.replanner` | Heuristic insertion, consensus (CBBA) |

An agent can have both: the preplanner produces a `PeriodicPlan`, and the replanner modifies it as percepts arrive.

## 2. Implement the required methods

### Periodic planner
`AbstractPeriodicPlanner` already implements the planning template: it collects available tasks, computes access
opportunities, builds `ObservationOpportunity` objects, **asserts your observation path is valid**, then adds maneuvers,
broadcasts and the next-replan action. You provide only:

| Method | Must return |
|---|---|
| `_schedule_observations(state, specs, orbitdata, observation_opportunities, mission, observation_history)` | `list[ObservationAction]`, feasible for the agent's slew limits |
| `_schedule_broadcasts(state, observations, orbitdata, t=None)` | `list[BroadcastMessageAction]` (or future broadcasts) |
| `print_results()` | nothing; close any `DataSink`s you open |

The smallest working example is `decentralized/blank.py` (16 lines). `decentralized/earliest.py` shows the heuristic-insertion
pattern: subclass and override only `_calc_heuristic`.

### Reactive planner
You implement three methods. Signatures follow the concrete planners, not the base `AbstractPlanner`:

| Method | Called | Purpose |
|---|---|---|
| `update_percepts(state, current_plan, tasks, incoming_reqs, misc_messages, completed_actions, aborted_actions, pending_actions)` | every decision cycle | Ingest new information. Bids and other planner messages arrive in `misc_messages`. |
| `needs_planning(state, specs, current_plan, orbitdata)` -> `bool` | every decision cycle | Cheap check for whether to replan now |
| `generate_plan(state, specs, current_plan, orbitdata, mission, tasks, observation_history)` -> `Plan` | only if `needs_planning` is true | Return the repaired plan |

`consensus/consensus.py` is the full-featured reference. Start from `decentralized/heuristic.py`
(`HeuristicInsertionReactivePlanner`) for something simpler.

## 3. Rules the framework relies on
- **Planning takes no mission time.** Only actions advance the clock, so a slow planner is not penalized in simulated time.
- **Plans must be feasible.** Periodic planners are checked with `is_observation_path_valid`; use `_schedule_maneuvers`
  and `is_maneuver_path_valid` from `AbstractPlanner` rather than reimplementing slew logic.
- **Communicate through actions.** To share information, put `FutureBroadcastMessageAction`s (types: `PLAN`, `BIDS`,
  `REQUESTS`, `OBSERVATIONS`, `REWARD_GRID`, `STATE`) into your plan. The agent compiles the contents when they come due.
- **The agent must always have a next action.** `get_next_planned_action` raises if the plan yields none, so keep a
  wait/replan action at the end of your plans (the periodic template does this for you).
- **Reusable helpers** on `AbstractPlanner`: `calculate_access_opportunities`, `create_observation_opportunities_from_accesses`,
  `estimate_observation_opportunity_value`, `cluster_task_observation_opportunities`, `get_available_accesses`.

## 4. Register it
Planners are selected by an `if/elif` chain in `dmas/core/simulation.py`:
`Simulation.__load_preplanner` and `Simulation.__load_replanner` (there is a `# add more preplanners here` marker).
Add a branch that maps your `@type` string to your class and reads any extra parameters from the config dict.

## 5. Configure it
Planner specs are JSON, attached to each agent under `planner`. Example (from `experiments/*/resources/templates/planners.json`):

```json
{
  "preplanners": {
    "greedy": { "@type": "earliest", "period": 500, "sharing": "periodic" }
  },
  "replanners": {
    "cbba": { "@type": "consensus", "model": "heuristicInsertion",
              "heuristic": "taskPriority", "replanThreshold": 1,
              "optimisticBiddingThreshold": 1 }
  }
}
```
Common preplanner keys: `@type`, `period`, `horizon` (defaults to `period`), `sharing` (`none` / `periodic` / `opportunistic`), `debug`.

## 6. Test it
Copy the pattern in `tests/planners/` (e.g. `earliest_test.py`, built on `tester.py`'s `PlannerTester`), which builds small
toy scenarios (single satellite / one event, two satellites / one event). Run with `make runtest` or
`cd tests && python -m unittest discover`.
