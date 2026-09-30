# DMAS Architecture

This page documents the synchronous DMAS implementation (`dmas/`). The `network/` package
(asynchronous execution) is under development and is **not** covered here. Diagram sources live in
`docs/diagrams/uml/*.mmd` (Mermaid, rendered natively by GitHub).

## 1. System context
What `dmas` is built from and what it produces.

```mermaid
flowchart LR
    subgraph EXP["experiments/&lt;study&gt;/"]
        study["study.py<br/>(trials CSV + templates)"]
        tmpl["resources/templates<br/>planners.json, spacecraft.json,<br/>MissionSpecs.json, ..."]
    end

    subgraph EXT["External libraries"]
        orbitpy["orbitpy / instrupy<br/>orbit + instrument propagation"]
        execsatm["execsatm<br/>Mission, Task, Requirement,<br/>ObservationOpportunity"]
    end

    subgraph DMAS["dmas (this repo)"]
        sim["core.Simulation<br/>from_dict() / execute()"]
        env["core.SimulationEnvironment<br/>step()"]
        agents["models.SimulationAgent (x N)<br/>decide_action()"]
        planners["models.planning<br/>preplanner + replanner"]
        orbitdata["utils.OrbitData<br/>precomputed access + comms intervals"]
        proc["utils.processing<br/>process_results / summarize_results"]
    end

    results[("results/&lt;scenario&gt;/<br/>per-agent + environment logs")]
    plots["experiments/&lt;study&gt;/analysis<br/>plots and tables"]

    study --> tmpl
    study -- "scenario dict" --> sim
    sim -- "precompute()" --> orbitdata
    orbitdata -. uses .-> orbitpy
    sim -. "missions, tasks" .-> execsatm
    sim -- "builds" --> env
    sim -- "builds via factories" --> agents
    agents --- planners
    agents -. reads .-> orbitdata
    env -. reads .-> orbitdata
    sim -- "execute()" --> results
    proc -- reads --> results
    proc --> plots
```

- `Simulation.from_dict()` turns a scenario dictionary into agents, an environment, and precomputed orbit data.
- Orbit propagation and instrument models come from `orbitpy` / `instrupy`. Missions, tasks, requirements and
  observation opportunities come from `execsatm`.
- Results are written per agent under the scenario results directory. `utils/processing.py` reads them back for analysis.

## 2. Planner class hierarchy (the extension point)
Every agent holds at most one **preplanner** (`AbstractPeriodicPlanner`) and at most one **replanner**
(`AbstractReactivePlanner`). Both derive from `AbstractPlanner`. To add an autonomy approach you subclass one of them
(see [ADDING_A_PLANNER.md](ADDING_A_PLANNER.md)).

```mermaid
classDiagram
    direction TB

    class SimulationAgent {
        +decide_action(curr_state, prev_action, prev_status, incoming_messages, my_measurements) (state, action)
        +get_next_planned_action(state) AgentAction
        -_plan : Plan
        -_known_tasks
        -_observations_tracker
    }

    class AbstractPlanner {
        <<abstract>>
        +update_percepts(...)*
        +needs_planning(...) bool*
        +generate_plan(...) Plan*
        +print_results()*
        +calculate_access_opportunities()
        +create_observation_opportunities_from_accesses()
        +estimate_observation_opportunity_value()
        +is_observation_path_valid()
    }

    class AbstractPeriodicPlanner {
        <<abstract>>
        horizon, period, sharing
        +needs_planning(state, specs, plan) bool
        +generate_plan(state, specs, orbitdata, mission, tasks, obs_history) PeriodicPlan
        #_schedule_observations(...) list*
        #_schedule_broadcasts(...) list*
    }

    class AbstractReactivePlanner {
        <<abstract>>
        +update_percepts(...)*
        +needs_planning(...) bool*
        +generate_plan(state, specs, current_plan, orbitdata, mission, tasks, obs_history) Plan*
    }

    AbstractPlanner <|-- AbstractPeriodicPlanner
    AbstractPlanner <|-- AbstractReactivePlanner
    SimulationAgent o-- "0..1" AbstractPeriodicPlanner : preplanner
    SimulationAgent o-- "0..1" AbstractReactivePlanner : replanner

    %% Periodic family
    class HeuristicInsertionPeriodicPlanner
    class EarliestAccessPeriodicPlanner
    class EarliestRequestArrivalPeriodicPlanner
    class NadirPointingPeriodicPlanner
    class DynamicProgrammingPlanner
    class BlankPlanner
    class AbstractEventAnnouncerPlanner {
        <<abstract>>
    }
    class InstantEventAnnouncerPlanner
    class GroundProcessorEventAnnouncerPlanner
    class DealerPlanner
    class DealerMILPPlanner
    class TestingDealer
    class WorkerPlanner

    AbstractPeriodicPlanner <|-- HeuristicInsertionPeriodicPlanner
    HeuristicInsertionPeriodicPlanner <|-- EarliestAccessPeriodicPlanner
    HeuristicInsertionPeriodicPlanner <|-- EarliestRequestArrivalPeriodicPlanner
    EarliestAccessPeriodicPlanner <|-- NadirPointingPeriodicPlanner
    AbstractPeriodicPlanner <|-- DynamicProgrammingPlanner
    AbstractPeriodicPlanner <|-- BlankPlanner
    AbstractPeriodicPlanner <|-- AbstractEventAnnouncerPlanner
    AbstractEventAnnouncerPlanner <|-- InstantEventAnnouncerPlanner
    AbstractEventAnnouncerPlanner <|-- GroundProcessorEventAnnouncerPlanner
    AbstractPeriodicPlanner <|-- DealerPlanner
    DealerPlanner <|-- DealerMILPPlanner
    DealerPlanner <|-- TestingDealer
    AbstractPeriodicPlanner <|-- WorkerPlanner

    %% Reactive family
    class HeuristicInsertionReactivePlanner
    class EarliestAccessReactivePlanner
    class EarliestRequestArrivalReactivePlanner
    class NadirPointingReactivePlanner
    class FixedPointingDefaultPlanner
    class ConsensusPlanner {
        <<abstract>>
        #_bundle_building_phase(...)*
        #_build_bundle_from_preplan(...)*
    }
    class HeuristicInsertionConsensusPlanner
    class AugmentedConsensusPlanner
    class AugmentedHeuristicInsertionConsensusPlanner

    AbstractReactivePlanner <|-- HeuristicInsertionReactivePlanner
    HeuristicInsertionReactivePlanner <|-- EarliestAccessReactivePlanner
    HeuristicInsertionReactivePlanner <|-- EarliestRequestArrivalReactivePlanner
    EarliestAccessReactivePlanner <|-- NadirPointingReactivePlanner
    AbstractReactivePlanner <|-- FixedPointingDefaultPlanner
    AbstractReactivePlanner <|-- ConsensusPlanner
    ConsensusPlanner <|-- HeuristicInsertionConsensusPlanner
    ConsensusPlanner <|-- AugmentedConsensusPlanner
    HeuristicInsertionConsensusPlanner <|-- AugmentedHeuristicInsertionConsensusPlanner
    AugmentedConsensusPlanner <|-- AugmentedHeuristicInsertionConsensusPlanner

    %% Plans
    class Plan {
        <<abstract>>
        +add(action, t)
        +get_next_actions(t)
        +update_action_completion(...)
    }
    class PeriodicPlan
    class ReactivePlan
    Plan <|-- PeriodicPlan
    Plan <|-- ReactivePlan
    AbstractPlanner ..> Plan : generate_plan returns
```

Note: `ConsensusPlanner` (the SC-CBBA family) and the centralized planners live in folders without an `__init__.py`,
so tools such as `pyreverse` skip them; they are drawn here from the source.

## 3. Simulation loop
The loop is synchronous and event-driven. Mission time `t` advances **only** to the earliest end time of the actions the
agents chose. Planner computation (including consensus) advances no mission time; it costs wall-clock time only.

```mermaid
sequenceDiagram
    autonumber
    participant Sim as Simulation.execute()
    participant Env as SimulationEnvironment
    participant A as SimulationAgent (each)
    participant Pre as Preplanner
    participant Re as Replanner

    Note over Sim: t = 0, every agent starts with (initial_state, None)
    loop while t < tf
        Sim->>Env: step(state_action_pairs, t)
        Note right of Env: applies each agent's current action,<br/>updates connectivity, routes broadcasts,<br/>resolves observations
        Env-->>Sim: percepts per agent<br/>(state, prev_action, status, messages, measurements)
        loop for each agent
            Sim->>A: decide_action(*percepts)
            A->>Pre: update_percepts(...)
            A->>Pre: needs_planning(...)?
            opt preplan needed
                A->>Pre: generate_plan(...)
                Pre-->>A: PeriodicPlan
            end
            A->>Re: update_percepts(...)
            A->>Re: needs_planning(...)?
            opt replan needed
                A->>Re: generate_plan(..., current_plan, ...)
                Re-->>A: Plan
            end
            A-->>Sim: (next_state, next_action)
        end
        Note over Sim: t += min(action.t_end) - t<br/>Planning consumed no mission time.<br/>Only actions (observe, maneuver, wait, ...) advance t.
    end
    Sim->>Env: step(pairs, tf) (final flush)
    Sim->>A: decide_action(...) (final)
    Sim->>A: print_results() (each agent)
```

## 4. Inside `SimulationAgent.decide_action`
The per-agent decision cycle. This is where your planner is called.

```mermaid
flowchart TD
    start(["decide_action(curr_state, prev_action, prev_status,<br/>incoming_messages, my_measurements)"])
    s1["Update state + state history"]
    s2["Classify incoming messages:<br/>task requests, external observations,<br/>reward grids, states, action statuses, misc"]
    s3["Process action completion<br/>(completed / aborted / pending)"]
    s4["Update known tasks + requests"]
    s5["Update plan completion"]
    s6["Process own measurements<br/>(may generate new task requests)"]
    s7["Update observation history + tracker<br/>(own + external observations)"]
    pre{"preplanner<br/>assigned?"}
    pre1["preplanner.update_percepts()"]
    pre2{"preplanner.<br/>needs_planning()?"}
    pre3["preplanner.generate_plan()<br/>-> new PeriodicPlan"]
    re{"replanner<br/>assigned?"}
    re1["replanner.update_percepts()"]
    re2{"replanner.<br/>needs_planning()?"}
    re3["replanner.generate_plan(current_plan)<br/>-> repaired Plan"]
    n1["get_next_planned_action():<br/>next actions from plan,<br/>materialize FutureBroadcast actions,<br/>attach observation requests, validate"]
    n2["Prepare next state for the chosen action"]
    done(["return (next_state, next_action)"])

    start --> s1 --> s2 --> s3 --> s4 --> s5 --> s6 --> s7 --> pre
    pre -- yes --> pre1 --> pre2
    pre -- no --> re
    pre2 -- yes --> pre3 --> re
    pre2 -- no --> re
    re -- yes --> re1 --> re2
    re -- no --> n1
    re2 -- yes --> re3 --> n1
    re2 -- no --> n1
    n1 --> n2 --> done
```

## 5. Message flow between agents
Planners do not send messages directly. They schedule `FutureBroadcastMessageAction`s in their plan; the agent compiles
the message contents when the action comes due; the environment delivers them.

```mermaid
sequenceDiagram
    autonumber
    participant P as Planner (agent A)
    participant A as SimulationAgent A
    participant Sim as Simulation loop
    participant E as SimulationEnvironment
    participant B as SimulationAgent B (in A's connected component)
    participant PB as Planner (agent B)

    P->>A: generate_plan() returns a Plan containing<br/>FutureBroadcastMessageAction(type, t_start)
    Note over A: at t_start, get_next_planned_action()
    A->>A: _materialize_future_broadcasts():<br/>compile messages (STATE, BIDS, REQUESTS,<br/>OBSERVATIONS, REWARD_GRID, PLAN)<br/>into one BusMessage
    A-->>Sim: BroadcastMessageAction
    Sim->>E: step(state_action_pairs, t)
    E->>E: __perform_broadcast(): log message,<br/>set state to MESSAGING
    E->>E: route msg to every agent in the sender's<br/>current connectivity component
    E-->>Sim: percepts (B receives msgs)
    Sim->>B: decide_action(..., incoming_messages)
    B->>B: __classify_incoming_messages()
    B->>PB: update_percepts(..., new_reqs, misc_messages, ...)
    Note over PB: e.g. ConsensusPlanner collects bids,<br/>runs consensus phase, may trigger replan
```

Delivery in `SimulationEnvironment.step()` goes to every agent in the sender's *current* connectivity component, taken
from the precomputed `comms_links` intervals in `OrbitData`.
