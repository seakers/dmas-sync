import json
import os
import random
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import pygad
from tqdm import tqdm
from dataclasses import dataclass

from dmas.utils.orbitdata import OrbitData
from utils.factory import get_base_path, load_templates, generate_scenario_mission_specs

from dmas.utils.tools import print_scenario_banner

"""
CONNECTIVITY STUDY
Generates connectivity specs for the high, medium, and low latency scenarios for each trial in the CBBA stress test study.
- High Latency Scenario: ISLs off, ISL to TDRSS off, SAT to NEN comms on
- Low Latency Scenario: ISLs on, TDRSS comms on, SAT to NEN comms on
- Medium Latency Scenario: 
    Uses the evaluated latency values for the high and low latency scenarios to generate a medium latency connectivity spec that is designed to have latency values between the two.
    Extracts latency requirement from high and low lavency values and uses Genetic Algorithm to search for a connectivity spec that meets
    the medium latency requirement.
"""

def generate_high_latency_connectivity_spec(imaging_sats : list, relay_sats : list, ground_stations : list) -> dict:
    # High Latency Scenario: ISLs off, ISL to TDRSS off, SAT to NEN comms on
    return {
        "default": "deny",

        "groups": {
            "instruments": imaging_sats,
            "relays": relay_sats,
            "groundStations": ground_stations
        },

        "rules": [
            {
            "action": "allow",
            "scope": "between",
            "targets": ["instruments", "groundStations"]
            },
            {
            "action": "deny",
            "scope": "within",
            "targets": "instruments"
            },
            {
            "action": "deny",
            "scope": "within",
            "targets": "relays"
            }
        ],

        "overrides": []
    }

def generate_low_latency_connectivity_spec(imaging_sats : list, relay_sats : list, ground_stations : list) -> dict:
    # Low Latency Scenario: ISLs on, TDRSS comms on, SAT to NEN comms on
    return {
        "default": "deny",

        "groups": {
            "instruments": imaging_sats,
            "relays": relay_sats,
            "groundStations": ground_stations
        },

        "rules": [
            {
            "action": "allow",
            "scope": "between",
            "targets": ["instruments", "relays", "groundStations"]
            },
            {
            "action": "allow",
            "scope": "within",
            "targets": "instruments"
            },
            {
            "action": "allow",
            "scope": "within",
            "targets": "relays"
            }
        ],

        "overrides": []
    }

def generate_intermediate_latency_connectivity_spec(imaging_sats : list, relay_sats : list, ground_stations : list) -> dict:
    # Intermediate Latency Scenario: ISLs off, TDRSS comms on, SAT to NEN comms off
    return {
        "default": "deny",

        "groups": {
            "instruments": imaging_sats,
            "relays": relay_sats,
            "groundStations": ground_stations
        },

        "rules": [
            {
            "action": "allow",
            "scope": "between",
            "targets": ["instruments", "relays"]
            },
            {
            "action": "allow",
            "scope": "between",
            "targets": ["relays", "groundStations"]
            },
            {
            "action": "deny",
            "scope": "within",
            "targets": "instruments"
            }
        ],

        "overrides": []
    }

EdgeIdx = Tuple[int, int]

# def generate_medium_latency_connectivity_spec(
#     imaging_sats: list,
#     relay_sats: list,
#     ground_stations: list,
#     high_latency_values: dict,
#     low_latency_values: dict,
#     messages: List[Tuple[str, str, float]],
#     medium_latency_filename: str,
#     *mission_specs_kwargs
# ) -> dict:

#     # ----------------------------
#     # 0) Targets derived from high/low
#     # ----------------------------
#     target_p95 = 0.5 * (low_latency_values["latency_95th"] + high_latency_values["latency_95th"])
#     target_p_on_time = 0.5 * (low_latency_values["p_on_time"] + high_latency_values["p_on_time"])
#     p_success_min = min(low_latency_values["p_success"], high_latency_values["p_success"])

#     # Tolerances for "medium" (tune these)
#     # e.g., within ±10% of target p95 and ±0.05 on p_on_time
#     p95_tol_frac = 0.10
#     p_on_time_tol = 0.05

#     # ----------------------------
#     # 1) Build agent list + candidate sat-sat edges
#     # ----------------------------
#     agents = list(imaging_sats) + list(relay_sats) + list(ground_stations)
#     n = len(agents)

#     imaging_set = set(imaging_sats)
#     relay_set = set(relay_sats)
#     ground_set = set(ground_stations)

#     def is_candidate_edge(u: str, v: str) -> bool:
#         # Only toggle sat<->sat
#         if (u in ground_set) or (v in ground_set):
#             return False
#         return True

#     edge_list: List[EdgeIdx] = []
#     for i in range(n):
#         for j in range(i + 1, n):
#             if is_candidate_edge(agents[i], agents[j]):
#                 edge_list.append((i, j))

#     if not edge_list:
#         raise ValueError("No candidate sat-sat edges were generated.")

#     # ----------------------------
#     # 2) Helpers: edge-set -> connectivity spec
#     # ----------------------------
#     def edge_set_to_connectivity_spec(enabled_edges: Set[EdgeIdx]) -> dict:
#         overrides = [{"pair": [agents[i], agents[j]], "action": "allow"} for (i, j) in sorted(enabled_edges)]

#         spec = {
#             "default": "deny",
#             "groups": {
#                 "imaging_sats": list(imaging_sats),
#                 "relay_sats": list(relay_sats),
#                 "ground_stations": list(ground_stations),
#             },
#             "rules": [
#                 {"action": "allow", "scope": "between", "targets": ["imaging_sats", "ground_stations"]},
#                 {"action": "allow", "scope": "between", "targets": ["relay_sats", "ground_stations"]},
#             ],
#             "overrides": overrides,
#         }
#         return spec

#     def evaluate_edge_set(enabled_edges: Set[EdgeIdx]) -> Dict[str, Any]:
#         conn_spec = edge_set_to_connectivity_spec(enabled_edges)

#         # save connectivity spec to file 
#         with open(medium_latency_filename, 'w') as f:
#             json.dump(conn_spec, f, indent=4)

#         mission_specs_candidate = generate_scenario_mission_specs(
#             *mission_specs_kwargs,  
#         )

#         metrics = evaluate_scenario_latency(
#             mission_specs_candidate,
#             messages,
#             # If you want p_on_time relative to a requirement, pass T_req here:
#             # T_req=...,
#             printouts=False,
#         )

#         # If evaluate_scenario_latency already returns a dict, great.
#         # If it returns tuple, wrap it here.
#         return metrics
    
#     # ----------------------------
#     # 3) Greedy forward-add to reach "medium"
#     # ----------------------------
#     eval_cache: Dict[frozenset, Dict[str, Any]] = {}

#     def eval_cached(E: Set[EdgeIdx]) -> Dict[str, Any]:
#         key = frozenset(E)
#         if key not in eval_cache:
#             eval_cache[key] = evaluate_edge_set(E)
#         return eval_cache[key]

#     def medium_ok(m: Dict[str, Any]) -> bool:
#         if float(m["p_success"]) < p_success_min:
#             return False
#         p95 = float(m["latency_95th"])
#         p_on = float(m["p_on_time"])
#         return (abs(p95 - target_p95) <= p95_tol_frac * target_p95) and (abs(p_on - target_p_on_time) <= p_on_time_tol)

#     # Start from empty sat-sat ISLs (but sat-ground allowed via rules)
#     enabled: Set[EdgeIdx] = set()
#     cur = eval_cached(enabled)

#     remaining = [e for e in edge_list if e not in enabled]

#     # To avoid O(|edges|) eval per step being too expensive, screen candidates per step
#     screen_k = min(200, len(remaining))  # tune

#     # We’ll optimize "distance to medium" rather than pure feasibility
#     def distance_to_medium(m: Dict[str, Any]) -> float:
#         # big penalty if violates success
#         if float(m["p_success"]) < p_success_min:
#             return 1e9 + (p_success_min - float(m["p_success"])) * 1e6

#         p95 = float(m["latency_95th"])
#         p_on = float(m["p_on_time"])
#         # normalized squared error
#         d_p95 = ((p95 - target_p95) / max(1e-9, target_p95)) ** 2
#         d_on = (p_on - target_p_on_time) ** 2
#         # small edge penalty to discourage unnecessary links
#         d_edges = 1e-4 * len(enabled)
#         return d_p95 + d_on + d_edges

#     best_dist = distance_to_medium(cur)

#     # Forward add loop
#     max_add_steps = 500  # safety
#     for _ in tqdm(range(max_add_steps), total=max_add_steps, desc="Greedy add", unit=' steps', disable=False):
#         if medium_ok(cur):
#             break
#         if not remaining:
#             break

#         # Screen candidates
#         if len(remaining) > screen_k:
#             cand_subset = np.random.choice(len(remaining), size=screen_k, replace=False)
#             pool = [remaining[i] for i in cand_subset]
#         else:
#             pool = remaining

#         best_edge = None
#         best_edge_metrics = None
#         best_edge_dist = best_dist

#         for e in tqdm(pool, desc="  Evaluating candidates", unit=' edges', leave=False):
#             trial = set(enabled)
#             trial.add(e)
#             m = eval_cached(trial)
#             d = distance_to_medium(m)
#             if d < best_edge_dist:
#                 best_edge_dist = d
#                 best_edge = e
#                 best_edge_metrics = m

#         # No improvement found
#         if best_edge is None:
#             break

#         enabled.add(best_edge)
#         remaining.remove(best_edge)
#         cur = best_edge_metrics
#         best_dist = best_edge_dist

#     # ----------------------------
#     # 4) Prune: remove edges while staying "medium-ish"
#     # ----------------------------
#     improved = True
#     while improved and enabled:
#         improved = False

#         # Evaluate removal candidates (screen if big)
#         enabled_list = list(enabled)
#         # You can sort by heuristic (e.g., remove edges involving imaging first), but simplest:
#         np.random.shuffle(enabled_list)

#         remove_screen_k = min(200, len(enabled_list))
#         pool = enabled_list[:remove_screen_k]

#         best_remove = None
#         best_remove_metrics = None
#         best_remove_dist = best_dist

#         for e in tqdm(pool, desc="  Evaluating removal candidates", unit=' edges', leave=False):
#             trial = set(enabled)
#             trial.remove(e)
#             m = eval_cached(trial)
#             # require still "medium" OR at least not worsen distance much
#             d = distance_to_medium(m)
#             # Prefer staying in band; if not in band, allow only if distance doesn't worsen
#             if d <= best_remove_dist and (medium_ok(m) or d <= best_dist):
#                 best_remove = e
#                 best_remove_metrics = m
#                 best_remove_dist = d

#         if best_remove is not None and medium_ok(best_remove_metrics):
#             enabled.remove(best_remove)
#             cur = best_remove_metrics
#             best_dist = best_remove_dist
#             improved = True

#     # ----------------------------
#     # 5) Save final spec and return
#     # ----------------------------
#     final_spec = edge_set_to_connectivity_spec(enabled)
#     with open(medium_latency_filename, "w") as f:
#         json.dump(final_spec, f, indent=4)

#     return final_spec

def generate_medium_latency_connectivity_spec(imaging_sats : list, 
                                              relay_sats : list, 
                                              ground_stations : list, 
                                              high_latency_values : dict, 
                                              low_latency_values : dict,
                                              messages : List[Tuple[str, str, float]],
                                              medium_latency_filename : str,
                                              *mission_specs_kwargs
                                            ) -> dict:
    """ 
    Uses the evaluated latency values for the high and low latency scenarios to generate 
    a medium latency connectivity spec that is designed to have latency values between the two.

    Extracts latency requirement from high and low lavency values and uses Genetic Algorithm to search 
    for a connectivity spec that meets the medium latency requirement.
    """
    
    # ----------------------------
    # 0) Targets derived from high/low
    # ----------------------------
    # Expect these keys exist in your metrics dict:
    #   "p_success", "p_on_time", "latency_mean", "latency_50th", "latency_95th", "latency_99th"
    target_p95 = 0.5 * (low_latency_values["latency_95th"] + high_latency_values["latency_95th"])
    target_p_on_time = 0.5 * (low_latency_values["p_on_time"] + high_latency_values["p_on_time"])

    # set successful delivery requirement 
    p_success_min = min(low_latency_values["p_success"], high_latency_values["p_success"]) 
    # p_success_min = 0.95 # Or set explicitly, e.g. 0.95

    # ----------------------------
    # 1) Build agent list + candidate edges
    # ----------------------------
    agents = list(imaging_sats) + list(relay_sats) + list(ground_stations)
    n = len(agents)
    agent_to_idx = {a: i for i, a in enumerate(agents)}

    imaging_set = set(imaging_sats)
    relay_set = set(relay_sats)
    ground_set = set(ground_stations)

    def is_candidate_edge(u: str, v: str) -> bool:
        """Restrict what the GA is allowed to turn on."""

        u_is_img = u in imaging_set
        v_is_img = v in imaging_set
        u_is_rel = u in relay_set
        v_is_rel = v in relay_set
        u_is_gs = u in ground_set
        v_is_gs = v in ground_set

        # if (u_is_gs and v_is_img) or (u_is_img and v_is_gs):
        if u_is_gs or v_is_gs:
            # Disallow direct toggling of sat<->ground links 
            # (already enabled in rules)
            return False

        return True  
    

        # Typical "medium latency" design space:
        # - allow imaging<->relay
        # - allow relay<->relay
        # - allow relay<->ground
        # - disallow imaging<->imaging (optional)
        # - disallow imaging<->ground direct (optional)
        if (u_is_img and v_is_rel) or (u_is_rel and v_is_img):
            return True
        if u_is_rel and v_is_rel:
            return True
        if (u_is_rel and v_is_gs) or (u_is_gs and v_is_rel):
            return True

        # Optional: allow imaging->ground direct links
        # if (u_is_img and v_is_gs) or (u_is_gs and v_is_img):
        #     return True

        return False

    # Enumerate undirected candidate edges as (i,j) with i<j
    edge_list: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            u = agents[i]
            v = agents[j]
            if is_candidate_edge(u, v):
                edge_list.append((i, j))

    if len(edge_list) == 0:
        raise ValueError("No candidate edges were generated. Check your candidate-edge rules.")

    num_genes = len(edge_list)

    # ----------------------------
    # 2) Helpers: chromosome -> adjacency -> connectivity spec
    # ----------------------------
    def chromosome_to_adjacency(solution: np.ndarray) -> np.ndarray:
        """Build an NxN symmetric adjacency matrix from a 0/1 chromosome."""
        adj = np.zeros((n, n), dtype=np.uint8)
        # Ensure ints 0/1
        bits = (solution > 0.5).astype(np.uint8)
        for gene_idx, (i, j) in enumerate(edge_list):
            if bits[gene_idx]:
                adj[i, j] = 1
                adj[j, i] = 1
        return adj

    def adjacency_to_connectivity_spec(adj: np.ndarray) -> dict:
        """
        Convert adjacency matrix to a connectivity spec dict.
        This format is intentionally simple: default deny + allowed pair overrides.
        """
        overrides = []
        for i, j in edge_list:
            if adj[i, j] == 1:
                overrides.append({"pair": [agents[i], agents[j]], "action": "allow"})

        spec = {
            "default": "deny",
            "groups": {
                "imaging_sats": list(imaging_sats),
                "relay_sats": list(relay_sats),
                "ground_stations": list(ground_stations),
            },
            "rules": [
                {
                    "action": "allow",
                    "scope": "between",
                    "targets": [
                        "imaging_sats",
                        "ground_stations"
                    ]
                },
                {
                    "action": "allow",
                    "scope": "between",
                    "targets": [
                        "relay_sats",
                        "ground_stations"
                    ]
                },
            ],
            "overrides": overrides,
        }
        return spec

    # ----------------------------
    # 3) Evaluation wrapper: spec -> mission_specs -> metrics dict
    # ----------------------------
    def evaluate_solution(solution: np.ndarray) -> Dict[str, Any]:
        
        # Convert chromosome to connectivity spec
        adj = chromosome_to_adjacency(solution)
        conn_spec = adjacency_to_connectivity_spec(adj)

        # save connectivity spec to file 
        with open(medium_latency_filename, 'w') as f:
            json.dump(conn_spec, f, indent=4)

        mission_specs_candidate = generate_scenario_mission_specs(
            *mission_specs_kwargs,  
        )

        metrics = evaluate_scenario_latency(
            mission_specs_candidate,
            messages,
            # If you want p_on_time relative to a requirement, pass T_req here:
            # T_req=...,
            printouts=False,
        )

        # If evaluate_scenario_latency already returns a dict, great.
        # If it returns tuple, wrap it here.
        return metrics, conn_spec, adj

    # ----------------------------
    # 4) Fitness function for pygad
    # ----------------------------
    # We use squared penalties for being far from targets and a mild edge penalty.
    # Also a big penalty if success is too low.
    #
    # Make sure these weights roughly match your units (seconds).
    w_p95 = 1.0
    w_on_time = 1.0
    w_edges = 0.01
    big_penalty = 1e6

    # Cache to avoid re-evaluating identical solutions (GA revisits often)
    eval_cache: Dict[bytes, float] = {}
    best_payload: Dict[str, Any] = {"fitness": -np.inf, "conn_spec": None, "metrics": None}

    def fitness_func(ga_instance, solution, solution_idx):
        # Cache key
        key = np.asarray(solution > 0.5, dtype=np.uint8).tobytes()
        if key in eval_cache:
            return eval_cache[key]

        try:
            metrics, conn_spec, adj = evaluate_solution(solution)
        except Exception:
            # If evaluation fails (infeasible spec, etc.), return very poor fitness
            fit = -big_penalty
            eval_cache[key] = fit
            return fit

        p_success = float(metrics["p_success"])
        p_on_time = float(metrics["p_on_time"])
        p95 = float(metrics["latency_95th"])

        # Edge count penalty (encourage simpler topologies)
        edge_count = int(np.sum(adj) // 2)  # undirected edges counted once
        edge_fraction = edge_count / max(1, len(edge_list))  # normalize

        # Hard-ish constraint on success (deliverability)
        if p_success < p_success_min:
            # Penalize heavily if too many messages are undeliverable
            cost = big_penalty * (p_success_min - p_success) ** 2
        else:
            cost = 0.0

        # Drive p95 to target
        # Normalize by target_p95 to make it unitless and comparable across scenarios
        cost += w_p95 * ((p95 - target_p95) / max(1e-9, target_p95)) ** 2

        # Drive on-time probability to target
        cost += w_on_time * (p_on_time - target_p_on_time) ** 2

        # Mild preference for fewer edges
        cost += w_edges * edge_fraction

        fit = -float(cost)

        # Track best
        if fit > best_payload["fitness"]:
            best_payload["fitness"] = fit
            best_payload["conn_spec"] = conn_spec
            best_payload["metrics"] = metrics

        eval_cache[key] = fit
        return fit

    # ----------------------------
    # 5) Run pygad GA
    # ----------------------------
    # Gene values are 0/1. pygad supports init_range_low/high, but to force 0/1:
    gene_space = [0, 1]

    ga = pygad.GA(
        num_generations=60,
        num_parents_mating=12,
        fitness_func=fitness_func,
        sol_per_pop=30,
        num_genes=num_genes,
        gene_space=gene_space,
        parent_selection_type="tournament",
        K_tournament=3,
        crossover_type="single_point",
        mutation_type="random",
        mutation_probability=0.05,
        keep_parents=2,
        stop_criteria=["saturate_10"],  # stop if no improvement for 10 gens
        random_seed=42,
        allow_duplicate_genes=True,
    )

    ga.run()

    # Retrieve best
    if best_payload["conn_spec"] is None:
        # Fallback: use GA's best solution
        solution, solution_fitness, _ = ga.best_solution()
        metrics, conn_spec, _ = evaluate_solution(solution)
        best_payload["conn_spec"] = conn_spec
        best_payload["metrics"] = metrics

    # You asked for the connectivity spec dict:
    # (Optionally you may also want to return best_payload["metrics"] for logging.)
    return best_payload["conn_spec"]

def generate_message_samples(mission_specs : dict, 
                             num_samples : int = 20_000,
                             reduced : bool = False
                            ) -> List[Tuple[str, str, float]]:
    # initiate samples list
    samples = []

    # reduce number of samples for testing purposes if reduced flag is set
    num_samples *= 0.1 if reduced else 1.0
    num_samples = int(num_samples)

    # unpack mission specs 
    duration = mission_specs['duration'] * 24 * 3600  # convert from days to seconds
    agent_names = [agent['name'] for agent in mission_specs['spacecraft']] + [gs['name'] for gs in mission_specs['groundOperator']]

    # generate message samples with random sender, receiver, and start time
    for _ in range(num_samples):
        # choose a random receiver and sender from the list of agents
        sender, receiver = random.sample(agent_names, 2)

        # random start time within mission duration
        start_time = duration * random.random()

        # add to samples list
        samples.append((sender,receiver,start_time))

    # return list of message samples sorted by start time
    return sorted(samples, key=lambda x: x[2])

@dataclass
class ConnectivityIndex:
    agents: List[str]
    agent_to_col: Dict[str, int]
    t_start: np.ndarray            # (K,)
    t_end: np.ndarray              # (K,)
    comp_idx: np.ndarray           # (K, N) uint16/uint32: compact component index per agent
    n_comps: np.ndarray            # (K,) number of components in each interval
    trans: List[List[int]]         # (K-1) lists; trans[k][c] is Python-int bitmask of comps in k+1

def build_connectivity_index(scenario_orbitdata: Dict[str, "OrbitData"]) -> ConnectivityIndex:
    """
    Build a compact connectivity index from comms_links that is safe for >64 components:
      - comp_idx[k, a] is the compact component id (0..Ck-1) of agent a in interval k
      - trans[k][c] is a Python-int bitmask over components in interval k+1 that overlap comp c in k
    """
    any_agent = next(iter(scenario_orbitdata))
    orbitdata = scenario_orbitdata[any_agent]

    agent_to_col = orbitdata.comms_target_indices
    col_to_agent = list(orbitdata.comms_target_columns)
    agents = col_to_agent
    N = len(agents)

    # Pull all rows once
    rows = list(orbitdata.comms_links.iter_rows_raw(t=0.0, include_current=True))
    K = len(rows)
    if K == 0:
        return ConnectivityIndex(
            agents=agents,
            agent_to_col=agent_to_col,
            t_start=np.array([], dtype=float),
            t_end=np.array([], dtype=float),
            comp_idx=np.zeros((0, N), dtype=np.uint16),
            n_comps=np.array([], dtype=np.uint16),
            trans=[],
        )

    t_start = np.empty(K, dtype=float)
    t_end = np.empty(K, dtype=float)

    # raw component labels (could be any ints) per interval per agent-col
    raw = np.empty((K, N), dtype=np.int64)
    for k, row in enumerate(rows):
        ts, te, *comp_ids = row
        t_start[k] = ts
        t_end[k] = te
        # make sure this becomes a numeric array, not object
        raw[k, :] = np.asarray(comp_ids, dtype=np.int64)

    # Remap raw labels to compact 0..Ck-1 per interval
    comp_idx = np.empty((K, N), dtype=np.uint16)
    n_comps = np.empty(K, dtype=np.uint16)
    for k in range(K):
        _, inv = np.unique(raw[k, :], return_inverse=True)
        comp_idx[k, :] = inv.astype(np.uint16)
        n_comps[k] = np.uint16(inv.max() + 1)

    # Build transitions as Python ints (NO numpy arrays here)
    trans: List[List[int]] = []
    for k in range(K - 1):
        Ck = int(n_comps[k])
        trans_k: List[int] = [0] * Ck  # Python ints only

        ck = comp_idx[k, :]
        cn = comp_idx[k + 1, :]

        # Each agent maps (comp in k) -> (comp in k+1)
        for a in range(N):
            u = int(ck[a])   # Python int
            v = int(cn[a])   # Python int
            trans_k[u] |= (1 << v)  # Python int bitmask (arbitrary precision)

        # Hard guarantee: Python int and non-negative
        # (If this ever fails, something *else* corrupted it.)
        for c, m in enumerate(trans_k):
            if not isinstance(m, int):
                trans_k[c] = int(m)
            if trans_k[c] < 0:
                raise ValueError(f"Overflow: trans[{k}][{c}] is negative ({trans_k[c]}).")

        trans.append(trans_k)

    return ConnectivityIndex(
        agents=agents,
        agent_to_col=agent_to_col,
        t_start=t_start,
        t_end=t_end,
        comp_idx=comp_idx,
        n_comps=n_comps,
        trans=trans,
    )


def calculate_message_latency_fast(
    sender: str,
    receiver: str,
    start_time: float,
    idx: "ConnectivityIndex",
) -> float:
    """
    Fast earliest-arrival latency for dynamic undirected connectivity with:
      - zero per-hop latency (instant inside component)
      - store-and-forward across time (agents carry the message across interval boundaries)
      - topology changes only at interval boundaries (idx.t_start / idx.comp_idx / idx.trans)

    Returns:
      float: (earliest arrival time - start_time), or np.inf if unreachable.

    IMPORTANT invariants for correctness/performance:
      - idx.trans[k][c] MUST be Python ints (arbitrary-precision), never np.int64.
      - idx.comp_idx is a (K, N) array of compact component indices per interval per agent-col.
    """

    if sender == receiver:
        return 0.0

    # Resolve sender/receiver columns
    s_col: Optional[int] = idx.agent_to_col.get(sender, None)
    r_col: Optional[int] = idx.agent_to_col.get(receiver, None)
    if s_col is None or r_col is None:
        return np.inf

    t_start_arr = idx.t_start
    K = int(len(t_start_arr))
    if K == 0:
        return np.inf

    # Interval index k0 with t_start[k0] <= start_time < t_start[k0+1] (assuming ordered intervals)
    k0 = int(np.searchsorted(t_start_arr, start_time, side="right") - 1)
    if k0 < 0:
        k0 = 0
    if k0 >= K:
        return np.inf

    # Initialize active component bitmask in interval k0 (Python int, avoids overflow)
    s_comp = int(idx.comp_idx[k0, s_col])
    active = int(1) << s_comp

    # Quick check: receiver in same component at k0 -> latency 0
    r_comp = int(idx.comp_idx[k0, r_col])
    if (active >> r_comp) & 1:
        return 0.0

    # Propagate active components forward across boundaries
    # idx.trans has length K-1; trans[k] maps comps at interval k -> bitmask of comps in interval k+1
    for k in range(k0, K - 1):
        trans_k = idx.trans[k]  # List[int] (Python ints)

        # Defensive: ensure 'active' is a non-negative Python int
        active = int(active)
        if active < 0:
            raise ValueError(
                f"Negative active bitmask at interval k={k} (overflow). active={active}. "
                f"Ensure idx.trans contains Python ints, not numpy int64."
            )

        # Compute next active set using bit iteration (no generator overhead)
        x = active
        next_active = 0  # Python int
        while x:
            lsb = x & -x              # lowest set bit
            c = lsb.bit_length() - 1  # component index in interval k
            if c < len(trans_k):
                # Force Python int in case something slipped through
                next_active |= int(trans_k[c])
            x -= lsb  # clears the lowest set bit (safe for positive ints)

        active = int(next_active)
        if active == 0:
            return np.inf

        # Delivered at the start of interval k+1 if receiver's component in k+1 is active
        r_comp_next = int(idx.comp_idx[k + 1, r_col])
        if (active >> r_comp_next) & 1:
            # earliest arrival is boundary time t_start[k+1]
            return float(idx.t_start[k + 1] - start_time)

    return np.inf

# def calculate_message_latency(sender : str, 
#                               receiver : str, 
#                               start_time : float, 
#                               scenario_orbitdata : Dict[str, OrbitData]
#                             ) -> float:
#     """
#     Earliest-arrival latency in a dynamic undirected graph with:
#       - zero per-hop latency (instant within a connected component)
#       - store-and-forward across time (agents carry the message)
#       - topology changes only at interval boundaries (as given by comms_links rows)

#     Returns:
#       latency (float): earliest arrival time - start_time, or np.inf if never reachable.
#     """
#     # check if sender and receiver are the same; if so, latency is zero
#     if sender == receiver: return 0.0

#     # get the sender's orbitdata
#     orbitdata = scenario_orbitdata[sender]

#     # Column index lookup: agent -> column index in the component_indices row
#     agent_to_col = orbitdata.comms_target_indices
#     col_to_agent = orbitdata.comms_target_columns

#     # Message is "held" by any reached agent; starts at the sender
#     reached : set = {sender}

#     # initialize earliest arrival time
#     t_arrival = np.inf

#     # Iterate intervals from start_time onward
#     for t_start, t_end, *component_indices in orbitdata.comms_links.iter_rows_raw(t=start_time,
#                                                                                   include_current=True
#                                                                                  ):

#         # 1) Compile which agents currently contain the message
#         #    i.e., component names of any reached agent.
#         reached_comp_ids = set()
#         for a in reached:
#             col = agent_to_col.get(a, None)
#             if col is None:
#                 continue  # if some agents aren't represented in this table
#             reached_comp_ids.add(component_indices[col])

#         # 2) Expand: everyone in any reached component gets the message instantly
#         #    (including potentially the receiver)
#         new_reached = set()
#         for col, comp_id in enumerate(component_indices):
#             if comp_id in reached_comp_ids:
#                 new_reached.add(col_to_agent[col])

#         if not new_reached.issubset(reached):
#             reached |= new_reached

#         # 3) Check delivery
#         if receiver in reached:
#             # Receiver becomes reachable in this interval.
#             # - If this is the first interval (start_time inside it), arrival is max(t_start, start_time) == start_time.
#             # - If later intervals, arrival is at the boundary t_start.
#             t_arrival = max(t_start, start_time)
#             break

#     return t_arrival - start_time

def sanity_check_trans(idx):
    for k, trans_k in enumerate(idx.trans):
        for c, m in enumerate(trans_k):
            if not isinstance(m, int):
                raise TypeError(f"idx.trans[{k}][{c}] is {type(m)}, expected Python int.")
            if m < 0:
                raise ValueError(f"idx.trans[{k}][{c}] is negative ({m}). Overflow occurred.")
            

def evaluate_scenario_latency(mission_specs: dict,
                              messages: List[Tuple[str, str, float]],
                              T_req: float = np.Inf,
                              printouts: bool = True) -> dict:

    # precompute coverage data for scenario mission specs
    orbitdata_dir = OrbitData.precompute(mission_specs, printouts=printouts)

    # load precomputed coverage data with relevant mission specs 
    scenario_orbitdata = OrbitData.from_directory(orbitdata_dir, mission_specs, printouts=printouts)

    # build connectivity index for fast latency calculations
    idx = build_connectivity_index(scenario_orbitdata) 

    sanity_check_trans(idx)  # catch any overflow issues early

    # intiate list of message latencies
    latencies = np.empty(len(messages), dtype=float)

    for i, (sender, receiver, start_time) in tqdm(
        enumerate(messages),
        total=len(messages),
        desc="Evaluating Message Latencies",
        unit=" msgs",
        disable=not printouts,
        leave=False
    ):
        latencies[i] = calculate_message_latency_fast(sender, receiver, start_time, idx)

    # calculate probability of successful communcations
    p_success = float(np.mean(np.isfinite(latencies)))
    
    # check latency requirement satisfaction (if given)
    p_on_time = float(np.mean(latencies <= T_req)) if np.isfinite(T_req) else p_success

    # create mask for infinite latencies (undelivered messages)
    inf_mask = np.isinf(latencies)

    # define simulation end time as mission duration in seconds
    t_end = mission_specs["duration"] * 24 * 3600
    
    # extract start times aligned with latencies
    start_times = np.fromiter((t0 for _, _, t0 in messages), dtype=float, count=len(messages))
    
    # cap infinte latencies to simulation end time minus the message start time for statistics calculation
    latencies = np.where(inf_mask, t_end - start_times, latencies)

    # return latency statistics (success rate, mean, median, 95th percentile, 99th percentile)
    return {
        "p_success": p_success,
        "p_on_time": p_on_time,
        "latency_mean": float(np.mean(latencies)),
        "latency_median": float(np.median(latencies)),
        "latency_95th": float(np.percentile(latencies, 95)),
        "latency_99th": float(np.percentile(latencies, 99)),
    }

if __name__ == "__main__":
    """
    Creates `connectivity.json` for the connectivity test scenarios
    """
    # set reduced duration flag for testing purposes;
    #  (set to False to generate full duration scenarios)
    reduced = True

    # load trial csv
    trials_filename = "full_factorial_trials_2026-02-22.csv"
    trials_path = os.path.join(".", 'experiments','1_cbba_validation', 'resources', "trials", trials_filename)
    trials_df = pd.read_csv(trials_path)

    # print welcome
    trial_stem = os.path.splitext(os.path.basename(trials_filename))[0]
    print_scenario_banner("CBBA Stress Test Study - {0}".format(trial_stem))

    # base path for experiment
    base_path: str = get_base_path()

    # load templates
    mission_specs_template, ground_operator_specs_template, \
        spacecraft_specs_template, instrument_specs, planner_specs \
            = load_templates(base_path)    

    # duration/step size
    duration = 1.5917 / 24.0 if reduced else 1.0  # [days]
    duration = min(duration, 1.0)
    step_size = 10  # [s]

    # calculate average latency for all trials
    for _, row in trials_df.iterrows():
        # define output dir, ensure it exists
        output_dir = os.path.join(base_path, 'resources', 'connectivity')
        os.makedirs(output_dir, exist_ok=True)

        # unpack row parameters
        trial_id = row['Trial ID']
        preplanner = row['Preplanner']
        replanner = row['Replanner'] 
        num_sats = row['Num Sats']
        latency = row['Latency'].lower()
        task_arrival_rate = row['Task Arrival Rate']
        target_distribution = row['Target Distribution']
        scenario_idx = row['Scenario']

        # define output file paths for connectivity spec and latency results
        low_latency_filename = os.path.join(output_dir, f"nsats-{num_sats}_latency-low.json")
        medium_latency_filename = os.path.join(output_dir, f"nsats-{num_sats}_latency-medium.json")
        high_latency_filename = os.path.join(output_dir, f"nsats-{num_sats}_latency-high.json")

        # check if trial has already been evaluated
        if os.path.exists(low_latency_filename) and os.path.exists(medium_latency_filename) and os.path.exists(high_latency_filename):
            print(f"Latency for trials with num_sats={num_sats} already evaluated; skipping.")
            continue  # if so; skip to next trial
        else:
            print(f"Evaluating Latency for trials with num_sats={num_sats}...")

        # normalize `nan` values to None
        preplanner = "none" if not isinstance(preplanner, str) and pd.isna(preplanner) else preplanner.lower()
        replanner = "none" if not isinstance(replanner, str) and pd.isna(replanner) else replanner.lower()

        # construct scenario mission specs
        generic_mission_specs = generate_scenario_mission_specs(mission_specs_template, duration, step_size,
                                                        base_path, trials_filename, trial_id, 
                                                        preplanner, replanner, num_sats, latency, 
                                                        task_arrival_rate, target_distribution, scenario_idx, 
                                                        spacecraft_specs_template, instrument_specs, planner_specs, 
                                                        ground_operator_specs_template, reduced)

        # define connectivity groups: instruments, relays, groundStations
        imaging_sats = [sat['name'] for sat in generic_mission_specs['spacecraft'] if 'instrument' in sat ]
        relay_sats = [sat['name'] for sat in generic_mission_specs['spacecraft'] if 'instrument' not in sat ]
        ground_stations = [gs['name'] for gs in generic_mission_specs['groundOperator'] ]

        # generate a list of message samples for the scenario to be used in latency evaluation
        messages = generate_message_samples(generic_mission_specs, reduced=reduced)

        # generate connectivity rules for latency scenarios (low, medium, high) 
        # 1) High Latency Scenario: ISLs off, ISL to TDRSS off, SAT to NEN comms on
        high_latency_connectivity_spec = generate_high_latency_connectivity_spec(imaging_sats, relay_sats, ground_stations)
        
        # save high latency scenario connectivity spec to file
        with open(high_latency_filename, 'w') as f:
            json.dump(high_latency_connectivity_spec, f, indent=4)

        # create mission specs for high latency scenario 
        high_latency_mission_specs = generate_scenario_mission_specs(mission_specs_template, duration, step_size,
                                                        base_path, trials_filename, trial_id, 
                                                        preplanner, replanner, num_sats, 'high', 
                                                        task_arrival_rate, target_distribution, scenario_idx, 
                                                        spacecraft_specs_template, instrument_specs, planner_specs, 
                                                        ground_operator_specs_template, reduced)
        
        # evaluate scenario latency
        high_latency_values = evaluate_scenario_latency(high_latency_mission_specs, messages)
        
        # 2) Low Latency Scenario: ISLs on, TDRSS comms on, SAT to NEN comms on
        low_latency_connectivity_spec = generate_low_latency_connectivity_spec(imaging_sats, relay_sats, ground_stations)

        # save low latency scenario connectivity spec to file
        with open(low_latency_filename, 'w') as f:
            json.dump(low_latency_connectivity_spec, f, indent=4)

        # create mission specs for low latency scenario 
        low_latency_mission_specs = generate_scenario_mission_specs(mission_specs_template, duration, step_size,
                                                        base_path, trials_filename, trial_id, 
                                                        preplanner, replanner, num_sats, 'low', 
                                                        task_arrival_rate, target_distribution, scenario_idx, 
                                                        spacecraft_specs_template, instrument_specs, planner_specs, 
                                                        ground_operator_specs_template, reduced)
        
        # evaluate scenario latency
        low_latency_values = evaluate_scenario_latency(low_latency_mission_specs, messages)

        # 3) Medium Latency Scenario: 
        medium_latency_connectivity_spec = generate_medium_latency_connectivity_spec(imaging_sats, relay_sats, ground_stations, 
                                                                                     high_latency_values, low_latency_values, messages, medium_latency_filename,
                                                                                     mission_specs_template, duration, step_size,
                                                                                     base_path, trials_filename, trial_id, 
                                                                                     preplanner, replanner, num_sats, 'medium', 
                                                                                     task_arrival_rate, target_distribution, scenario_idx, 
                                                                                     spacecraft_specs_template, instrument_specs, planner_specs, 
                                                                                     ground_operator_specs_template, reduced
                                                                                    )

        # save medium latency scenario connectivity spec to file
        with open(medium_latency_filename, 'w') as f:
            json.dump(medium_latency_connectivity_spec, f, indent=4)

        # create mission specs for medium latency scenario
        medium_latency_mission_specs = generate_scenario_mission_specs(mission_specs_template, duration, step_size,
                                                        base_path, trials_filename, trial_id, 
                                                        preplanner, replanner, num_sats, 'medium', 
                                                        task_arrival_rate, target_distribution, scenario_idx, 
                                                        spacecraft_specs_template, instrument_specs, planner_specs, 
                                                        ground_operator_specs_template, reduced)
        
        # evaluate scenario latency
        medium_latency_values = evaluate_scenario_latency(medium_latency_mission_specs, messages)

        # print latency results for trial
        print(f"Latency Results for num_sats={num_sats}:")
        low_latency_values['scenario'] = 'low'
        medium_latency_values['scenario'] = 'medium'
        high_latency_values['scenario'] = 'high'
        results_df = print(pd.DataFrame([low_latency_values, medium_latency_values, high_latency_values]))
        print(results_df) 

        x = 1 # breakpoint for debugging       

    print('DONE!')
