"""
Generate constellation connectivity specifications
"""

from collections import defaultdict
import json
import os
from typing import List

from dmas.utils.tools import print_scenario_banner

GS_NETWORK = "NEN"
GS_ANNOUNCER = "GS Announcer"
GS_PLANNER = "GS Planner"

def generate_gs_connectivity_specs(constellation_spec : List[dict], connectivity_template : dict) -> dict:
    # copy connectivity template
    connectivity_spec = connectivity_template.copy()

    # remove overrides 
    connectivity_spec['overrides'] = []

    # define groups
    instruments_group = [
        sat_spec['name'] for sat_spec in constellation_spec
        if "tdrss" not in sat_spec['name'].lower() # exclude TDRSS from instrument group
    ]
    relay_group = [
        sat_spec['name'] for sat_spec in constellation_spec
        if "tdrss" in sat_spec['name'].lower() # include only TDRSS in relay group
    ]  
    ground_station_group = [GS_ANNOUNCER, GS_PLANNER]

    # add groups to connectivity spec
    connectivity_spec['groups'] = {
        "instruments": instruments_group,
        "relays": relay_group,
        "ground_stations": ground_station_group
    }

    # define connectivity rules
    gs_only_connectivity = {
        "action" : "allow",
        "scope" : "between",
        "targets" : ["instruments", "ground_stations"]
    }
    no_intrasat_connectivity = {
        "action": "deny",
        "scope": "within",
        "targets": "instruments"
    }
    no_intrarelay_connectivity = {
        "action": "deny",
        "scope": "within",
        "targets": "relays"
    }

    # set connectivity rules in spec
    connectivity_spec['rules'] = [
        gs_only_connectivity,
        no_intrasat_connectivity,
        no_intrarelay_connectivity
    ]

    # return connectivity specification
    return connectivity_spec

def generate_commercial_intraconstellation_connectivity_specs(constellation_spec : List[dict], connectivity_template : dict) -> dict:
    # copy connectivity template
    connectivity_spec = connectivity_template.copy()

    # remove overrides 
    connectivity_spec['overrides'] = []

    # define groups by constellation
    groups = defaultdict(list)
    for sat_spec in constellation_spec:
        if "planet" in sat_spec['name'].lower():
            groups["planet_labs"].append(sat_spec['name'])
        elif "constellr" in sat_spec['name'].lower():
            groups["constellr"].append(sat_spec['name'])
        elif "capella" in sat_spec['name'].lower():
            groups["capella"].append(sat_spec['name'])
        elif "ororatech" in sat_spec['name'].lower():
            groups["ororatech"].append(sat_spec['name'])
        elif "cnes" in sat_spec['name'].lower():
            groups["cnes"].append(sat_spec['name'])
        elif "tdrss" in sat_spec['name'].lower():
            groups["relays"].append(sat_spec['name'])
        else:
            raise ValueError(f"Satellite {sat_spec['name']} does not match any known constellation group")
    groups['ground_stations'] = [GS_ANNOUNCER, GS_PLANNER]

    # add groups to connectivity spec
    connectivity_spec['groups'] = dict(groups)

    # define connectivity rules
    planet_intraconstellation_connectivity = {
        "action": "allow",
        "scope": "within",
        "targets": "planet_labs",
    }
    constellr_intraconstellation_connectivity = {
        "action": "allow",
        "scope": "within",
        "targets": "constellr",
    }
    capella_intraconstellation_connectivity = {
        "action": "allow",
        "scope": "within",
        "targets": "capella",       
    }
    ororatech_intraconstellation_connectivity = {
        "action": "allow",
        "scope": "within",
        "targets": "ororatech",
    }
    cnes_intraconstellation_connectivity = {
        "action": "allow",
        "scope": "within",
        "targets": "cnes",
    }
    intrarelay_connectivity = {
        "action": "allow",
        "scope": "within",
        "targets": "relays"
    }
    
    planet_gs_connectivity = {
        "action" : "allow",
        "scope" : "between",
        "targets" : ["planet_labs", "ground_stations"]
    }
    constellr_gs_connectivity = {
        "action" : "allow",
        "scope" : "between",
        "targets" : ["constellr", "ground_stations"]
    }
    capella_gs_connectivity = {
        "action" : "allow", 
        "scope" : "between",
        "targets" : ["capella", "ground_stations"]
    }
    ororatech_gs_connectivity = {
        "action" : "allow",
        "scope" : "between",
        "targets" : ["ororatech", "ground_stations"]
    }
    cnes_gs_connectivity = {
        "action" : "allow",
        "scope" : "between",
        "targets" : ["cnes", "ground_stations"]
    }
    tdrss_gs_connectivity = {
        "action" : "allow",
        "scope" : "between",
        "targets" : ["relays", "ground_stations"]
    }

    # set connectivity rules in spec
    connectivity_spec['rules'] = [
        planet_intraconstellation_connectivity,
        constellr_intraconstellation_connectivity,
        capella_intraconstellation_connectivity,
        ororatech_intraconstellation_connectivity,
        cnes_intraconstellation_connectivity,
        intrarelay_connectivity,
        planet_gs_connectivity,
        constellr_gs_connectivity,
        capella_gs_connectivity,
        ororatech_gs_connectivity,
        cnes_gs_connectivity,
        tdrss_gs_connectivity
    ]

    # return connectivity specification
    return connectivity_spec

def generate_walker_intraconstellation_connectivity_specs(constellation_spec : List[dict], connectivity_template : dict) -> dict:
    # copy connectivity template
    connectivity_spec = connectivity_template.copy()

    # remove overrides 
    connectivity_spec['overrides'] = []

    # define groups by constellation
    groups = defaultdict(list)
    for sat_spec in constellation_spec:
        if "tdrss" in sat_spec['name'].lower():
            groups["relays"].append(sat_spec['name'])
        elif "algal" in sat_spec['name'].lower():
            groups["algal_bloom_monitoring"].append(sat_spec['name'])
        elif "flood" in sat_spec['name'].lower():
            groups["flood_monitoring"].append(sat_spec['name'])
        elif "wildfire" in sat_spec['name'].lower():
            groups["wildfire_monitoring"].append(sat_spec['name'])
        else:
            raise ValueError(f"Satellite {sat_spec['name']} does not match any known constellation group")
    groups['ground_stations'] = [GS_ANNOUNCER, GS_PLANNER]

    # add groups to connectivity spec
    connectivity_spec['groups'] = dict(groups)

    # define connectivity rules
    intra_tdrss_connectivity = {
        "action": "allow",
        "scope": "within",
        "targets": "relays"
    }
    intra_algal_bloom_monitoring_connectivity = {
        "action": "allow",
        "scope": "within",
        "targets": "algal_bloom_monitoring"
    }
    intra_flood_monitoring_connectivity = {
        "action": "allow",
        "scope": "within",
        "targets": "flood_monitoring"
    }
    intra_wildfire_monitoring_connectivity = {
        "action": "allow",
        "scope": "within",
        "targets": "wildfire_monitoring"
    }

    tdrss_gs_connectivity = {
        "action" : "allow",
        "scope" : "between",
        "targets" : ["relays", "ground_stations"]
    }
    algal_bloom_monitoring_gs_connectivity = {
        "action" : "allow",
        "scope" : "between",
        "targets" : ["algal_bloom_monitoring", "ground_stations"]
    }
    flood_monitoring_gs_connectivity = {
        "action" : "allow",
        "scope" : "between",
        "targets" : ["flood_monitoring", "ground_stations"]
    }
    wildfire_monitoring_gs_connectivity = {
        "action" : "allow",
        "scope" : "between",
        "targets" : ["wildfire_monitoring", "ground_stations"]
    }

    # set connectivity rules in spec
    connectivity_spec['rules'] = [
        intra_tdrss_connectivity,
        intra_algal_bloom_monitoring_connectivity,
        intra_flood_monitoring_connectivity,
        intra_wildfire_monitoring_connectivity,
        tdrss_gs_connectivity,
        algal_bloom_monitoring_gs_connectivity,
        flood_monitoring_gs_connectivity,
        wildfire_monitoring_gs_connectivity
    ]

    # return connectivity specification
    return connectivity_spec

def generate_interconstellation_connectivity_specs(constellation_spec : List[dict], connectivity_template : dict) -> dict:
    # copy connectivity template
    connectivity_spec = connectivity_template.copy()

    # remove overrides 
    connectivity_spec['overrides'] = []

    # define groups
    instruments_group = [
        sat_spec['name'] for sat_spec in constellation_spec
        if "tdrss" not in sat_spec['name'].lower() # exclude TDRSS from instrument group
    ]
    relay_group = [
        sat_spec['name'] for sat_spec in constellation_spec
        if "tdrss" in sat_spec['name'].lower() # include only TDRSS in relay group
    ]  
    ground_station_group = [GS_ANNOUNCER, GS_PLANNER]

    # add groups to connectivity spec
    connectivity_spec['groups'] = {
        "instruments": instruments_group,
        "relays": relay_group,
        "ground_stations": ground_station_group
    }

    # define connectivity rules
    interconstellation_connectivity = {
        "action" : "allow",
        "scope" : "between",
        "targets" : ["instruments", "relays", "ground_stations"]
    }
    intraconstellation_connectivity = {
        "action": "allow",
        "scope": "within",
        "targets": "instruments"
    }
    intrarelay_connectivity = {
        "action": "allow",
        "scope": "within",
        "targets": "relays"
    }

    # set connectivity rules in spec
    connectivity_spec['rules'] = [
        interconstellation_connectivity,
        intraconstellation_connectivity,
        intrarelay_connectivity
    ]

    # return connectivity specification
    return connectivity_spec

def save_connectivity_spec(commercial_connectivity : dict, walker_delta_connectivity : dict, connectivit_level : str):
    commercial_output_path = os.path.join(output_path, f"commercial_{connectivit_level.lower()}.json")
    walker_delta_output_path = os.path.join(output_path, f"walker-delta_{connectivit_level.lower()}.json")

    with open(commercial_output_path, 'w') as f:
        json.dump(commercial_connectivity, f, indent=4)
    with open(walker_delta_output_path, 'w') as f:
        json.dump(walker_delta_connectivity, f, indent=4)

    return commercial_output_path, walker_delta_output_path

if __name__ == "__main__":
    # print welcome
    print_scenario_banner('Connectivity Specification Generator for Planner Comparative Study')
    
    # define template resources path 
    resources_path = os.path.join('.', 'experiments','2_centralized_vs_decentralized', 'resources')
    constellations_path = os.path.join(resources_path, "constellations")
    templates_path = os.path.join(resources_path, "templates")

    # load constellations
    commercial_constellation_path = os.path.join(constellations_path, "commercial_constellation.json")
    walker_delta_constellation_path = os.path.join(constellations_path, "walker_delta_constellation.json")
    with open(commercial_constellation_path, 'r') as f:
        commercial_constellation = json.load(f)
    with open(walker_delta_constellation_path, 'r') as f:
        walker_delta_constellation = json.load(f)

    # load connectivity template
    connectivity_template_path = os.path.join(templates_path, "connectivity.json")
    with open(connectivity_template_path, 'r') as f:
        connectivity_template = json.load(f)

    # define output path for connectivity specifications
    output_path = os.path.join(resources_path, "connectivity")

    # create directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # CASE 1: GS-only connectivity 
    # generate connectivity specifications for each constellation
    commercial_connectivity_gs = generate_gs_connectivity_specs(commercial_constellation, connectivity_template)
    walker_delta_connectivity_gs = generate_gs_connectivity_specs(walker_delta_constellation, connectivity_template)

    # save connectivity specifications to output directory
    save_connectivity_spec(commercial_connectivity_gs, walker_delta_connectivity_gs, connectivit_level="GS")

    # CASE 2: Intra-constellation connectivity 
    # generate connectivity specifications for each constellation    
    commercial_connectivity_intraconstellation = generate_commercial_intraconstellation_connectivity_specs(commercial_constellation, connectivity_template)
    walker_delta_connectivity_intraconstellation = generate_walker_intraconstellation_connectivity_specs(walker_delta_constellation, connectivity_template)

    # save connectivity specifications to output directory
    save_connectivity_spec(commercial_connectivity_intraconstellation, walker_delta_connectivity_intraconstellation, connectivit_level="Intraconstellation")

    # CASE 3: Inter-constellation connectivity
    # generate connectivity specifications for each constellation   
    commercial_connnectivity_interconstellation = generate_interconstellation_connectivity_specs(commercial_constellation, connectivity_template)
    walker_delta_connectivity_interconstellation = generate_interconstellation_connectivity_specs(walker_delta_constellation, connectivity_template)

    # save connectivity specifications to output directory
    save_connectivity_spec(commercial_connnectivity_interconstellation, walker_delta_connectivity_interconstellation, connectivit_level="Interconstellation")

    # print completion message
    print("DONE")