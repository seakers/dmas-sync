"""
Constellation Design Script

Generates different satellite constellation designs for comparison between 
centralized and decentralized approaches in satellite communication networks.
"""

import copy
import json
import os

from dmas.utils.constellations import Constellation, WalkerDeltaConstellation

def assign_instrument_to_constellation(
        constellation : Constellation,
        preplanner : str, 
        replanner : str,
        instrument : str,
        instrument_spec : dict, 
        spacecraft_specs_template : dict, 
        planner_specs : dict
    ) -> list:
    # convert to list of orbit states for each satellite in constellation
    constellation_orbit_states = constellation.to_orbital_elements()

    # initialte list to hold satellite specifications for constellation
    constellation_specs = []

    # iterate through constellation parameters to create satellite specifications
    for sat_idx,orbit_state in enumerate(constellation_orbit_states):
        # create satellite specification from template
        sat = copy.deepcopy(spacecraft_specs_template)

        # planner settings
        if preplanner.lower() != 'none':
            sat['planner']['preplanner'] = planner_specs['preplanners'][preplanner.lower()]
        if replanner.lower() != 'none':
            sat['planner']['replanner'] = planner_specs['replanners'][replanner.lower()]

        # remove planner if no preplanner or replanner specified
        if preplanner.lower() == 'none' and replanner.lower() == 'none':
            sat.pop('planner', None)  

        # assign orbit state
        sat['orbitState']['state'] = orbit_state

        # get matching instrument spec for satellite
        instrument_spec = copy.deepcopy(instrument_specs[instrument])

        # assign deep copy of instrument to satellite
        sat['instrument'] = instrument_spec

        # determine satellite name and ID
        sat['@id'] = f"{sat['instrument']['@id']}_{sat_idx}"
        sat['name'] = f"SSO Shell A - {sat['instrument']['name']} Sat {sat['@id']}"

        # add to list of satellite specifications
        constellation_specs.append(sat)

    # reutrn list of satellite specifications for constellation
    return constellation_specs

def generate_commercial(
        preplanner : str, 
        replanner : str, 
        spacecraft_specs_template : dict,
        instrument_specs : dict,
        planner_specs : dict,
    ) -> dict:
    """
    ## Case 1 — Heritage-Based Commercial Constellation
    +------------------+----------------------+------------+----------+-------------+-----------------------------------------------+----------+------------+--------------+-----------------+--------------------------------+
    | Sub-constellation| Instrument           | Orbit type | Altitude | Inclination | LTAN / ECT                                    | N planes | Sats/plane | N sats total | Off-nadir limit | Platform basis                 |
    +------------------+----------------------+------------+----------+-------------+-----------------------------------------------+----------+------------+--------------+-----------------+--------------------------------+
    | SSO shell A      | VNIR (Dove-class)    | SSO        | 500 km   | 97°         | 09:30-11:30 asc.                              | 2        | 24         | 48           | ±60°            | Planet SuperDove PSB.SD        |
    | SSO shell B      | TIR (HiVE-class)     | SSO        | 510 km   | 97.5°       | 10:30 asc. (SkyBee-1) / 13:30 asc. (SkyBee-2) | 2        | 5          | 10           | ±30°            | constellr SkyBee / HiVE        |
    | SSO shell C      | MIR (OroraTech)      | SSO        | 550 km   | 97°         | ~13:30-14:00 desc.                            | 1        | 8          | 8            | ±45°            | OroraTech OTC-P1 / Spire LEMUR |
    | MIO overlay A    | VNIR-HR (Pelican)    | MIO        | 475 km   | 53°         | N/A                                           | 2        | 3          | 6            | ±60°            | Planet Pelican Gen-1           |
    | MIO overlay B    | SAR (Capella-class)  | MIO        | 525 km   | 53°         | N/A                                           | 3        | 6          | 18           | ±45°            | Capella Acadia (X-band)        |
    | SSO shell D      | Altimeter (SMASH)    | SSO        | 550 km   | 97°         | N/A                                           | 1        | 10         | 10           | ±0° (nadir)     | SMASH / BWI REVALTO concept    |
    | TOTAL            |                      |            |          |             |                                               | 11       |            | 100          |                 |                                |
    +------------------+----------------------+------------+----------+-------------+-----------------------------------------------+----------+------------+--------------+-----------------+--------------------------------+

    Notes:
        - RAAN distribution for SSO shells follows natural rideshare injection. Relative RAAN between
        SSO and MIO planes drifts with time due to differential nodal precession rates:
            SSO precesses at ~+0.9856 deg/day
            MIO (53 deg) precesses at ~-2.06 deg/day
        This provides natural geometric diversity across simulation dates.
        - The altimeter sub-constellation is nadir-fixed by physical design constraint (pulse-limited
        ranging requires nadir pointing). All other sub-constellations are steerable within their
        respective off-nadir limits.
        - Satellite counts are scaled from operational deployments to meet the 100-satellite study
        budget while preserving the relative proportions and orbital geometries of the reference
        missions, following the approach of [reference paper].
    """

    # SSO Shell A
    sso_shell_a = WalkerDeltaConstellation(
        alt=500,
        inc=97.0,
        num_sats=48,
        num_planes=2,
        phasing_param=1,
        raan_offset=298.48
    )
    sso_shell_a_specs = assign_instrument_to_constellation(
        constellation=sso_shell_a,
        preplanner=preplanner,
        replanner=replanner,
        instrument="VNIR_agile",
        instrument_spec=instrument_specs,
        spacecraft_specs_template=spacecraft_specs_template,
        planner_specs=planner_specs
    )
    assert all('Passive Optical Scanner' in sat['instrument']['@type'] for sat in sso_shell_a_specs), \
        "Error: Not all satellites in SSO Shell A were assigned the VNIR instrument."
    
    # SSO Shell B
    sso_shell_b = WalkerDeltaConstellation(
        alt=510,
        inc=97.5,
        num_sats=10,
        num_planes=2,
        phasing_param=1,
        raan_offset=305.98
    )

    sso_shell_b_specs = assign_instrument_to_constellation(
        constellation=sso_shell_b,
        preplanner=preplanner,
        replanner=replanner,
        instrument="TIR",
        instrument_spec=instrument_specs,
        spacecraft_specs_template=spacecraft_specs_template,
        planner_specs=planner_specs
    )
    assert all('tir' in sat['instrument']['@id'] for sat in sso_shell_b_specs), \
        "Error: Not all satellites in SSO Shell B were assigned the TIR instrument."
    

    # SSO Shell C
    sso_shell_c = WalkerDeltaConstellation(
        alt=550,
        inc=97.0,
        num_sats=8,
        num_planes=1,
        phasing_param=1,
        raan_offset=354.73
    )
    sso_shell_c_specs = assign_instrument_to_constellation(
        constellation=sso_shell_c,
        preplanner=preplanner,
        replanner=replanner,
        instrument="MIR",
        instrument_spec=instrument_specs,
        spacecraft_specs_template=spacecraft_specs_template,
        planner_specs=planner_specs
    )

    # MIO Overlay A
    mio_overlay_a = WalkerDeltaConstellation(
        alt=475,
        inc=53.0,
        num_sats=6,
        num_planes=2,
        phasing_param=1,
        raan_offset=0.0
    )
    mio_overlay_a_specs = assign_instrument_to_constellation(
        constellation=mio_overlay_a,
        preplanner=preplanner,
        replanner=replanner,
        instrument="VNIR_agile",
        instrument_spec=instrument_specs,
        spacecraft_specs_template=spacecraft_specs_template,
        planner_specs=planner_specs
    )

    # MIO Overlay B
    mio_overlay_b = WalkerDeltaConstellation(
        alt=525,
        inc=53.0,
        num_sats=18,
        num_planes=3,
        phasing_param=1,
        raan_offset=0.0
    )
    mio_overlay_b_specs = assign_instrument_to_constellation(
        constellation=mio_overlay_b,
        preplanner=preplanner,
        replanner=replanner,
        instrument="SAR",
        instrument_spec=instrument_specs,
        spacecraft_specs_template=spacecraft_specs_template,
        planner_specs=planner_specs
    )

    # SSO Shell D
    sso_shell_d = WalkerDeltaConstellation(
        alt=550,
        inc=97.0,
        num_sats=10,
        num_planes=1,
        phasing_param=1,
        raan_offset=305.98
    )
    sso_shell_d_specs = assign_instrument_to_constellation(
        constellation=sso_shell_d,
        preplanner=preplanner,
        replanner=replanner,
        instrument="Altimeter",
        instrument_spec=instrument_specs,
        spacecraft_specs_template=spacecraft_specs_template,
        planner_specs=planner_specs
    )

    # compile constellation design into dictionary format
    constellation_design = []
    for sat_specs in [sso_shell_a_specs, sso_shell_b_specs, sso_shell_c_specs, mio_overlay_a_specs, mio_overlay_b_specs, sso_shell_d_specs]:
        constellation_design.extend(sat_specs)

    # return constellation design
    return constellation_design

def generate_walker_delta(
        instrument_specs : dict
    ) -> dict: 
    """
    ## Case 3 — Walker-Delta Constellation

    +---------------------+------------------------+------------+----------------+----------+-------------+------------+--------------+-------------+------------+---------------------+--------------+-----------------+------------------------------------------+
    | Mission             | Primary mission        | Instrument | Walker (T/P/F) | Altitude | Inclination | Orbit type | LTAN         | RAAN offset | Sats/plane | N sats (instrument) | N sats total | Off-nadir limit | Instrument arrangement                   |
    +---------------------+------------------------+------------+----------------+----------+-------------+------------+--------------+-------------+------------+---------------------+--------------+-----------------+------------------------------------------+
    | Water quality       | Algal bloom monitoring | VNIR       | 36/6/1         | 500 km   | 97°         | SSO        | 10:30 asc.   | 0°          | 6          | 18                  | 36           | ±60°            | Pattern 2: alternating VNIR-ALT per plane|
    | Water quality       | Algal bloom monitoring | Altimeter  | 36/6/1         | 500 km   | 97°         | SSO        | 10:30 asc.   | 0°          | 6          | 18                  | —            | ±0° (nadir)     | Pattern 2: alternating VNIR-ALT per plane|
    | Flood monitoring    | High-flow monitoring   | SAR        | 30/5/2         | 550 km   | 53°         | MIO        | N/A          | 36°         | 6          | 15                  | 30           | ±45°            | Pattern 2: alternating SAR-ALT per plane |
    | Flood monitoring    | High-flow monitoring   | Altimeter  | 30/5/2         | 550 km   | 53°         | MIO        | N/A          | 36°         | 6          | 15                  | —            | ±0° (nadir)     | Pattern 2: alternating SAR-ALT per plane |
    | Fire monitoring     | Wildfire monitoring    | MIR        | 36/3/1         | 550 km   | 97°         | SSO        | 13:30 asc.   | 60°         | 12         | 12                  | 36           | ±45°            | Pattern 2: cycling MIR-SAR-VNIR per plane|
    | Fire monitoring     | Wildfire monitoring    | SAR        | 36/3/1         | 550 km   | 97°         | SSO        | 13:30 asc.   | 60°         | 12         | 12                  | —            | ±45°            | Pattern 2: cycling MIR-SAR-VNIR per plane|
    | Fire monitoring     | Wildfire monitoring    | VNIR       | 36/3/1         | 550 km   | 97°         | SSO        | 13:30 asc.   | 60°         | 12         | 12                  | —            | ±60°            | Pattern 2: cycling MIR-SAR-VNIR per plane|
    | TOTAL               |                        |            |                |          |             |            |              |             |            |                     | 102          |                 |                                          |
    +---------------------+------------------------+------------+----------------+----------+-------------+------------+--------------+-------------+------------+---------------------+--------------+-----------------+------------------------------------------+

    Notes:
        - Walker notation T/P/F: total satellites / number of planes / phasing factor.
        Phasing factor F determines the argument of latitude offset between adjacent planes:
            delta_u = F * 360 / T  (degrees)
        - RAAN plane spacing within each constellation:
            delta_RAAN = 360 / P
            Water quality (P=6): delta_RAAN = 60 deg
            Flood         (P=5): delta_RAAN = 72 deg
            Fire          (P=3): delta_RAAN = 120 deg
        - RAAN base offsets between constellations (0, 36, 60 deg) distribute the 14 total planes
        with maximum longitudinal separation between the two SSO constellations (water quality
        and fire, offset by 60 deg) and the MIO flood constellation (offset 36 deg from water
        quality). Exact optimization should be validated with a coverage analysis tool
        (e.g., STK, GMAT, or Orekit).
        - Pattern 2 instrument arrangement: instruments alternate or cycle within each plane in
        argument-of-latitude order. Within-plane consecutive instrument gap:
            6  sats/plane -> ~15.8 min between consecutive different instruments
            12 sats/plane -> ~7.9  min between consecutive different instruments
        - The water quality and flood constellations each share one Walker shell between two
        instrument types (VNIR+ALT and SAR+ALT respectively). Instrument assignment is by
        position within the plane, not by separate orbital shells.
        - The fire constellation cycles three instrument types (MIR-SAR-VNIR) with 4 satellites
        of each type per plane.
        - Total satellite count is 102. To reduce to exactly 100, adjust the flood constellation
        to 28/4/2 (7 sats/plane, 14 per instrument type, 28 total):
            36 (water quality) + 28 (flood) + 36 (fire) = 100 satellites.
    """
    raise NotImplementedError("Walker-Delta constellation design not implemented yet.")

if __name__ == "__main__":
    """ TEST SCRIPT FOR GENERATING CONSTELLATION DESIGNS """
    # set params
    preplanner = "DP"
    replanner = "CBBA"

    # define template resources path 
    resources_path = os.path.join('.', 'experiments','2_centralized_vs_decentralized', 'resources', "templates")

    # define spacecraft specs template path
    spacecraft_specs_template_path = os.path.join(resources_path, "spacecraft.json")

    # load spacecraft specs template json
    with open(spacecraft_specs_template_path, "r") as f:
        spacecraft_specs_template = json.load(f)

    # define instrument specs path
    instrument_specs_path = os.path.join(resources_path, "instruments.json")
        
    # load instrument specs json
    with open(instrument_specs_path, "r") as f:
        instrument_specs = json.load(f)

    # define planner specs path
    planner_specs_path = os.path.join(resources_path, "planners.json")

    # load planner specs json
    with open(planner_specs_path, "r") as f:
        planner_specs = json.load(f)
    
    # generate commercial constellation design
    commercial_constellation : dict = generate_commercial(preplanner, replanner, spacecraft_specs_template, instrument_specs, planner_specs)

    # generate walker delta constellation design    
    walker_delta_constellation : dict = generate_walker_delta(instrument_specs)
    
    # define output path for constellation designs
    out_dir = os.path.join('.', 'experiments','2_centralized_vs_decentralized', 'resources', 'constellations')
    
    # ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # save constellation designs to json
    commercial_out_path = os.path.join(out_dir, "commercial_constellation.json")
    with open(commercial_out_path, "w") as f:
        json.dump(commercial_constellation, f, indent=4)

    walker_delta_out_path = os.path.join(out_dir, "walker_delta_constellation.json")
    with open(walker_delta_out_path, "w") as f:
        json.dump(walker_delta_constellation, f, indent=4)

    # diagnostic print
    print(f"Commercial constellation design saved to: \n\t`{commercial_out_path}`")
    print(f"Walker delta constellation design saved to: \n\t`{walker_delta_out_path}`")

    # TODO print summary of constellation designs
    # print("\nSummary of Constellation Designs:")
    
    # `done` message
    print('DONE')
    

    