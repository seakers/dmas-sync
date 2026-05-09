"""
Constellation Design Script

Generates different satellite constellation designs for comparison between 
centralized and decentralized approaches in satellite communication networks.
"""

import copy
import json
import os
from typing import List

from dmas.utils.constellations import Constellation, WalkerDeltaConstellation

def assign_instruments_to_constellation(
        constellation : Constellation,
        const_name : str,
        const_id : str,
        instruments : List[str],
        instrument_specs : dict, 
        spacecraft_specs_template : dict
    ) -> List[dict]:
    # convert to list of orbit states for each satellite in constellation
    constellation_orbit_states = constellation.to_orbital_elements()

    # initialte list to hold satellite specifications for constellation
    constellation_specs = []

    # iterate through constellation parameters to create satellite specifications
    for sat_idx,orbit_state in enumerate(constellation_orbit_states):
        # create satellite specification from template
        sat = copy.deepcopy(spacecraft_specs_template)

        # assign orbit state
        sat['orbitState']['state'] = orbit_state

        # select instrument 
        instrument_idx = sat_idx % len(instruments)
        instrument = instruments[instrument_idx]

        # get matching instrument spec for satellite
        # instrument_spec = copy.deepcopy(instrument_specs[instrument])
        instrument_spec = instrument_specs[instrument].copy()

        # assign deep copy of instrument to satellite
        sat['instrument'] = instrument_spec

        # determine satellite name and ID
        n_instr = sat_idx // len(instruments)
        sat['@id'] = f"{const_id}_{sat['instrument']['@id']}_{n_instr}"
        sat['name'] = f"{const_name} - {sat['instrument']['name']} Sat {n_instr}"

        # add to list of satellite specifications
        constellation_specs.append(sat)

    # return list of satellite specifications for constellation
    return constellation_specs 

def assign_single_instrument_to_constellation(
        constellation : Constellation,
        name : str,
        const_id : str,
        instrument : str,
        instrument_specs : dict, 
        spacecraft_specs_template : dict
    ) -> List[dict]:
    return assign_instruments_to_constellation(
        constellation=constellation,
        const_name=name,
        const_id=const_id,
        instruments=[instrument],
        instrument_specs=instrument_specs,
        spacecraft_specs_template=spacecraft_specs_template
    )

def assign_mission_to_constellation(constellation_specs : List[dict], mission_name : str) -> List[dict]:
    for sat in constellation_specs:
        sat['mission'] = mission_name
    return constellation_specs

def generate_tdrss(spacecraft_specs_template : dict) -> List[dict]:
    """ TRDSS satellites to the scenario for increased connectivity """
    # define GEO constellation parameters
    geo_alt = 35_786.0  # [km]
    geo_inc = 0.0     # [deg]
    geo_num_sats = 3  # number of GEO satellites
    geo_num_planes = 1  # number of planes in GEO constellation
    geo_phasing_factor = 0  # phasing factor for GEO constellation

    # create GEO constellation instance
    geo_constellation = WalkerDeltaConstellation(geo_alt, geo_inc, geo_num_sats, geo_num_planes, geo_phasing_factor)
    
    # extract orbital elements
    geo_orbital_elements : List[dict] = geo_constellation.to_orbital_elements()

    # initiate list to hold satellite specifications for constellation
    satellite_specifications = []

    # create satellite specifications list
    for geo_sat_idx,geo_orbit_state in enumerate(geo_orbital_elements):
        # create satellite specification from template
        geo_satellite_spec : dict = copy.deepcopy(spacecraft_specs_template)

        # do not assign a planner to the GEO satellites; they are just relays in this scenario
        geo_satellite_spec.pop('planner', None)

        # assign orbit state
        geo_satellite_spec['orbitState']['state'] = geo_orbit_state

        # remove instrument; GEO satellites are just relays in this scenario
        geo_satellite_spec.pop('instrument', None)

        # define satellite name and ID
        geo_satellite_spec['name'] = f"tdrss_sat_{geo_sat_idx}"
        geo_satellite_spec['@id'] = f"tdrss_{geo_sat_idx}"

        # add to list of satellite specifications
        satellite_specifications.append(geo_satellite_spec)

    # return list of satellite specifications for constellation
    return satellite_specifications 

def generate_commercial(
        spacecraft_specs_template : dict,
        instrument_specs : dict,
        mission_type : str,
    ) -> List[dict]:
    """
    ## Case 1 — Heritage-Based Commercial Constellation
    +------------------+----------------------+------------+----------+-------------+-----------------------------------------------+----------+------------+--------------+-----------------+--------------------------------+
    | Sub-constellation| Instrument           | Orbit type | Altitude | Inclination | LTAN / ECT                                    | N planes | Sats/plane | N sats total | Off-nadir limit | Platform basis                 |
    +------------------+----------------------+------------+----------+-------------+-----------------------------------------------+----------+------------+--------------+-----------------+--------------------------------+
    | SSO shell A      | VNIR (Dove-class)    | SSO        | 500 km   | 97°         | 09:30-11:30 asc.                              | 2        | 24         | 48           | ±60°            | Planet SuperDove PSB.SD        |
    | MIO overlay A    | VNIR-HR (Pelican)    | MIO        | 475 km   | 53°         | N/A                                           | 2        | 3          | 6            | ±60°            | Planet Pelican Gen-1           |
    | SSO shell B      | TIR (HiVE-class)     | SSO        | 510 km   | 97.5°       | 10:30 asc. (SkyBee-1) / 13:30 asc. (SkyBee-2) | 2        | 5          | 10           | ±30°            | constellr SkyBee / HiVE        |
    | MIO overlay B    | SAR (Capella-class)  | MIO        | 525 km   | 53°         | N/A                                           | 3        | 6          | 18           | ±45°            | Capella Acadia (X-band)        |
    | SSO shell C      | MIR (OroraTech)      | SSO        | 550 km   | 97°         | ~13:30-14:00 desc.                            | 1        | 8          | 8            | ±45°            | OroraTech OTC-P1 / Spire LEMUR |
    | SSO shell D      | Altimeter (SMASH)    | SSO        | 550 km   | 97°         | N/A                                           | 1        | 10         | 10           | ±0° (nadir)     | SMASH / BWI REVALTO concept    |
    | TOTAL            |                      |            |          |             |                                               | 11       |            | 100          |                 |                                |
    +------------------+----------------------+------------+----------+-------------+-----------------------------------------------+----------+------------+--------------+-----------------+--------------------------------+
    
    General notes:
        - All satellite counts are scaled from operational deployments to meet
        the 100-satellite study budget while preserving relative proportions
        and orbital geometries of the reference missions, following the approach
        of [reference paper].
        - The reference paper models the Planet constellation as 2 SSO planes x 95
        satellites + 2 MIO planes x 5 satellites, and the Walker constellation as
        6 planes x 14 satellites + 2 planes x 12 satellites at 88 deg and 51.6
        deg inclinations respectively. Our Case 1 architecture follows the same
        SSO-primary + MIO-overlay pattern; our Case 3 Walker architecture follows
        the uniform Walker-delta pattern of that paper's Walker constellation.
        - Parameters marked as estimated (off-nadir limits for OroraTech, Pelican
        MIO inclination, SMASH altitude) are not publicly disclosed by the
        respective operators and have been derived from analogous systems or
        from the physical/operational constraints described above.
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
    sso_shell_a_specs = assign_single_instrument_to_constellation(
        constellation=sso_shell_a,
        name="Planet Labs - Dove",
        const_id="sso_vnir",
        instrument="VNIR_agile",
        instrument_specs=instrument_specs,
        spacecraft_specs_template=spacecraft_specs_template
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
    mio_overlay_a_specs = assign_single_instrument_to_constellation(
        constellation=mio_overlay_a,
        name="Planet Labs - Pelican",
        const_id="mio_vnir",
        instrument="VNIR_agile",
        instrument_specs=instrument_specs,
        spacecraft_specs_template=spacecraft_specs_template,
    )

    # SSO Shell B
    sso_shell_b = WalkerDeltaConstellation(
        alt=510,
        inc=97.5,
        num_sats=10,
        num_planes=2,
        phasing_param=1,
        raan_offset=305.98
    )
    sso_shell_b_specs = assign_single_instrument_to_constellation(
        constellation=sso_shell_b,
        name="constellr - HiVe",
        const_id="sso_tir",
        instrument="TIR",
        instrument_specs=instrument_specs,
        spacecraft_specs_template=spacecraft_specs_template,
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
    mio_overlay_b_specs = assign_single_instrument_to_constellation(
        constellation=mio_overlay_b,
        name="Capella - Acadia X-SAR",
        const_id="mio_sar",
        instrument="SAR",
        instrument_specs=instrument_specs,
        spacecraft_specs_template=spacecraft_specs_template,
    )

    # SSO Shell C
    sso_shell_c = WalkerDeltaConstellation(
        alt=550,
        inc=97.0,
        num_sats=8,
        num_planes=1,
        phasing_param=1,
        raan_offset=354.73
    )
    sso_shell_c_specs = assign_single_instrument_to_constellation(
        constellation=sso_shell_c,
        name="OroraTech - Wildfire",
        const_id="sso_mir",
        instrument="MIR",
        instrument_specs=instrument_specs,
        spacecraft_specs_template=spacecraft_specs_template,
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
    sso_shell_d_specs = assign_single_instrument_to_constellation(
        constellation=sso_shell_d,
        name="CNES - SMASH",
        const_id="sso_alt",
        instrument="Altimeter",
        instrument_specs=instrument_specs,
        spacecraft_specs_template=spacecraft_specs_template,
    )

    # TDRSS 
    tdrss_specs = generate_tdrss(spacecraft_specs_template)

    # compile constellation design into dictionary format
    constellation_design = []
    for sat_specs in [sso_shell_a_specs, sso_shell_b_specs, sso_shell_c_specs, mio_overlay_a_specs, mio_overlay_b_specs, sso_shell_d_specs, tdrss_specs]:
        constellation_design.extend(sat_specs)

    # return constellation design
    return constellation_design

def generate_walker_delta(
        spacecraft_specs_template : dict,
        instrument_specs : dict,
        mission_type : str,
    ) -> List[dict]: 
    """
    ## Case 2 - Walker-Delta Constellation 

    Constellation design follows an always-on / taskable ConOps where each orbital
    plane carries a mix of UNTASKABLE (U) wide-FOV event detectors and TASKABLE (T)
    narrow-FOV steerable measurement instruments. Untaskable satellites perform
    continuous nadir surveillance and generate event requests via onboard processing.
    Taskable satellites respond to these requests and execute targeted observations.

    Instrument key format: {INSTRUMENT_TYPE}_{CONSTELLATION}_{T or U}
        T = Taskable   - has maneuver field; narrow FOV; steerable; creates scheduling decisions
        U = Untaskable - no maneuver field; wide FOV; nadir-fixed; always-on event detector

    +---------------------+------------------------+--------------+----------------+----------+-------------+------------+--------------+-------------+------------+---------+------------------+--------------+---------------------+----------------------------------+
    | Mission             | Primary mission        | Instrument   | Walker (T/P/F) | Altitude | Inclination | Orbit type | LTAN         | RAAN offset | Sats/plane | Role    | N sats (instr.)  | N sats total | Off-nadir limit     | Instrument arrangement           |
    +---------------------+------------------------+--------------+----------------+----------+-------------+------------+--------------+-------------+------------+---------+------------------+--------------+---------------------+----------------------------------+
    | Water quality       | Algal bloom monitoring | VNIR_WQ_U    | 54/6/1         | 500 km   | 97°         | SSO        | 10:30 asc.   | 0°          | 3          | U (det) | 18               | 54           | ±0° (nadir-fixed)   | Pattern 2: cycling               |
    | Water quality       | Algal bloom monitoring | TIR_WQ_T     | 54/6/1         | 500 km   | 97°         | SSO        | 10:30 asc.   | 0°          | 3          | T (msr) | 18               | —            | ±30°                | VNIR_WQ_U – TIR_WQ_T – ALT_WQ_U |
    | Water quality       | Algal bloom monitoring | ALT_WQ_U     | 54/6/1         | 500 km   | 97°         | SSO        | 10:30 asc.   | 0°          | 3          | U (msr) | 18               | —            | ±0° (nadir-fixed)   | per plane (3 of each × 6 planes) |
    | Flood monitoring    | High-flow monitoring   | VNIR_FL_U    | 50/10/2        | 550 km   | 53°         | MIO        | N/A          | 36°         | 2          | U (det) | 20               | 50           | ±0° (nadir-fixed)   | Pattern 2: cycling               |
    | Flood monitoring    | High-flow monitoring   | VNIR_FL_T    | 50/10/2        | 550 km   | 53°         | MIO        | N/A          | 36°         | 2          | T (msr) | 20               | —            | ±45°                | VNIR_FL_U – VNIR_FL_T – ALT_FL_U |
    | Flood monitoring    | High-flow monitoring   | ALT_FL_U     | 50/10/2        | 550 km   | 53°         | MIO        | N/A          | 36°         | 1          | U (msr) | 10               | —            | ±0° (nadir-fixed)   | (2-2-1 per plane × 10 planes)    |
    | Fire monitoring     | Wildfire monitoring    | MIR_FR_U     | 48/4/1         | 550 km   | 97°         | SSO        | 13:30 asc.   | 60°         | 6          | U (det) | 24               | 48           | ±0° (nadir-fixed)   | Pattern 2: cycling                          |
    | Fire monitoring     | Wildfire monitoring    | VNIR_FR_T    | 48/4/1         | 550 km   | 97°         | SSO        | 13:30 asc.   | 60°         | 2          | T (msr) | 8                | —            | ±45°                | MIR_FR_U(×6) – VNIR_FR_T(×2) –             |
    | Fire monitoring     | Wildfire monitoring    | SAR_FR_T     | 48/4/1         | 550 km   | 97°         | SSO        | 13:30 asc.   | 60°         | 2          | T (msr) | 8                | —            | ±15°–55° (bilateral)  | SAR_FR_T(×2) – TIR_FR_T(×2) per plane      |
    | Fire monitoring     | Wildfire monitoring    | TIR_FR_T     | 48/4/1         | 550 km   | 97°         | SSO        | 13:30 asc.   | 60°         | 2          | T (msr) | 8                | —            | ±30°                | (6-2-2-2 per plane × 4 planes)              |
    | TOTAL               |                        |              |                |          |             |            |              |
    +---------------------+------------------------+--------------+----------------+----------+-------------+------------+--------------+-------------+------------+---------+------------------+--------------+---------------------+----------------------------------+

    Taskable satellites:   TIR_WQ_T (18) + VNIR_FL_T (20) + VNIR_FR_T (12) + SAR_FR_T (12) = 62 taskable
    Untaskable satellites: VNIR_WQ_U (18) + ALT_WQ_U (18) + VNIR_FL_U (20) + ALT_FL_U (10)
                        + MIR_FR_U (24)                                                       = 90 untaskable
    Total: 152 satellites

    Instrument summary:
        VNIR_WQ_U : HawkEye-class ocean color (SeaHawk heritage, 3U CubeSat)
                    8-band VNIR 412-865nm; 120m GSD; 216km swath; nadir-fixed
                    Untaskable Chl-A anomaly detector for algal bloom event generation
        TIR_WQ_T  : constellr HiVE TIR (SkyBee-1 class, ~100kg microsatellite)
                    4-band LWIR 8-12um; 28.9m GSD; 18.5km swath; ±30deg off-nadir
                    Taskable water surface temperature retrieval on detected blooms
        ALT_WQ_U  : SMASH Ka-band nadir altimeter (BWI/CNES REVALTO, nanosatellite)
                    Ka-band 35.75GHz; 10cm height accuracy; nadir-only virtual stations
                    Untaskable scheduled water level — NOT an autonomous event detector

        VNIR_FL_U : Planet SuperDove PSB.SD class (3U CubeSat, ~5kg)
                    8-band VNIR 431-885nm; 3.7m GSD; 22.7km swath; nadir-fixed
                    Untaskable NDWI change detector for flood onset event generation
        VNIR_FL_T : Dragonfly Aerospace Caiman/Komodo class (6-12U CubeSat, ~12kg)
                    11-band Sentinel-2 matched VNIR 443-1390nm; 6.5m GSD; 15km swath; ±45deg
                    Taskable high-resolution flood boundary delineation
        ALT_FL_U  : SMASH Ka-band nadir altimeter — flood shell (same spec as ALT_WQ_U)
                    53deg MIO orbit provides mid-latitude river basin ground track weighting
                    Untaskable scheduled river stage — NOT an autonomous event detector

        MIR_FR_U  : OroraTech OTC-P1 SAFIRE (8U CubeSat on Spire LEMUR bus, ~8kg total)
                    3-channel MWIR+LWIR 3-12um; 625m native GSD; 400km swath; nadir-fixed
                    Untaskable fire pixel detector (onboard Nvidia Jetson AI, <3min alert)
        VNIR_FR_T : Dragonfly Aerospace Chameleon-MS+SWIR (6U CubeSat, ~4.5kg)
                    9-band VNIR+SWIR 450-1750nm; 20m GSD; 40km swath; ±45deg off-nadir
                    Taskable fire extent (NDVI) and burn severity (dNBR proxy at 1600nm)
        SAR_FR_T  : Capella Space X-SAR (Acadia class, ~100kg microsatellite)
                    X-band 9.4-9.9GHz; spotlight 0.25m / stripmap 1.2m; bilateral ±15-55deg
                    Taskable all-weather night/smoke fire extent and burn structure mapping
        TIR_FR_T  : constellr HiVE TIR (SkyBee-1 class, ~100kg microsatellite)
                    4-band LWIR 8-12um; 28.9m GSD; 18.5km swath; ±30deg off-nadir
                    Taskable fire LST retrieval — smoldering detection and perimeter thermal
                    gradient mapping at baseline spatial resolution (28.9m vs SAFIRE 625m)
                    Complements MIR_FR_U LWIR channels with cryocooled MCT sensitivity

    ConOps notes:
        - Untaskable (U) satellites: no maneuver field in JSON; fixed nadir pointing;
        execute pre-programmed observation strips; process data onboard to detect events;
        broadcast event requests to taskable satellites via ISL or ground relay.
        - Taskable (T) satellites: have maneuver field in JSON; receive event requests;
        participate in task allocation (CBBA, MILP, greedy planners); execute targeted
        observations; subject to scheduling decisions by autonomous planners.
        - ALT_WQ_U and ALT_FL_U are classified as untaskable measurement assets rather
        than event detectors — they do not autonomously generate event requests. Their
        nadir-only coverage is insufficient for reliable autonomous flood/bloom detection.
        Water level data is forwarded to ground for offline analysis.
        - The untaskable detection coverage gradient (MIR_FR_U >> VNIR_WQ_U >> VNIR_FL_U
        by swath width: 400km > 216km > 22.7km) creates observable differences in
        Onboard vs Oracle performance across constellations, motivating RQ2 findings.

    RAAN values at epoch 2019-02-15 12:00 UTC (Sun RA = 328.48 deg):
        Formula: RAAN = (RA_sun + LTAN_deg - 180) mod 360
        Water quality (LTAN 10:30, P=6, dRAAN=60 deg):
            Base RAAN = 305.98 deg
            Planes: 305.98, 5.98, 65.98, 125.98, 185.98, 245.98 deg
        Fire (LTAN 13:30, P=4, dRAAN=90 deg):
            Base RAAN = 350.98 deg
            Planes: 350.98, 80.98, 170.98, 260.98 deg
        Flood (MIO 53 deg, P=10, dRAAN=36 deg, epoch only):
            Base RAAN = 341.98 deg (offset +36 deg from water quality base)
            Planes: 341.98, 17.98, 53.98, 89.98, 125.98, 161.98, 197.98, 233.98, 269.98, 305.98 deg
            WARNING: RAAN precesses at ~-2.06 deg/day at 53 deg inclination.
                    Values valid only at epoch 2019-02-15 12:00 UTC.
                    Recompute for each simulation date using:
                    RAAN(t) = RAAN(t0) + (dRAAN/dt) * (t - t0)
    """
    # Algal bloom monitoring constellation (SSO, 500 km, 97 deg, 36 sats in 6 planes)
    algal_bloom_monitoring = WalkerDeltaConstellation(
        alt=500,
        inc=97.0,
        num_sats=36,
        num_planes=6,
        phasing_param=1,
        raan_offset=0.0
    )
    algal_bloom_monitoring_specs = assign_instruments_to_constellation(
        constellation=algal_bloom_monitoring,
        const_name="Algal Bloom Monitoring",
        const_id="wq",
        instruments=["VNIR-WQ", "TIR-WQ", "ALT-WQ"],
        instrument_specs=instrument_specs,
        spacecraft_specs_template=spacecraft_specs_template,
    )
    assign_mission_to_constellation(algal_bloom_monitoring_specs, f"algal bloom {mission_type}")

    # High-flow monitoring constellation (MIO, 550 km, 53 deg, 30 sats in 5 planes)
    flood_monitoring = WalkerDeltaConstellation(
        alt=550,
        inc=53.0,
        num_sats=30,
        num_planes=5,
        phasing_param=2,
        raan_offset=36.0
    )
    flood_monitoring_specs = assign_instruments_to_constellation(
        constellation=flood_monitoring,
        const_name="Flood Monitoring",
        const_id="fl",
        instruments=["VNIR-FL", "ALT-FL"],
        instrument_specs=instrument_specs,
        spacecraft_specs_template=spacecraft_specs_template
    )
    assign_mission_to_constellation(flood_monitoring_specs, f"high flow river {mission_type}")

    # Wildfire monitoring constellation (SSO, 550 km, 97 deg, 36 sats in 3 planes)
    wildfire_monitoring = WalkerDeltaConstellation(
        alt=550,
        inc=97.0,
        num_sats=32,
        num_planes=4,
        phasing_param=1,
        raan_offset=60.0
    )
    wildfire_monitoring_specs = assign_instruments_to_constellation(
        constellation=wildfire_monitoring,
        const_name="Wildfire Monitoring",
        const_id="fr",
        instruments=["TIR-FR", "MIR-FR", "VNIR-FR", "SAR-FR"],
        instrument_specs=instrument_specs,
        spacecraft_specs_template=spacecraft_specs_template,
    )
    assign_mission_to_constellation(wildfire_monitoring_specs, f"wildfire {mission_type}")

    # TDRSS constellation for increased connectivity
    tdrss_specs = generate_tdrss(spacecraft_specs_template)
    assign_mission_to_constellation(tdrss_specs, f"none")

    # compile constellation design into dictionary format
    constellation_design = []
    for sat_specs in [algal_bloom_monitoring_specs, flood_monitoring_specs, wildfire_monitoring_specs, tdrss_specs]:
        constellation_design.extend(sat_specs)

    # return constellation design
    return constellation_design

if __name__ == "__main__":
    """ TEST SCRIPT FOR GENERATING CONSTELLATION DESIGNS """

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
    
    # # generate commercial constellation design
    # commercial_constellation : dict = generate_commercial(spacecraft_specs_template, instrument_specs)

    # generate walker delta constellation design    
    walker_delta_constellation : dict = generate_walker_delta(spacecraft_specs_template, instrument_specs, 'monitoring')
    
    # define output path for constellation designs
    out_dir = os.path.join('.', 'experiments','2_centralized_vs_decentralized', 'resources', 'constellations')
    
    # ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # save constellation designs to json
    # commercial_out_path = os.path.join(out_dir, "commercial_constellation.json")
    # with open(commercial_out_path, "w") as f:
    #     json.dump(commercial_constellation, f, indent=4)

    walker_delta_out_path = os.path.join(out_dir, "walker_delta_constellation.json")
    with open(walker_delta_out_path, "w") as f:
        json.dump(walker_delta_constellation, f, indent=4)

    # diagnostic print
    # print(f"Commercial constellation design saved to: \n\t`{commercial_out_path}`")
    print(f"Walker delta constellation design saved to: \n\t`{walker_delta_out_path}`")

    # TODO print summary of constellation designs
    # print("\nSummary of Constellation Designs:")
    
    # `done` message
    print('DONE')
    

    