#!/usr/bin/env python3
"""Write the fixed priority-ordered Directive-v12 case table."""
import sys

source128 = "../../.hpc3/inputs/v9_event_restart_006900.npz"
source192 = "../../.hpc3/inputs/v12_event_restart_006900_192.npz"
source256 = "../../.hpc3/inputs/v12_event_restart_006900_256.npz"
cases = [
    ("A1_baseline_bulge",128,12000,1000,1100,.75,3,1,0,source128,"A"),
    ("A2_no_bulge",128,12000,1000,1100,0,3,1,0,source128,"A"),
    ("A3_window_2um",128,6000,1000,1100,.75,2,1,0,source128,"A"),
    ("A4_window_3p5um",128,6000,1000,1100,.75,3.5,1,0,source128,"A"),
    ("A5_radius_0p5um",128,6000,1000,1100,.5,3,1,0,source128,"A"),
    ("A6_radius_1um",128,6000,1000,1100,1,3,1,0,source128,"A"),
    ("B1_mobility_0p3",128,6000,1000,1100,.75,3,.3,0,source128,"B"),
    ("B2_mobility_3",128,6000,1000,1100,.75,3,3,0,source128,"B"),
    ("B3_temperature_1000K",128,6000,1000,1000,.75,3,1,0,source128,"B"),
    ("B4_temperature_1200K",128,6000,1000,1200,.75,3,1,0,source128,"B"),
    ("B5_rate_300",128,1800,300,1100,.75,3,1,0,source128,"B"),
    ("B6_rate_3000",128,18000,3000,1100,.75,3,1,0,source128,"B"),
    ("D1_grid192_bulge",192,6000,1000,1100,.75,3,1,0,source192,"D"),
    ("D2_grid192_control",192,6000,1000,1100,0,3,1,0,source192,"D"),
    ("D3_grid256_bulge",256,4000,1000,1100,.75,3,1,0,source256,"D"),
    # Fixed 35.5 MPa drag leaves ~0.83 MPa net initial flat drive, giving the
    # declared verification target Rc=gamma/net_drive ~=0.60 um.
    ("E1_subcritical_fixture",128,6000,1000,1100,.42,3,1,3.55e7,source128,"E"),
    ("E2_supercritical_fixture",128,6000,1000,1100,.78,3,1,3.55e7,source128,"E"),
]
with open(sys.argv[1], "w") as stream:
    for row in cases:
        stream.write("|".join(map(str, row))+"\n")
