import numpy as np

from full_model.analysis.audit_v63_candidate_profiles import (
    component_at_anchor, profile,
)


def geometry():
    return {"center_m": [0.5, 0.5], "tangent": [1.0, 0.0],
            "normal": [0.0, 1.0], "tangent_half_width_m": 0.5}


def test_zero_overlap_closes_track_instead_of_selecting_nearest_replacement():
    previous = np.zeros((16, 16), bool); previous[2:5, 2:5] = True
    field = np.zeros((16, 16)); field[10:13, 10:13] = 10.0
    selected = component_at_anchor(field, (2/16, 2/16), 1.0, previous)
    assert not np.any(selected)


def test_two_peak_profile_is_envelope_not_single_peak_fwhm():
    n = 128
    y = (np.arange(n)+0.5)/n
    peaks = (np.exp(-0.5*((y-.3)/.02)**2)
             +np.exp(-0.5*((y-.7)/.02)**2))
    field = np.broadcast_to(peaks, (n, n))
    row = profile(field, geometry(), 1.0, 128)
    assert row["profile_classification"] == "MULTI_PEAK_ENVELOPE"
    assert row["half_maximum_run_count"] == 2
    assert row["half_maximum_envelope_m"] > 5*row["connected_peak_fwhm_m"]


def test_single_peak_connected_width_matches_envelope():
    n = 128
    y = (np.arange(n)+0.5)/n
    field = np.broadcast_to(np.exp(-0.5*((y-.5)/.04)**2), (n, n))
    row = profile(field, geometry(), 1.0, 128)
    assert row["profile_classification"] == "SINGLE_CONNECTED_PEAK"
    assert row["half_maximum_envelope_m"] == row["connected_peak_fwhm_m"]
