import numpy as np

from full_model.analysis.audit_v63_candidate_profiles import (
    anchor_geometry, component_at_anchor, jaccard, profile,
)


def test_candidate_and_fixed_profile_track_periodic_band():
    n = 32; length = 1.0
    i, j = np.indices((n, n)); x = i/n; y = j/n
    distance = (y-.25+.5) % 1.0-.5
    field = np.exp(-.5*(distance/.04)**2)*(1.0+.2*np.cos(2*np.pi*x))
    component = component_at_anchor(field, (.5, .25), length)
    geometry = anchor_geometry(field, component, length)
    result = profile(field, geometry, length, 64)
    assert component.sum() > 0
    assert result["second_moment_width_m"] < .1
    assert result["fwhm_m"] > 0.0


def test_jaccard_is_exact_for_same_mask():
    mask = np.eye(4, dtype=bool)
    assert jaccard(mask, mask) == 1.0
