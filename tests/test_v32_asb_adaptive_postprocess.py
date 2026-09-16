from full_model.analysis.postprocess_v32_asb_adaptive import rank_key


def _record(active, ipr, entropy, heating, softening, qualifying=0):
    return {"summary": {
        "raw_conjunction": {"qualifying_snapshot_count": qualifying},
        "latest_active_fraction": active,
        "latest_inverse_participation_fraction": ipr,
        "latest_entropy_effective_fraction": entropy,
        "latest_temperature_excess_K": heating,
        "latest_softening_fraction": softening,
    }}


def test_rank_prefers_strict_conjunction_before_subcritical_metric():
    strict = _record(.24, .24, .30, 60, .21, qualifying=2)
    nonstrict = _record(.10, .10, .15, 200, .40, qualifying=0)
    assert rank_key(strict) < rank_key(nonstrict)


def test_rank_then_prefers_lower_active_support():
    localized = _record(.30, .32, .40, 80, .18)
    broad = _record(.60, .60, .70, 300, .30)
    assert rank_key(localized) < rank_key(broad)
