import pytest

from full_model.analysis.run_v67_drx_asb_overlap import (
    _production_source_mode,
)


def test_production_source_geography_defaults_to_gb_tj():
    assert _production_source_mode(True, "gb_tj_only") == "gb_tj_only"


def test_matched_random_is_an_explicit_production_control():
    assert _production_source_mode(True, "matched_random") == "matched_random"


def test_source_geography_cannot_be_enabled_without_production_sources():
    assert _production_source_mode(False, "gb_tj_only") == "disabled"
    with pytest.raises(ValueError, match="requires --production-rx"):
        _production_source_mode(False, "matched_random")
