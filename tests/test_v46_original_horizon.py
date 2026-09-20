import numpy as np

from full_model.analysis.run_v46_original_horizon import plan


def test_original_horizon_plan_has_exact_split_clock_and_front_count():
    macro_dt = 7.8125e-6
    rows, dt = plan(macro_dt, 8, 8)
    assert len(rows) == 7*17
    assert sum(row["kind"] == "front" for row in rows) == 7
    assert sum(row["kind"] == "mura" for row in rows) == 7*16
    np.testing.assert_allclose(dt, 4.8828125e-7, rtol=0.0, atol=1e-21)
    np.testing.assert_allclose(rows[-1]["physical_time_s"], 62.5e-6,
                               rtol=0.0, atol=1e-20)
    assert [row["global_mura_index"] for row in rows
            if row["kind"] == "mura"] == list(range(17, 129))
