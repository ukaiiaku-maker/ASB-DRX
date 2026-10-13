import json

import pytest

from full_model.analysis.merge_adaptive_histories import merge_histories


def write(path, rows):
    path.write_text(json.dumps(rows))
    return path


def row(index, time):
    return {"adaptive_interval": index, "physical_time_s": time}


def test_merge_orders_segments_and_declares_gap(tmp_path):
    right = write(tmp_path/"right.json", [row(4, 4.0)])
    left = write(tmp_path/"left.json", [row(1, 1.0), row(2, 2.0)])
    result = merge_histories([right, left])
    assert [x["adaptive_interval"] for x in result["rows"]] == [1, 2, 4]
    assert result["gaps"] == [{"after_interval": 2, "before_interval": 4,
                               "missing_count": 1}]
    assert result["history_complete"] is False


def test_merge_accepts_identical_overlap_but_rejects_conflict(tmp_path):
    a = write(tmp_path/"a.json", [row(1, 1.0)])
    b = write(tmp_path/"b.json", [row(1, 1.0), row(2, 2.0)])
    assert merge_histories([a, b])["duplicate_intervals"] == [1]
    c = write(tmp_path/"c.json", [row(1, 1.1)])
    with pytest.raises(ValueError, match="conflicting"):
        merge_histories([a, c])
