"""Medsboost column slots (MEDS_tabular_automl.medsboost_model.column_slots)."""

from types import SimpleNamespace

from omegaconf import OmegaConf

from MEDS_tabular_automl.medsboost_model import column_slots


def test_recency_joins_the_code_count_slot():
    """A code's windowed code/count columns and its code/recency column share one slot; value aggregations get
    their own slot per code, and static columns -1."""
    cfg = OmegaConf.create(
        {
            "path": {"input_tabularized_dir": "unused"},
            "tabularization": {
                "window_sizes": ["1d", "full"],
                "aggs": ["static/present", "code/count", "code/recency", "value/max"],
            },
        }
    )
    codes, values, statics = ["A/code", "B/code"], ["A/value"], ["S/static/present"]
    # Column names in get_model_files order (sorted paths): 1d/code/count, 1d/value/max, full/code/count,
    # full/code/recency, full/value/max, none/static/present.
    names = (
        [f"{c}/count/1d" for c in codes]
        + [f"{v}/max/1d" for v in values]
        + [f"{c}/count/full" for c in codes]
        + [f"{c}/recency/full" for c in codes]
        + [f"{v}/max/full" for v in values]
        + [f"{s}/none" for s in statics]
    )
    dataset = SimpleNamespace(
        cfg=cfg,
        split="train",
        _data_shards=["0"],
        get_all_column_names=lambda: names,
        code_masks={
            "code/count": [True, True],
            "code/recency": [True, False],  # B's recency is not kept.
            "value/max": [True],
            "static/present": [True],
        },
    )
    slots = column_slots(dataset).tolist()
    a_count, b_count, a_max = slots[0], slots[1], slots[2]
    assert len({a_count, b_count, a_max}) == 3
    assert slots[3:6] == [a_count, b_count, a_count]  # full counts, then A's recency in A's count slot.
    assert slots[6:] == [a_max, -1]
