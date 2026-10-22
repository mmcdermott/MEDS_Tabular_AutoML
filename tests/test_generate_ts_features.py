import numpy as np
import polars as pl

from MEDS_tabular_automl.generate_ts_features import get_flat_ts_rep, get_long_value_df


def test_get_long_value_df_matches_raw_codes_to_value_features():
    df = pl.LazyFrame(
        {
            "code": ["LAB//RESULT//51227//%", "OTHER", "LAB//RESULT//51227//%"],
            "numeric_value": [1.5, 2.5, None],
        }
    )

    data, (rows, columns) = get_long_value_df(df, ["LAB//RESULT//51227//%/value"])

    np.testing.assert_array_equal(data, [1.5])
    np.testing.assert_array_equal(rows, [0])
    np.testing.assert_array_equal(columns, [0])


def test_value_sum_flat_representation_contains_numeric_values():
    code = "LAB//RESULT//51227//%"
    df = pl.LazyFrame(
        {
            "subject_id": [1, 1],
            "time": [1, 2],
            "code": [code, code],
            "numeric_value": [1.5, None],
        }
    )

    _, matrix = get_flat_ts_rep("value/sum", [f"{code}/value"], df)

    np.testing.assert_array_equal(matrix.toarray(), [[1.5], [0.0]])
