import datetime

import numpy as np
import pandas as pd
import pytest

from frictionless import Package, formats
from frictionless.resources import TableResource

# Read


def test_parquet_parser():
    with TableResource(path="data/table.parq") as resource:
        assert resource.header == ["id", "name"]
        assert resource.read_rows() == [
            {"id": 1, "name": "english"},
            {"id": 2, "name": "中国人"},
        ]


def test_parquet_parser_parquet_extension():
    with TableResource(path="data/table.parquet") as resource:
        assert resource.header == ["id", "name"]
        assert resource.read_rows() == [
            {"id": 1, "name": "english"},
            {"id": 2, "name": "中国人"},
        ]


def test_parquet_parser_columns():
    control = formats.ParquetControl(columns=["id"])
    with TableResource(path="data/table.parq", control=control) as resource:
        assert resource.header == ["id"]
        assert resource.read_rows() == [
            {"id": 1},
            {"id": 2},
        ]


def test_parquet_parser_converts_each_row_once(tmp_path, mocker):
    dataframe = pd.DataFrame(
        {
            "label": ["first", "second", "中国人"],
            **{f"value_{column}": [column] * 3 for column in range(93)},
        }
    )
    path = tmp_path / "wide.parquet"
    dataframe.to_parquet(path, index=False)
    expected = dataframe.to_dict("records")
    to_dict = mocker.spy(pd.Series, "to_dict")
    resource = TableResource(path=path.name, basepath=str(tmp_path))

    with resource:
        assert resource.header == list(dataframe.columns)
        assert resource.read_rows() == expected
    assert to_dict.call_count == len(dataframe)

    report = resource.validate()
    assert report.valid
    assert to_dict.call_count == 2 * len(dataframe)


def test_parquet_package_preserves_nulls_as_missing(tmp_path):
    # Mixed wide tables box NaNs as distinct Python floats. Parquet nulls must
    # remain missing when they pass through the pandas reader.
    dataframe = pd.DataFrame(
        {
            "label": ["synthetic"] * 1200,
            **{f"value_{column}": [-1.0] * 1200 for column in range(93)},
        }
    )
    dataframe.loc[106, dataframe.columns[1:]] = np.nan
    path = tmp_path / "nullable.parquet"
    dataframe.to_parquet(path, index=False)
    package = Package.from_descriptor(
        {
            "profile": "data-package",
            "name": "nullable",
            "resources": [
                {
                    "profile": "data-resource",
                    "name": "nullable",
                    "path": path.name,
                    "format": "parquet",
                }
            ],
        },
        basepath=str(tmp_path),
    )

    report = package.validate()

    assert report.valid, report.to_descriptor()
    assert report.to_descriptor()["tasks"][0]["stats"]["rows"] == 1200
    with package.get_table_resource("nullable") as resource:
        rows = resource.read_rows()
        assert all(rows[106][name] is None for name in dataframe.columns[1:])
        assert all(rows[105][name] == -1 for name in dataframe.columns[1:])


@pytest.mark.ci
def test_parquet_parser_remote():
    with TableResource(
        path="https://raw.githubusercontent.com/fdtester/test-repo-with-parquet-data-file/main/table.parq"
    ) as resource:
        assert resource.read_rows() == [
            {"id": 1, "name": "english"},
            {"id": 2, "name": "中国人"},
        ]


# Write


def test_parquet_parser_write(tmpdir):
    source = TableResource(path="data/table.csv")
    target = TableResource(path=str(tmpdir.join("table.parq")))
    source.write(target)
    with target:
        assert target.format == "parq"
        assert target.header == ["id", "name"]
        assert target.read_rows() == [
            {"id": 1, "name": "english"},
            {"id": 2, "name": "中国人"},
        ]


def test_parquet_parser_write_datetime_field_with_timezone(tmpdir):
    source = TableResource(data=[["datetimewithtimezone"], ["2025-08-24T15:20:10Z"]])
    target = TableResource(path=str(tmpdir.join("table.parq")))
    source.write(target)
    with target:
        assert target.format == "parq"
        assert target.header == ["datetimewithtimezone"]
        assert target.read_rows() == [
            {
                "datetimewithtimezone": datetime.datetime(
                    2025, 8, 24, 15, 20, 10, tzinfo=datetime.timezone.utc
                )
            }
        ]
