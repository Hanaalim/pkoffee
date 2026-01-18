import pandas as pd
from pkoffee.data import (
    ColumnTypeError,
    RequiredColumn,
    validate,
)


def test_validate() -> None:
    """Test validate with valide DataFrame."""
    assert validate(pd.DataFrame({"cups": [0], "productivity": [1.2]})) is None
