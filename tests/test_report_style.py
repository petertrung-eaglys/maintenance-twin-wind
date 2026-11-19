"""Tests for the report's output helpers."""

import matplotlib.pyplot as plt
import pandas as pd

from src.report.style import markdown_table, save


def test_markdown_table_puts_the_index_first_and_rounds_floats() -> None:
    frame = pd.DataFrame({"physics": [3.811, 3.5]}, index=["KLM", "PEN"])

    assert markdown_table(frame.rename_axis("farm")) == "\n".join(
        [
            "| farm | physics |",
            "| --- | --- |",
            "| KLM | 3.81 |",
            "| PEN | 3.50 |",
        ]
    )


def test_save_writes_a_png_into_a_new_folder(tmp_path) -> None:
    fig, _ = plt.subplots()

    path = save(fig, tmp_path / "figures" / "x.png", dpi=50)

    assert path.read_bytes().startswith(b"\x89PNG")
    assert not plt.fignum_exists(fig.number)  # closed
