from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
import textwrap

# -------- public helper --------
STATE_COLS: list[str] = []  # will be set after first enforce_schema call

def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    global STATE_COLS
    if "weather_datetime" in df.columns:
        df["weather_datetime"] = pd.to_datetime(df["weather_datetime"], errors="coerce")
    # infer state cols (anything not meta)
    meta = {"sector", "subsector", "dispatch_feeder", "weather_datetime", "weather_year"}
    STATE_COLS = [c for c in df.columns if c not in meta]
    for c in STATE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[STATE_COLS] = df[STATE_COLS].fillna(0.0)
    for c in ("sector", "subsector", "dispatch_feeder"):
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df

# -------- package API --------
@dataclass
class ScenarioIntegrator:
    """Notebook-friendly wrapper. Fill in real logic next."""
    data_root: Path = Path("./files")

    # step 1 placeholder
    def process(
        self,
        *,
        year: int,
        scenario: str,
        custom_values: dict | None = None,
        fallback_scenarios: dict | None = None,
        state_base_scenarios: dict | None = None,
        shed_shift_enabled: dict | None = None,
        diagnostics: bool = False,
    ) -> pd.DataFrame:
        # temporary: load the base CSV to prove import/flow works
        path = self.data_root / f"{year}_{scenario}.csv.gz"
        df = pd.read_csv(path)
        return enforce_schema(df)
    @property
    def help(self):
        """Pretty, pandas-style help on what you can tweak."""
        print("ScenarioIntegrator Help".ljust(80, "="))
        print("Main methods:\n")

        print("• process(year, scenario, *, ...)")
        print(textwrap.dedent("""
            - year: int
            - scenario: str ("baseline" | "central" | "conservative")
            - custom_values: dict[(state, subsector) -> float]
                  Example: {("texas","passenger car"): 1.10}
            - fallback_scenarios: dict[(state, subsector) -> scenario]
            - state_base_scenarios: dict[state -> scenario]
            - shed_shift_enabled: dict[state -> bool]
            - diagnostics: bool
        """).strip())

        print("\n• add_datacenter_capacity(year, base_year, *, ...)")
        print(textwrap.dedent("""
            - adding_MW: float = 100_000
                  Total MW to add across all states
            - cooling_prop_scenario: str
                  One of {"average","baseline","central","conservative"}
            - wy_scale_scenario: str
                  One of {"average","baseline","central","conservative"}
            - src_path: Path | None
                  Override input file location
        """).strip())

        print("\nData root (where CSVs live):")
        print(f"  {self.data_root.resolve()}")

        print("=" * 80)
        return None

    # step 2 placeholder
    def add_datacenter_capacity(
        self,
        *,
        year: int,
        base_year: int,
        adding_MW: float = 100_000.0,
        cooling_prop_scenario: str = "average",
        wy_scale_scenario: str = "average",
        src_path: str | Path | None = None,
    ) -> pd.DataFrame:
        # temporary: just echo the input file (so package imports fine)
        src = Path(src_path) if src_path else (self.data_root / "files" / f"{year}_custom_output.csv")
        df = pd.read_csv(src)
        return enforce_schema(df)

    def add_datacenter_capacity_to_csv(self, out_path: str | Path, **kwargs) -> str:
        df = self.add_datacenter_capacity(**kwargs)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        return str(out_path)
