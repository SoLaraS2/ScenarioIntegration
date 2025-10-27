from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataPaths:
"""All configurable input locations.
Point these wherever your files actually live.
"""
# Base folders
data_root: Path = Path("../Model_Integration_Tool_1")
files_dir: Path = Path("files") # e.g., repo-local aux files


# Step 1 source pattern (compressed CSVs per (year, scenario))
# Example: ../Model_Integration_Tool_1/files/2030_baseline.csv.gz
step1_glob_pattern: str = "{year}_{scenario}.csv.gz"


# Step 2 helpers
states_dc_map_csv: Path = Path("../Model_Integration_Tool_1/states_dc_map.csv")
weather_year_index_csv: Path = Path("files/weather_year_propagation.csv")
cooling_prop_files: dict = None


def __post_init__(self):
if self.cooling_prop_files is None:
self.cooling_prop_files = {
"average": self.files_dir / "avg_dc_cooling_prop.csv",
"baseline": self.files_dir / "baseline_dc_cooling_prop.csv",
"central": self.files_dir / "central_dc_cooling_prop.csv",
"conservative": self.files_dir / "conservative_dc_cooling_prop.csv",
}


# Helpers
def scenario_csv(self, *, year: int, scenario: str) -> Path:
return (self.data_root / "files" / self.step1_glob_pattern.format(year=year, scenario=scenario)).resolve()