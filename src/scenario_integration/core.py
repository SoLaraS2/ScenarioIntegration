from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
import textwrap

# ------------------------------ Final formatting ------------------------------
FINAL_COLUMNS = [
    "sector", "subsector", "dispatch_feeder", "weather_datetime",
    "alabama","alaska","arizona","arkansas","california","colorado",
    "connecticut","delaware","district of columbia","florida","georgia",
    "hawaii","idaho","illinois","indiana","iowa","kansas","kentucky",
    "louisiana","maine","maryland","massachusetts","michigan","minnesota",
    "mississippi","missouri","montana","nebraska","nevada","new hampshire",
    "new jersey","new mexico","new york","north carolina","north dakota",
    "ohio","oklahoma","oregon","pennsylvania","rhode island","south carolina",
    "south dakota","tennessee","texas","utah","vermont","virginia","washington",
    "west virginia","wisconsin","wyoming"
]
STATE_COLS = [c for c in FINAL_COLUMNS if c not in ("sector","subsector","dispatch_feeder","weather_datetime")]

def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
      - Force EXACT column order (FINAL_COLUMNS)
      - Drop extras
      - Parse weather_datetime
      - Numeric only for state columns, fill NaNs with 0
      - Category dtype for meta cols
    """
    df = df.reindex(columns=FINAL_COLUMNS)

    if "weather_datetime" in df.columns:
        df["weather_datetime"] = pd.to_datetime(df["weather_datetime"], errors="coerce")

    for c in STATE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df[STATE_COLS] = df[STATE_COLS].fillna(0.0)

    for c in ("sector","subsector","dispatch_feeder"):
        if c in df.columns:
            df[c] = df[c].astype("category")

    return df

# ------------------------------ Utils ------------------------------
def optimize_dataframe_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Aggressively downcast to save memory."""
    for col in df.select_dtypes(include=["object"]).columns:
        if col in ["subsector", "sector", "dispatch_feeder"]:
            df[col] = df[col].astype("category")
    for col in df.select_dtypes(include=["int64","int32","int16","int8"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64","float32"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df

def _norm_state(s: str) -> str:
    return str(s).strip().lower()

@dataclass
class ScenarioIntegrator:
    """
    Assumptions about files:
      - Scenario inputs at: data_root / "{year}_{scenario}.csv.gz"
      - Step1 output often written to: data_root / "{year}_custom_output.csv"
      - Step2 aux files located under:
            data_root / "files/..."

    """
    data_root: Path = Path("./files")

    # ------------------------------ Internal file helpers ------------------------------
    def _scenario_path(self, year: int, scen: str) -> Path:
        return self.data_root / f"{year}_{scen}.csv.gz"

    def _find_file(self, *cands: Path) -> Path | None:
        for p in cands:
            if p and Path(p).exists():
                return Path(p)
        return None

    def _load_scenario_file_minimal(self, scenario_name: str, year: int, columns=None) -> pd.DataFrame:
        """Load only required columns to minimize memory."""
        file_path = self._scenario_path(year, scenario_name)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        df = pd.read_csv(file_path, usecols=columns, low_memory=False)
        df = optimize_dataframe_dtypes(df)
        return df

    # ------------------------------ Step 1 ------------------------------
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
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Returns:
          - diagnostics=True -> audit DF 
          - otherwise -> schema-enforced DataFrame
        """
        import gc

        custom_values = custom_values or {}
        fallback_scenarios = fallback_scenarios or {}
        state_base_scenarios = state_base_scenarios or {}


        _ = shed_shift_enabled

        baseline_only_subsectors: dict = {}

        gc.collect()
        if verbose:
            print("[process] loading base scenario:", scenario, "year:", year)

        full_df = self._load_scenario_file_minimal(scenario, year)

        orig_subs = full_df["subsector"].astype(str).str.strip()
        orig_set = set(orig_subs.str.lower())

        # Per-state base overrides
        parsed_state_base_scenarios = {str(k).strip().lower(): v for k, v in state_base_scenarios.items()}

        for state, state_scenario in parsed_state_base_scenarios.items():
            if state_scenario != scenario and state != "__all_states__":
                if state not in full_df.columns:
                    continue

                needed_cols = ["subsector", "weather_datetime", state]
                state_df = self._load_scenario_file_minimal(state_scenario, year, columns=needed_cols)

                for subsector in full_df["subsector"].unique():
                    mask_full = (full_df["subsector"] == subsector)
                    mask_state = (state_df["subsector"] == subsector)
                    if not mask_state.any():
                        continue
                    scenario_map = dict(zip(
                        state_df.loc[mask_state, "weather_datetime"],
                        state_df.loc[mask_state, state]
                    ))
                    full_df.loc[mask_full, state] = (
                        full_df.loc[mask_full, "weather_datetime"]
                        .map(scenario_map)
                        .fillna(full_df.loc[mask_full, state])
                    )

                del state_df
                gc.collect()

        # Per-(state, subsector) fallback scenario overrides 
        parsed_fallback_scenarios: dict[tuple[str, str], str] = {}
        for key_string, fb_scenario in fallback_scenarios.items():
            if isinstance(key_string, str):
                parts = key_string.split(",")
                if len(parts) >= 2:
                    st = parts[0].strip().lower()
                    ss = ",".join(parts[1:]).strip()
                    parsed_fallback_scenarios[(st, ss)] = fb_scenario
            else:
                # allow tuple keys too
                try:
                    st, ss = key_string
                    parsed_fallback_scenarios[(str(st).strip().lower(), str(ss).strip())] = fb_scenario
                except Exception:
                    pass

        for (state, subsector), fb_scenario in parsed_fallback_scenarios.items():
            if subsector in baseline_only_subsectors:
                continue

            current_base = parsed_state_base_scenarios.get(state, scenario)
            if fb_scenario == current_base:
                continue

            fallback_df = self._load_scenario_file_minimal(fb_scenario, year)
            fallback_subsector_df = fallback_df[fallback_df["subsector"] == subsector].copy()

            if state == "__all_states__":
                full_df = full_df[full_df["subsector"] != subsector]
                full_df = pd.concat([full_df, fallback_subsector_df], ignore_index=True)
            else:
                if state in full_df.columns and state in fallback_subsector_df.columns:
                    subsector_mask = (full_df["subsector"] == subsector)
                    full_df.loc[subsector_mask, state] = fallback_subsector_df[state].values

            del fallback_df, fallback_subsector_df
            gc.collect()

        # Force baseline 
        if baseline_only_subsectors:
            baseline_df = self._load_scenario_file_minimal("baseline", year)
            for subsector in baseline_only_subsectors:
                full_df = full_df[full_df["subsector"] != subsector]
                baseline_subsector_df = baseline_df[baseline_df["subsector"] == subsector].copy()
                full_df = pd.concat([full_df, baseline_subsector_df], ignore_index=True)
            del baseline_df
            gc.collect()

        # Custom percent tweaks 
        parsed_custom_values: dict[tuple[str, str], float] = {}
        for key_string, percent in custom_values.items():
            if isinstance(key_string, str):
                parts = key_string.split(",")
                if len(parts) >= 2:
                    st = parts[0].strip().lower()
                    ss = ",".join(parts[1:]).strip()
                    parsed_custom_values[(st, ss)] = float(percent)
            else:
                try:
                    st, ss = key_string
                    parsed_custom_values[(str(st).strip().lower(), str(ss).strip())] = float(percent)
                except Exception:
                    pass

        for (state, subsector), percent in parsed_custom_values.items():
            if subsector in baseline_only_subsectors or percent == 0.0:
                continue
            mask = (full_df["subsector"] == subsector)
            if state == "__all_states__":
                for col in full_df.columns[4:]:
                    if col not in ["weather_datetime", "weather_year"]:
                        full_df.loc[mask, col] = pd.to_numeric(full_df.loc[mask, col], errors="coerce") * float(percent)
            elif state in full_df.columns:
                full_df.loc[mask, state] = pd.to_numeric(full_df.loc[mask, state], errors="coerce") * float(percent)

        full_df["weather_datetime"] = pd.to_datetime(full_df["weather_datetime"], errors="coerce")
        full_df = full_df.dropna(subset=["weather_datetime"])


        final_subs = full_df["subsector"].astype(str).str.strip()
        final_set = set(final_subs.str.lower())
        if diagnostics:
            return pd.DataFrame({
                "unique_in_count": [len(orig_set)],
                "unique_out_count": [len(final_set)],
                "missing_from_output": [sorted(orig_set - final_set)],
                "added_in_output": [sorted(final_set - orig_set)],
            })

        return enforce_schema(full_df)

    # ------------------------------ Step 2 (Flask /process_step2) ------------------------------
    def add_datacenter_capacity(
        self,
        *,
        year: int,
        base_year: int,
        adding_MW: float = 100_000.0,
        cooling_prop_scenario: str = "average",
        wy_scale_scenario: str = "average",
        src_path: str | Path | None = None,
        base_src_path: str | Path | None = None, 
        verbose: bool = True,
        replace_dc_load: bool = False,
    ) -> pd.DataFrame:
        import gc

        def find_file(*cands: Path) -> Path | None:
            return self._find_file(*cands)

        def detect_state_cols(csv_path: Path):
            head = pd.read_csv(csv_path, nrows=10)
            fixed = {"subsector", "weather_datetime", "sector", "dispatch_feeder", "weather_year"}
            return [c for c in head.columns if c not in fixed]

        def load_wy_index(scenario: str) -> dict[int, float]:
            cand = find_file(
                self.data_root / "files" / "weather_year_propagation.csv",
                self.data_root / "weather_year_propagation.csv",
                self.data_root.parent / "files" / "weather_year_propagation.csv",
                self.data_root.parent / "weather_year_propagation.csv",
            )
            if not cand:
                raise FileNotFoundError("weather-year index file (year,scenario,avg_prop) not found")

            df = pd.read_csv(cand)
            df.columns = [str(c).strip().lower() for c in df.columns]
            if not {"year","scenario","avg_prop"}.issubset(df.columns):
                raise ValueError("Index file must have columns: year, scenario, avg_prop")

            df["scenario"] = df["scenario"].astype(str).str.strip().str.lower()
            s = df[df["scenario"].eq(scenario)].copy()
            if s.empty:
                raise ValueError(f"No rows for scenario '{scenario}' in weather-year index")

            s["year"] = pd.to_numeric(s["year"], errors="coerce").astype("Int64")
            s = s.dropna(subset=["year"]).copy()
            s["year"] = s["year"].astype(int)
            s["avg_prop"] = pd.to_numeric(s["avg_prop"], errors="coerce").fillna(1.0)

            return {int(r.year): float(r.avg_prop) for r in s.itertuples(index=False)}

        def load_state_weights(state_cols: list[str]):
            mpath = find_file(
                self.data_root / "states_dc_map.csv",
                self.data_root / "files" / "states_dc_map.csv",
                self.data_root.parent / "states_dc_map.csv",
                self.data_root.parent / "files" / "states_dc_map.csv",
            )
            if not mpath:
                raise FileNotFoundError("states_dc_map.csv not found")

            m = pd.read_csv(mpath)
            m.columns = [str(c).strip().lower() for c in m.columns]
            if "state" not in m.columns:
                raise ValueError("states_dc_map.csv must have a 'state' column")
            m["state"] = m["state"].astype(str).str.strip().str.lower()

            def norm_col(series):
                v = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
                if v.max() > 3.0:
                    v = v / 100.0
                return v

            if {"percent_it", "percent_cool"}.issubset(set(m.columns)):
                m["w_it"] = norm_col(m["percent_it"])
                m["w_cool"] = norm_col(m["percent_cool"])
            elif "percent" in m.columns:
                w = norm_col(m["percent"])
                m["w_it"] = w
                m["w_cool"] = w
            else:
                raise ValueError("states_dc_map.csv needs 'percent' or both 'percent_it' and 'percent_cool'")

            s2w_it = dict(zip(m["state"], m["w_it"]))
            s2w_cool = dict(zip(m["state"], m["w_cool"]))

            def vec(mapper):
                arr = np.array([float(mapper.get(_norm_state(c), 0.0)) for c in state_cols], dtype="float32")
                tot = float(arr.sum())
                return (arr / tot).astype("float32") if tot > 0 else np.full(
                    len(state_cols), 1.0/len(state_cols), dtype="float32"
                )

            return vec(s2w_it), vec(s2w_cool)

        # NEW: scan helper that returns:
        # - full per-state exist_df indexed by (wy, ts)
        # - per-row total dc (sum across states, both subsectors)
        # - the union set of (wy, ts) pairs observed in file
        def scan_existing_dc(src_csv: Path, state_cols: list[str], *, label: str):
            exist: dict[tuple[int, pd.Timestamp], dict[str, float]] = {}
            all_pairs: set[tuple[int, pd.Timestamp]] = set()

            chunk_iter = pd.read_csv(src_csv, chunksize=50_000, low_memory=False)
            chunk_num = 0
            for chunk in chunk_iter:
                chunk_num += 1
                if verbose and chunk_num % 10 == 0:
                    print(f"[step2:{label}] scan chunk {chunk_num}")

                chunk = optimize_dataframe_dtypes(chunk)
                chunk["weather_datetime"] = pd.to_datetime(chunk["weather_datetime"], errors="coerce")
                if "weather_year" not in chunk.columns:
                    chunk["weather_year"] = chunk["weather_datetime"].dt.year
                chunk = chunk.dropna(subset=["weather_datetime", "weather_year"])

                subs = chunk["subsector"].astype(str).str.strip().str.lower()
                dc_mask = subs.isin({"data center it", "data center cooling"})
                cols = [c for c in state_cols if c in chunk.columns]

                # track all (wy,ts) pairs in file (so we can align)
                grp_all = chunk[["weather_year","weather_datetime"]].drop_duplicates()
                for _, r in grp_all.iterrows():
                    all_pairs.add((int(r["weather_year"]), pd.Timestamp(r["weather_datetime"])))

                if dc_mask.any() and cols:
                    for c in cols:
                        chunk.loc[:, c] = pd.to_numeric(chunk[c], errors="coerce").fillna(0.0)

                    grouped = (
                        chunk.loc[dc_mask, ["weather_year","weather_datetime"] + cols]
                        .groupby(["weather_year","weather_datetime"])[cols].sum()
                    )
                    for (wy, ts), row in grouped.iterrows():
                        key = (int(wy), pd.Timestamp(ts))
                        if key not in exist:
                            exist[key] = {s: 0.0 for s in state_cols}
                        for s in cols:
                            exist[key][s] += float(row[s])

                del chunk
                gc.collect()

            idx = pd.MultiIndex.from_tuples(sorted(all_pairs), names=["weather_year","weather_datetime"])
            exist_df = pd.DataFrame(exist).T.reindex(idx, fill_value=0.0)
            total = exist_df.sum(axis=1).astype("float32")
            return exist_df, total, idx

        # ---------- start ----------
        gc.collect()
        if verbose:
            print(f"[step2] start year={year} base_year={base_year} adding_MW={adding_MW}")

        # RULE: nothing changes before base_year
        if int(year) < int(base_year):
            if verbose:
                print(f"[step2] year {year} < base_year {base_year}: no DC added.")
            out = []
            for chunk in pd.read_csv(src_path or (self.data_root / f"{year}_custom_output.csv"), chunksize=50_000, low_memory=False):
                out.append(enforce_schema(chunk))
            return pd.concat(out, ignore_index=True)

        cooling_prop_scenario = str(cooling_prop_scenario).strip().lower()
        wy_scale_scenario = str(wy_scale_scenario).strip().lower()
        if cooling_prop_scenario not in {"average","baseline","central","conservative"}:
            cooling_prop_scenario = "average"
        if wy_scale_scenario not in {"average","baseline","central","conservative"}:
            wy_scale_scenario = "average"

        # current year step1 output (src)
        src = Path(src_path) if src_path else None
        if src is None:
            src = self._find_file(
                self.data_root / f"{year}custom_output.csv",
                self.data_root / f"{year}_custom_output.csv",
                self.data_root / f"{year}_custom_output_with_dc.csv",
            )
        if not src:
            raise FileNotFoundError(f"Input not found for year {year}. Provide src_path or save {year}_custom_output.csv in data_root.")

        # base year step1 output (NEW)
        base_src = Path(base_src_path) if base_src_path else None
        if base_src is None:
            base_src = self._find_file(
                self.data_root / f"{base_year}custom_output.csv",
                self.data_root / f"{base_year}_custom_output.csv",
                self.data_root / f"{base_year}_custom_output_with_dc.csv",
            )
        if not base_src:
            raise FileNotFoundError(
                f"Base-year input not found for base_year={base_year}. "
                f"Provide base_src_path or save {base_year}_custom_output.csv in data_root."
            )

        state_cols = detect_state_cols(src)
        wy_index = load_wy_index(wy_scale_scenario)
        vec_it, vec_cool = load_state_weights(state_cols)

        # scan existing DC totals for BOTH years
        exist_df_year, total_year, idx_year = scan_existing_dc(src, state_cols, label=str(year))
        exist_df_base, total_base, idx_base = scan_existing_dc(base_src, state_cols, label=str(base_year))

        # align to the current year's index (we only add rows for timestamps present in the current year file)
        total_base_aligned = total_base.reindex(idx_year, fill_value=0.0).to_numpy(dtype="float32")
        total_year_aligned = total_year.reindex(idx_year, fill_value=0.0).to_numpy(dtype="float32")

        # progression delta = (this_year_existing - base_year_existing), never negative
        delta_prog = (total_year_aligned - total_base_aligned).astype("float32")
        delta_prog[delta_prog < 0] = 0.0

        # cooling prop by scenario (same as before)
        file_map = {
            "average":      "avg_dc_cooling_prop.csv",
            "baseline":     "baseline_dc_cooling_prop.csv",
            "central":      "central_dc_cooling_prop.csv",
            "conservative": "conservative_dc_cooling_prop.csv",
        }
        cool_path = find_file(
            self.data_root / "files" / file_map[cooling_prop_scenario],
            self.data_root / file_map[cooling_prop_scenario],
            self.data_root.parent / "files" / file_map[cooling_prop_scenario],
        )
        if not cool_path:
            raise FileNotFoundError(f"{file_map[cooling_prop_scenario]} not found")

        cprops = pd.read_csv(cool_path, usecols=["weather_datetime","state","cooling_prop"])
        cprops["weather_datetime"] = pd.to_datetime(cprops["weather_datetime"], errors="coerce")
        cprops = cprops.dropna(subset=["weather_datetime"])
        cprops["cooling_prop"] = pd.to_numeric(cprops["cooling_prop"], errors="coerce")

        prop_series = (cprops.groupby("weather_datetime")["cooling_prop"].mean().astype("float32"))
        ts_index = idx_year.get_level_values("weather_datetime").tz_localize(None)
        prop_series.index = prop_series.index.tz_localize(None)
        prop_series = prop_series[~prop_series.index.duplicated(keep="last")]
        prop = prop_series.reindex(ts_index)
        mean_prop = float(prop.mean()) if prop.notna().any() else 0.5
        prop = prop.fillna(mean_prop).to_numpy(dtype="float32")

        # build ADDITION TARGET:
        # base_year gets +adding_MW (scaled by weather-year ratios) per timestamp
        # later years get +adding_MW + progression_delta
        wy_vals = idx_year.get_level_values("weather_year").to_numpy()
        ratios = np.array([float(wy_index.get(int(w), 1.0)) for w in wy_vals], dtype="float32")

        add_base = (float(adding_MW) * ratios).astype("float32")
        add_total = (add_base + delta_prog).astype("float32")  # <- THIS is the new rule

        if float(add_total.sum()) <= 0.0:
            if verbose:
                print("[step2] add_total is zero; returning schema-enforced original.")
            out = []
            for chunk in pd.read_csv(src, chunksize=50_000, low_memory=False):
                out.append(enforce_schema(chunk))
            return pd.concat(out, ignore_index=True)

        # split add_total into COOL/IT using time-varying prop
        scaled_total = add_total[:, None].astype("float32")
        cool_share = (scaled_total * prop[:, None]).astype("float32")
        it_share = (scaled_total * (1.0 - prop)[:, None]).astype("float32")

        # split across states with weights (fixed)
        cool_mat = (cool_share * vec_cool[None, :]).astype("float32")
        it_mat = (it_share * vec_it[None, :]).astype("float32")

        ts_vals = idx_year.get_level_values("weather_datetime").to_numpy()
        df_cool = pd.DataFrame({
            "subsector": np.full(len(ts_vals), "data center cooling", dtype=object),
            "weather_datetime": ts_vals,
            "sector": np.full(len(ts_vals), "commercial", dtype=object),
            "dispatch_feeder": np.full(len(ts_vals), "Commercial", dtype=object),
            **{col: cool_mat[:, i] for i, col in enumerate(state_cols)}
        })
        df_it = pd.DataFrame({
            "subsector": np.full(len(ts_vals), "data center it", dtype=object),
            "weather_datetime": ts_vals,
            "sector": np.full(len(ts_vals), "commercial", dtype=object),
            "dispatch_feeder": np.full(len(ts_vals), "Commercial", dtype=object),
            **{col: it_mat[:, i] for i, col in enumerate(state_cols)}
        })

        df_cool = optimize_dataframe_dtypes(df_cool)
        df_it = optimize_dataframe_dtypes(df_it)

        out = []
        for chunk in pd.read_csv(src, chunksize=50_000, low_memory=False):
            out.append(enforce_schema(chunk))
        out_df = pd.concat(out, ignore_index=True)

        df_cool = enforce_schema(df_cool)
        df_it = enforce_schema(df_it)

        # IMPORTANT: we APPEND, we do not modify existing DC rows
        #filter out the dc it and cool from out_df
        if replace_dc_load: 
            out_df = out_df[~out_df["subsector"].isin(["data center cooling", "data center it"])]

        out_df = pd.concat([out_df, df_cool, df_it], ignore_index=True)
        return out_df


    def add_datacenter_capacity_to_csv(self, out_path: str | Path, **kwargs) -> str:

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df = self.add_datacenter_capacity(**kwargs)
        df.to_csv(out_path, index=False)
        return str(out_path)

    @property
    def help(self):
        print("ScenarioIntegrator Help".ljust(80, "="))
        print("Main methods:\n")
        print("• Nedded files in 'files/..'")
        print(textwrap.dedent("""
            - states_dc_map.csv [state | percent]
            - \{scenario\}_dc_cooling_prop.csv [weather_datetime | state | cooling_prop] -- For whichever scenario you would like
            - weather_year_propagation.csv [year | scenario | avg_prop]
        """).strip())

        print("• process(year, scenario, *, ...)")
        print(textwrap.dedent("""
            - year: int
            - scenario: str ("baseline" | "central" | "conservative") -- Overall Fallback Scenario
            - custom_values: dict[(state, subsector) -> float] -- All specified values will be multiplied by this factor
                  Example: {("texas","passenger car"): 1.10} OR {"texas, passenger car": 1.50} OR {"__all_states__, data center it": 0.85}
            - fallback_scenarios: dict[(state, subsector) -> scenario] -- Overrides Original Scenario and State Based Scenarios
                  Example: {"texas, passenger car": "baseline"} 
            - state_base_scenarios: dict[state -> scenario] -- Overrides Original Scenario
                  Example: {"texas": "baseline"}
            - diagnostics: bool -- If "True" Only diagnostics are returned, use for testing expected behavior
            - verbose: bool -- Toggle true to see print statements of updated progress
        """).strip())

        print("\n• add_datacenter_capacity(year, base_year, *, ...)")
        print(textwrap.dedent("""
            - adding_MW: float = 100_000 Fallback
            - cooling_prop_scenario: str in {"average","baseline","central","conservative"} -- Average is an average of the three other scenarios
            - wy_scale_scenario: str in {"average","baseline","central","conservative"}
            - src_path: Path | None (defaults to {data_root}/{year}_custom_output.csv if present)
            - base_src_path: Path | None (defaults to src_path) -- If you are calculatign a future year (lets say 2040 if your base year is 2030) put the base year here
            - replace_dc_load: bool = False (Set to True if you want to "replace" the DC load in the file with the new calculated load, set to False if you want to "add" the new calculated load on top of the existing load in the file.)
        """).strip())

        print("\nData root (where scenario CSVs live):")
        print(f"  {self.data_root.resolve()}")

        print("=" * 80)
        return None
