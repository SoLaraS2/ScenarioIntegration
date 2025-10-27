from __future__ import annotations
cprops = cprops.dropna(subset=["weather_datetime"])
cprops["cooling_prop"] = pd.to_numeric(cprops["cooling_prop"], errors="coerce")
prop_series = cprops.groupby("weather_datetime")["cooling_prop"].mean().astype("float32")
ts_index = exist_df.index.get_level_values("weather_datetime").tz_localize(None)
prop_series.index = prop_series.index.tz_localize(None)
prop_series = prop_series[~prop_series.index.duplicated(keep="last")]
prop = prop_series.reindex(ts_index)
mean_prop = float(prop.mean()) if prop.notna().any() else 0.5
prop = prop.fillna(mean_prop).to_numpy(dtype="float32")


# Build per-(wy,ts) target using index ratios
wy_vals = exist_df.index.get_level_values("weather_year").to_numpy()
ratios = np.array([float(wy_index.get(int(w), 1.0)) for w in wy_vals], dtype="float32")
target_vec = (float(adding_MW) * ratios).astype("float32")


# Remainder after existing
existing_total = exist_df.sum(axis=1).to_numpy(dtype="float32")
remain = (target_vec - existing_total).astype("float32")
remain[remain < 0] = 0.0
if float(remain.sum()) <= 0.0:
# nothing to add; return original (schema-enforced)
out = []
for chunk in pd.read_csv(src, chunksize=50_000, low_memory=False):
out.append(enforce_schema(chunk))
return pd.concat(out, ignore_index=True)


# Split into COOL/IT, then across states
scaled_total = remain[:, None].astype("float32")
cool_share = (scaled_total * prop[:, None]).astype("float32")
it_share = (scaled_total * (1.0 - prop)[:, None]).astype("float32")
cool_mat = (cool_share * vec_cool[None, :]).astype("float32")
it_mat = (it_share * vec_it[None, :]).astype("float32")


ts_vals = exist_df.index.get_level_values("weather_datetime").to_numpy()
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
df_cool = enforce_schema(df_cool)
df_it = enforce_schema(df_it)


# Return original + appended DC blocks (schema enforced)
orig = []
for chunk in pd.read_csv(src, chunksize=50_000, low_memory=False):
orig.append(enforce_schema(chunk))
base_df = pd.concat(orig, ignore_index=True)
return pd.concat([base_df, df_cool, df_it], ignore_index=True)


def add_datacenter_capacity_to_csv(self, out_path: os.PathLike | str, **kwargs) -> str:
df = self.add_datacenter_capacity(**kwargs)
out_path = str(out_path)
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)
return out_path