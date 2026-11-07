"""Local, reproducible checks supporting the figure-by-figure reading."""
from pathlib import Path
import json
import numpy as np
from scipy.ndimage import map_coordinates, uniform_filter1d
from magviz import Plane

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
MU0, MP, QE = 4e-7 * np.pi, 1.6726e-27, 1.602e-19
result = {"scope": "Local slices and caches; no simulation rerun; all times in code units."}
eq, mer = Plane("eq"), Plane("mer")
avg = {p: dict(np.load(OUT / "cache" / f"avg_{p}.npz")) for p in ("eq", "mer")}
rep = json.loads((OUT / "repro.json").read_text(encoding="utf-8"))

def summary(a):
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std()),
            "median": float(np.median(a)), "p10_p90": np.percentile(a, [10, 90]).tolist()}

def zero_candidate(x, bz):
    candidates = np.flatnonzero((x[1:] < -10) & (bz[1:] > 0) & (bz[:-1] < 0)) + 1
    return float(x[candidates[-1]]) if len(candidates) else None

result["snapshots"] = {}
for pl, times in ((eq, [0, 8, 16, 25, 50, 80, 85, 110, 120, 125, 150, 160, 170, 180]),
                  (mer, [0, 8, 16, 25, 50, 80, 90, 110, 125, 150, 170])):
    frames = pl.frames()
    expected = .01 + .25 * (np.asarray(frames) - 1)
    for target in times:
        frame = frames[np.argmin(abs(expected - target))]
        d = pl.load(frame)
        band = abs(pl.tu) < .3
        b = d["Bz"][:, band].mean(1)
        n = d["N"][:, band].mean(1)
        v = d["Vx"][:, band].mean(1)
        day = (pl.xu > 4) & (pl.xu < 21.5)
        ix = np.flatnonzero(day & (b < 0))
        bs = np.flatnonzero(day & (n > 2 * rep["upstream"]["n"]))
        tail = (pl.xu > -44) & (pl.xu < -12)
        row = {"frame": int(frame), "actual_time": d["time"],
               "subsolar_Bz_sign_MP": float(pl.xu[ix[0]]) if len(ix) else None,
               "subsolar_density_BS": float(pl.xu[bs[-1]]) if len(bs) else None,
               "tail_Bz_zero_candidate": zero_candidate(pl.xu, uniform_filter1d(b, 5)),
               "tail_central_Vx_p01_p99": np.percentile(v[tail], [1, 99]).tolist()}
        if pl.plane == "mer":
            lob = (pl.XU > -24) & (pl.XU < -20) & (abs(pl.TU) > 5) & (abs(pl.TU) < 9)
            e = np.stack([d["E" + c] for c in "xyz"], axis=-1) * 1e-3
            B = np.stack([d["B" + c] for c in "xyz"], axis=-1) * 1e-9
            eb = np.cross(e, B) / np.maximum((B * B).sum(-1)[..., None], 1e-22) / 1e3
            sign = -np.sign(pl.TU[lob])
            row["lobe_inward_Uz_kms"] = float(np.mean(d["Vz"][lob] * sign))
            row["lobe_inward_ExBz_kms"] = float(np.mean(eb[..., 2][lob] * sign))
        result["snapshots"][f"{pl.plane}_{target}"] = row
print("snapshot checks complete", flush=True)

# Extract a narrow interpolated strip from each late frame, rather than loading full maps.
series = np.load(OUT / "eq_timeseries.npz")
late_frames = [f for f, t in zip(eq.frames(), series["t"]) if t >= 100]
sub = eq._idx[:, :, abs(eq.tu) < .3]
records = []
pressure = []
shock_flux_ratio = []
for i, f in enumerate(late_frames):
    with np.load(Path(eq.dir) / f"f{f:05d}.npz") as raw:
        g = lambda k: map_coordinates(raw[k], sub, order=1, mode="nearest")
        n = g("Ni") * eq.u["Ni"] * 1e-6
        B = np.stack([g("B" + c) * eq.u["B"] for c in "xyz"], -1)
        v = np.stack([g("Up" + c) * eq.u["Up"] for c in "xyz"], -1)
        tpar = g("Ti1par") * eq.u["T"] / 11604.518
        tper = g("Ti1per") * eq.u["T"] / 11604.518
        b2 = (B * B).sum(-1)
        ppar, pper = n * 1e6 * tpar * QE, n * 1e6 * tper * QE
        pxx = pper + (ppar - pper) * B[..., 0] ** 2 / np.maximum(b2, 1e-24)
        pin = (n * 1e6 * MP * v[..., 0] ** 2 + pxx + b2/(2*MU0) - B[..., 0]**2/MU0) * 1e9
        scalar = (n * 1e6 * MP * (v*v).sum(-1) + (ppar+2*pper)/3 + b2/(2*MU0)) * 1e9
        pressure.append(np.stack([pin.mean(1), scalar.mean(1)]))
        density = n.mean(1)
        bz = B[..., 2].mean(1)
        nsw = float(series["up_N"][series["t"] >= 100][i])
        bounds = {}
        for factor in (1.5, 2.0, 2.5):
            found = np.flatnonzero((eq.xu > 5) & (eq.xu < 21.5) & (density > factor*nsw))
            bounds[str(factor)] = float(eq.xu[found[-1]]) if len(found) else np.nan
        neg = np.flatnonzero((eq.xu > 3.5) & (bz < 0))
        bounds["mp"] = float(eq.xu[neg[0]]) if len(neg) else np.nan
        records.append(bounds)
        rel = eq.xu - bounds["2.0"]
        flux = (n * v[..., 0]).mean(1)
        up, dn = (rel > 2) & (rel < 4), (rel > -1) & (rel < -.5)
        shock_flux_ratio.append(float(flux[dn].mean()/flux[up].mean()))
result["BS_threshold_scan"] = {k: summary([r[k] for r in records]) for k in ("1.5", "2.0", "2.5", "mp")}
profile = np.mean(pressure, axis=0)
mp0 = rep["shue_fit"]["eq"]["r0"]
bs0 = rep["shue_fit"]["eq"]["bs_nose"]
regions = {"upstream": (17, 21), "sheath": (mp0+.5, bs0-.8), "inside_MP": (mp0-.8, mp0-.3)}
result["momentum_flux"] = {
    name: {"Pi_xx_ion_plus_magnetic_nPa": float(profile[0, (eq.xu > lo) & (eq.xu < hi)].mean()),
           "scalar_proxy_nPa": float(profile[1, (eq.xu > lo) & (eq.xu < hi)].mean())}
    for name, (lo, hi) in regions.items()
}
result["shock_lab_frame_mass_flux_ratio"] = summary(shock_flux_ratio)
np.savez_compressed(OUT / "figure_diagnostics_profiles.npz", x=eq.xu, Pi_xx_ion_magnetic=profile[0], scalar=profile[1])
print("179-frame boundary and pressure checks complete", flush=True)

# Audit both the event-selection windows and the numerical summary used in the SEA figure.
pe = np.load(OUT / "cache/probe_eq.npz")
t = pe["t"]
dt = float(np.median(np.diff(t)))
W = int(round(10/dt))
events = [(x, int(np.argmin(abs(t-t0)))) for x,t0 in rep["df"]["events"]]
result["DF_window_audit"] = {
    "all_records": len(events), "records_t_ge_100": int(sum(t[i] >= 100 for x,i in events)),
    "pre_actual_offsets": ((np.arange(W-6, W-1)-W)*dt).tolist(),
    "post_jump_actual_offsets": ((np.arange(W+2, W+6)-W)*dt).tolist(),
    "minimum_separation_code_units": 6.0,
}
result["DF_SEA_groups"] = {}
for start in (40, 100):
    ev = [(x,i) for x,i in events if t[i] > start]
    sea = {k: np.array([pe[f"{x}|{k}"][i-W:i+W+1] for x,i in ev]) for k in ("Bz","N","Vx","T")}
    pre = slice(W-6,W-1)
    db = sea["Bz"]-sea["Bz"][:,pre].mean(1,keepdims=True)
    nr = sea["N"]/sea["N"][:,pre].mean(1,keepdims=True)
    result["DF_SEA_groups"][str(start)] = {
        "n_records": len(ev), "mean_dBz_post_nT": float(db[:,W+2:W+6].mean()),
        "median_record_density_ratio": float(np.median(nr[:,W+2:W+8].mean(1))),
        "mean_record_peak_Vx_kms": float(sea["Vx"][:,W-2:W+3].max(1).mean()),
        "median_record_temperature_ratio": float(np.median(sea["T"][:,W+2:W+8].mean(1)/sea["T"][:,pre].mean(1))),
    }

# FTE threshold counts are zero-crossing records, not de-duplicated events.
pm = np.load(OUT / "cache/probe_mer.npz")
result["FTE_threshold_scan"] = {}
for name in ("mpN20", "mpN30", "mpS20", "mpS30"):
    lat = int(name[3:])
    sign = 1 if name[2] == "N" else -1
    theta = np.radians(lat)
    r = 11.6 * (2/(1+np.cos(theta)))**.6
    nh = np.array([np.cos(theta),sign*np.sin(theta)]) - .6*np.sin(theta)/(1+np.cos(theta))*np.array([-np.sin(theta),sign*np.cos(theta)])
    nh /= np.linalg.norm(nh)
    bn = pm[f"{name}|Bx"]*nh[0]+pm[f"{name}|Bz"]*nh[1]
    b = bn[pm["t"] >= 60]
    b = b - uniform_filter1d(b,9)
    zc = np.flatnonzero(np.diff(np.sign(b)) != 0)
    result["FTE_threshold_scan"][name] = {str(th): sum(1 for z in zc if z > 2 and z < len(b)-3 and b[z-2]*b[z+3] < 0 and min(abs(b[z-2]),abs(b[z+3])) > th) for th in (2,3,5,7)}

result["B0_estimator_samples_nT"] = []
for sign in (1,-1):
    for r in (5,6,7):
        i,j = np.argmin(abs(eq.xu)),np.argmin(abs(eq.tu-sign*r))
        result["B0_estimator_samples_nT"].append(float(avg["eq"]["Bz"][i,j]*r**3))
result["units"] = {"u_L_over_u_V_seconds": eq.u["L"]/eq.u["Up"], "u_t_seconds": eq.u["t"],
                   "physical_reference_Omega_ci_per_second": QE*eq.u["B"]/MP,
                   "Omega_ci_times_u_t": QE*eq.u["B"]/MP*eq.u["t"]}
(OUT / "figure_diagnostics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print("Wrote output/figure_diagnostics.json", flush=True)
