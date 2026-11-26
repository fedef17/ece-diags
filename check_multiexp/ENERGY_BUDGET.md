# Energy budget across the EC-Earth4 components

How every number of the energy-budget diagram and tables is computed (`energy_budget.py`: `budget_terms`,
`budget_table`, `budget_checks`, `plot_budget_flow`; notebook `energy_budget.ipynb`). For each number: the model
output it needs, its formula, the closure check of its box, and why each variable was chosen. Term names used in
the code are in `monospace`; model output variables keep their names in the files.

Conventions:

- **Units and sign.** W m⁻². Fluxes are positive downward, into the lower compartment. Residuals are positive when
  energy is created.
- **Residual symbol ε.** In the diagram every residual is a coloured badge marked ε, with its definition underneath:
  - ε_OIFS = `atm_imb`; ε_HTESSEL = `land_res`;
  - ε of each OASIS row (out − in) and of the whole band (`band_res`);
  - ε_NEMO/SI3 = `oce_res`; total ε = `total_res`.

  Colours: green |ε| < 0.05, orange < 0.15, red above. The water rows of the OASIS band are lilac: NEMO's
  bookkeeping of the water heat content, not an error.
- **Time.** All terms are annual means of the monthly output (equal month weights). The tables and the diagram
  average them over a window of years (default: years 2 to 10 of each run). The window mean does not skip missing
  years: a term missing in one year is n/a.
- **Global or regional units.** By default every number is a **contribution to the global mean**, W m⁻² per Earth
  m² (`budget_table(..., units = 'global')`): all the numbers can be summed and the total budget closes.
  - Option `units = 'region'` gives each term per unit area of its own region. The diagram then shows the exact
    values, and the sums are not drawn.
  - The formulas below are written per region: atmosphere per Earth area, OIFS ocean and land per their own area,
    coupling, ice and ocean per NEMO ocean area.
  - The global contribution is the region value × its area / `A_E`: × `A_N / A_E` for the NEMO terms, × `A_o / A_E`
    and × `A_l / A_E` for the OIFS ocean and land means.
- **Rounding in the diagram** (`_display_values`, global units).
  - Only the fluxes are rounded, to 0.01. Residuals, sums and storages are recomputed from the rounded fluxes with
    the same formulas.
  - Where a flux splits, one part is derived from the rounded total: snow on open water = snowfall − snow on ice;
    heat to open water = heat received − heat to ice.
  - Every box, the band and the total then add up in the digits shown. A residual that sums n rounded fluxes can
    differ from its exact value by up to n × 0.005, typically 0.01 (xax0: ε_HTESSEL −0.02 shown, −0.028 exact;
    ACX0: ε_NEMO/SI3 −0.05 shown, −0.036 exact).
  - The tables and the checks use the exact values.
- **Notation.** `⟨X⟩_E`, `⟨X⟩_o`, `⟨X⟩_l`, `⟨X⟩_N` are the means defined in section 2. `L_f` = 334 000 J kg⁻¹ is
  the latent heat of fusion. One year = 365.25 days.

---

## 1. Required data

### 1.1 Model output (monthly means)

"Standard": written by the default EC-Earth4 XIOS templates (`scripts/runtime/templates/xios/file_def_*.xml.j2`,
revision 8f19493). **Non-standard variables (⚠) must be added to the run's `file_def` template.**

| file | variable | standard | meaning (units) | used for |
|---|---|---|---|---|
| `<exp>_atm_cmip6_1m_<yyyy>-<yyyy>.nc` (OIFS) | `rsdt` | yes | TOA incident shortwave (W m⁻²) | TOA net |
| | `rsut` | yes | TOA outgoing shortwave (W m⁻²) | TOA net |
| | `rlut` | yes | TOA outgoing longwave (W m⁻²) | TOA net |
| | `rsns` | yes | surface net shortwave, downward (W m⁻²) | surface net |
| | `rlns` | yes | surface net longwave, downward (W m⁻²) | surface net |
| | `hfss` | yes | surface sensible heat flux, upward (W m⁻²) | surface net |
| | `hfls` | yes | surface latent heat flux, upward (W m⁻²) | surface net |
| | `prsn` | yes | snowfall (kg m⁻² s⁻¹) | snowfall melting |
| | `ci` | yes | sea-ice fraction of OIFS (0–1) | snow on ice / open water, OIFS side |
| `<exp>_oce_1m_T_<yyyy>-<yyyy>.nc` (NEMO) | `heatc` | yes | vertically integrated heat content (J m⁻²) | OHC tendency |
| | `qt_oce` | yes | atmosphere → open water, with `qemp_oce` (W m⁻²) | open water / ice split |
| | `qemp_oce` | **⚠ no** | heat flux of E − P over open water (W m⁻²) | E − P heat |
| | `hflx_evap_cea` | **⚠ no** | heat content of evaporation (W m⁻², written positive) | E − P check |
| | `hflx_rain_cea` | **⚠ no** | heat content of rain (W m⁻²) | E − P check |
| | `hflx_snow_cea` | **⚠ no** | heat flux of snowfall, with melting (W m⁻²) | snow on open water |
| | `hflx_rnf_cea` | **⚠ no** | heat content of runoff (W m⁻²) | runoff heat |
| | `hflx_cal_cea` | **⚠ no** | latent heat of calving (W m⁻²; = −L_f × `calving_cea`) | calving latent heat |
| | `hflx_icb_cea` | **⚠ no** | latent heat of icebergs (W m⁻²; not written without icebergs) | calving latent heat |
| `<exp>_ice_1m_<yyyy>-<yyyy>.nc` (SI³) | `qt_atm_oi` | yes | atmosphere → ice + ocean (W m⁻²) | what NEMO receives |
| | `qt_oce_ai` | yes | ice + atmosphere → liquid ocean (W m⁻²) | ocean input |
| | `qemp_ice` | yes | heat flux of snowfall over the ice (W m⁻²) | snow on ice |

The ⚠ fields are written in xax0 and ACX0, added to the `_1m_T` file of `templates/xios/file_def_nemo-oce.xml.j2`:

```xml
<field field_ref="qemp_oce"      name="qemp_oce"/>
<field field_ref="hflx_evap_cea" name="hflx_evap_cea"/>
<field field_ref="hflx_rain_cea" name="hflx_rain_cea"/>
<field field_ref="hflx_snow_cea" name="hflx_snow_cea"/>
<field field_ref="hflx_rnf_cea"  name="hflx_rnf_cea"/>
<field field_ref="hflx_cal_cea"  name="hflx_cal_cea"/>
<field field_ref="hflx_icb_cea"  name="hflx_icb_cea"/>
```

All NEMO and SI³ fields are per unit cell area. Runs without the ⚠ fields get the combined residuals only
(section 6). Some standard NEMO fields are not used yet but could replace ⚠ fields: `calving_cea` (so that
`cal_latent = −L_f × calving_cea`), `snowpre` and `runoffs` (mass checks) (section 6).

### 1.2 Run directory (`/ec/res4/scratch/<user>/ece4/<exp>/`)

| file | content | used for |
|---|---|---|
| `areas.nc`, `masks.nc` (OASIS) | OIFS cell areas `<grid>-ocean.srf`, `<grid>-land.srf` and masks `<grid>-ocean.msk`, `<grid>-land.msk` (0 = active), with `<grid>` = TL63, TL255, …; NEMO T cells `<nemo>-T.srf`, `<nemo>-T.msk` (ORCA2, PALEORCA2, eORCA1-…) | OIFS ocean/land split; NEMO ocean area |
| `domain_cfg.nc` | `e1t`, `e2t`, `top_level`, `nav_lat`, `nav_lon` | NEMO areas and mask when the OASIS files are missing; positions for the geothermal map |
| `Goutorbe_ghflux.nc` | `gh_flux` (mW m⁻², 1°, covers land too): the map NEMO reads (`ln_trabbc`, `nn_geoflx = 2`) | geothermal flux; fallback if missing: `ECE4-DATA-4.1.11/nemo/initial/Goutorbe_ghflux.nc` |

---

## 2. Averages and areas

OIFS ocean and land weights, per OIFS cell, from the OASIS files:

- `w_o = <grid>-ocean.srf × (1 − <grid>-ocean.msk)`, `w_l = <grid>-land.srf × (1 − <grid>-land.msk)`
- `A_o = Σ w_o`, `A_l = Σ w_l`, `A_E = A_o + A_l`, `f_o = A_o / A_E`

NEMO ocean area: `A_N = Σ <nemo>-T.srf × (1 − <nemo>-T.msk)`, or `e1t × e2t` where `top_level` > 0.

Values (×10¹² m²):
- TL63 runs: `A_E` = 510.1 (= 4πR²), `A_o` = 364.7 (`f_o` = 71.5 %), `A_l` = 145.4.
- NEMO: `A_N` = 357.0 on ORCA2, 357.1 on PALEORCA2.
- TL255–eORCA1 (aI09): `A_o` = 362.0, `A_l` = 148.1, `A_N` = 362.7.

Means used below:

- `⟨X⟩_E`: global mean of an OIFS field over the reduced Gaussian grid with cos(latitude) weights (`check_run.global_mean`).
- `⟨X⟩_o = Σ w_o X / A_o`, `⟨X⟩_l = Σ w_l X / A_l`: OIFS means over its ocean and land points.
- `⟨X⟩_N = Σ area × mask × X / A_N`: NEMO/SI³ fields over the NEMO ocean. SI³ fields are integrated per hemisphere
  and summed (N + S).

---

## 3. The diagram, frame by frame

One row per run. From left to right:
- the **OIFS / HTESSEL** frame: atmosphere and land;
- the **OASIS** band, which every flux crosses;
- the **NEMO / SI3** frame: sea ice and liquid ocean;
- the **total**.

TOA enters from the left. Arrows and values are red when positive and blue when negative. † marks a number that
is not an output field (section 3.6).

### 3.1 OIFS / HTESSEL

**OIFS atmosphere** (per Earth area):

| quantity | term | formula |
|---|---|---|
| TOA net (arrow in, from the left) | `toa` | `⟨rsdt − rlut − rsut⟩_E` |
| surface net, without snowfall melting | `srf_net_nosn` | `⟨rsns + rlns − hfss − hfls⟩_E` |
| surface net | `srf_net` | `⟨rsns + rlns − hfss − hfls − L_f · prsn⟩_E` |
| **ε_OIFS** | `atm_imb` | `srf_net − toa` |

The surface flux leaves the atmosphere in two arrows: over the ocean points, to OASIS, and over the land, down to
HTESSEL. OIFS has no ocean component, so its fluxes over the ocean points go straight to OASIS.

| quantity | term | formula |
|---|---|---|
| flux over the OIFS ocean points, per OIFS ocean area | `srf_oce` | `⟨rsns + rlns − hfss − hfls − L_f · prsn⟩_o` |
| land flux, per land area | `srf_land` | `⟨rsns + rlns − hfss − hfls − L_f · prsn⟩_l` |
| ocean-point share of the global mean | `srf_oce_glob` | `srf_oce × f_o` |
| land share of the global mean (arrow "land") | `srf_land_glob` | `srf_land × (1 − f_o)` |
| ocean flux per NEMO ocean area (arrow "ocean") | `F_oifs` | `srf_oce × A_o / A_N` |
| land flux per NEMO ocean area (`srf_land_glob` in global units) | `land_in` | `srf_land_glob × A_E / A_N` |

**HTESSEL land.** The land stores no heat (checked below), so what it receives should leave as calving. The runoff
mapper moves the excess snow of the glaciers to NEMO, which pays its latent heat (`hflx_cal_cea`).

| quantity | term | formula (global units) |
|---|---|---|
| calving from the land (through OASIS to the ocean) | `cal_latent` | `(⟨hflx_cal_cea⟩_N + ⟨hflx_icb_cea⟩_N) × A_N / A_E` |
| **ε_HTESSEL** | `land_res` | `cal_latent − srf_land_glob` (calving out − land flux in) |
| **ε of the frame** | `oifs_res` | `atm_imb + land_res` |

**ε_HTESSEL is a loss, not storage.** Two independent checks rule out storage:
- **Land snow mass is flat.** OIFS `sd`, annual means integrated over the land, stays at 1.548·10¹⁷ kg in xax0 and
  ACX0. The trends are +5 and −17 Gt/yr, i.e. −0.0001 and +0.0004 W m⁻² as storage.
- **Total land storage is negligible.** The 1 January restart snapshots give soil + snow storage of about
  +0.001 W m⁻² globally (section 5).

Accumulating snow would make ε_HTESSEL positive. The observed −0.028 W m⁻² (xax0: calving −0.053 against a land flux
of −0.026) is therefore a loss on the snow → calving path. Possible contributors:
- **Remapping of the calving** (section 3.2), which is not conservative. Likely about 0.003 W m⁻².
- **The glacier snow cap, not checked yet.** The snow removed above HTESSEL's glacier cap may leave without its
  latent heat, while NEMO charges the full `L_f` for the calving.

**Checks:**
- identity: `toa − srf_net + atm_imb = 0`
- identity: `srf_land_glob − cal_latent + land_res = 0`
- identity: `oifs_res − atm_imb − land_res = 0`
- real (tolerance 0.005): `srf_net − srf_oce_glob − srf_land_glob = 0`. The global mean uses cos-latitude weights,
  the split uses the OASIS cell areas: −0.0003 W m⁻² in xax0 and ACX0.

### 3.2 OASIS band

**How the coupling works** (namcouple of xax0):
- **OIFS sends separate ocean and ice fluxes:** `A_Qs_oce`, `A_Qns_oce`, `A_Qs_ice`, `A_Qns_ice`, `A_dQns_dT`. NEMO
  rebuilds the cell total with its own ice fraction (`sbccpl.F90`, ~l. 1997).
- **Every field is remapped BILINEAR, with no global conservation.** This covers the heat fluxes, evaporation and
  precipitation (`A_Precip_solid` → `OTotSnow`), and runoff and calving (TL63-land → F128R → ORCA2).

OASIS is drawn as a grey band. Every flux crosses it, with what OIFS gives on the left (**in**), what NEMO / SI3
books on the right (**out**), and the mismatch ε = out − in in the middle. The sums are under the band.

| row | in (OIFS) | out (NEMO / SI3) | ε |
|---|---|---|---|
| snow on ice | `snow_ice_oifs = −L_f · ⟨prsn · ci⟩_o × A_o / A_N` (monthly product) | `qemp_ice` | `snow_res_ice` |
| heat, without snowfall | `F_oifs_nosn = ⟨rsns + rlns − hfss − hfls⟩_o × A_o / A_N` | `oasis_recv = ⟨qt_atm_oi⟩_N − qemp − cal_latent` † | `oasis_res` |
| snow on open water | `snow_ow_oifs = snow_melt − snow_ice_oifs` † | `snow_ow = ⟨hflx_snow_cea⟩_N − qemp_ice` † | `snow_res_ow` |
| evaporation + rain | 0 (water, mass only) | `evap_rain_ow = ⟨qemp_oce⟩_N − snow_ow` † | `evap_rain_ow` (lilac) |
| calving | `cal_latent` | `cal_latent` | ≡ 0, assumed |
| runoff | 0 (water, mass only) | `runoff_heat = ⟨hflx_rnf_cea⟩_N` | `runoff_heat` (lilac) |
| **sum** | `band_in = F_oifs + cal_latent` | `band_out = ⟨qt_atm_oi⟩_N + runoff_heat` | **`band_res = band_out − band_in`** |

Here `qemp = ⟨qemp_oce⟩_N + ⟨qemp_ice⟩_N` and `snow_melt = F_oifs − F_oifs_nosn = −L_f · ⟨prsn⟩_o × A_o / A_N`.
`band_res = oasis_res + snow_res + water_cycle`, with `snow_res = snow_res_ice + snow_res_ow` and
`water_cycle = evap_rain_ow + runoff_heat`.

- **Heat row (`oasis_res`, +0.21 W m⁻² in xax0, +0.18 in ACX0).** `F_oifs_nosn` is the OIFS grid-box mean over its
  ocean points, not literally what OASIS carries: OASIS carries the ocean and ice tile fluxes, and NEMO recombines
  them with its own ice fraction. `oasis_res` therefore mixes two effects:
  - the non-conservative bilinear remapping;
  - the difference between OIFS's and NEMO's ice fractions, which weight the ice flux (negative) and the open-water
    flux (positive) differently.

  The two cannot be separated without the OIFS tile fluxes in the output.
- **Heat over ice and open water.** NEMO's out value splits after the band into ice (`recv_ice`) and open water
  (`recv_ow`). OIFS writes only grid-box means, so only the total is compared. Splitting with `ci` would assume the
  same flux on ice and water, which have opposite signs.
- **Snow rows.** OIFS's snowfall over its ocean points is split with OIFS's own ice fraction `ci`. The product
  `prsn · ci` is taken on the monthly means, before the annual mean (`check_run.compute_atm_clim`, variable
  `snow_ice_oce`), so correlations within a month are missed. NEMO splits with its own fraction for snow on ice,
  which includes blowing snow (`rn_snwblow` = 0.66), so the split is not the ice fraction alone.

  The mismatch (`snow_res`, +0.017 W m⁻² in xax0) is mostly **snow mass lost in the bilinear remapping**:
  - the snowfall NEMO receives (`snowpre`) is 0.96 of OIFS `prsn` over its ocean points (xax0, 1855 and 1852;
    total precipitation: 0.978);
  - −L_f × the lost mass predicts +0.025 W m⁻² per ocean m², against +0.023 measured;
  - the snow's sensible heat (c_p·T_snow, which NEMO adds and OIFS neglects) is small and of the other sign
    (≈ −0.003).
- **Water rows (lilac).** Evaporation, rain and runoff are only mass on the OIFS side: OIFS and HTESSEL give the
  water no heat content. NEMO books their heat content c_p·T, so the whole NEMO value appears as the mismatch. This
  is bookkeeping by one component, not an error of either (section 5).
- **Calving row.** The calving latent heat is set equal on both sides (≡ 0, assumed). The calving mass is remapped
  bilinearly, without conservation, so the true mismatch is not zero. It cannot be measured, because OIFS does not
  write the calving it sends. The comparison of NEMO's `runoffs` + `calving_cea` with OIFS `rnf` suggests a loss of
  under 1 %, about 0.003 W m⁻². Whatever it is, it ends up in ε_HTESSEL.
- **Runoff** enters NEMO at max(SST, 0 °C) (`sbcrnf.F90` l. 156), through a separate term (section 5).

**Checks:**
- identity: `snow_ice_oifs + snow_ow_oifs − snow_melt = 0`
- identity (in): `band_in − F_oifs_nosn − snow_ice_oifs − snow_ow_oifs − cal_latent = 0`
- identity (out): `band_out − oasis_recv − qemp_ice − snow_ow − evap_rain_ow − cal_latent − runoff_heat = 0`
- identity (rows): `band_res − oasis_res − snow_res_ice − snow_res_ow − evap_rain_ow − runoff_heat = 0`
- identity: `band_out − band_in − band_res = 0`
- identity: `snow_melt + snow_res − qemp_ice − snow_ow = 0`
- identity: `qemp_oce − evap_rain_ow − snow_ow = 0`
- real (tolerance 0.001): `qemp_oce + hflx_evap_cea − hflx_rain_cea − hflx_snow_cea + qemp_ice = 0`, i.e. NEMO's
  E − P heat flux against its separate diagnostics (`hflx_evap_cea` is written positive).

### 3.3 NEMO / SI3

Sea ice (SI3) and liquid ocean (NEMO) sit in one frame. The rows leaving the OASIS band end on two collectors, and
one summed arrow enters each box:

| quantity | term | formula |
|---|---|---|
| into the sea ice | `qt_ai` † | `⟨qt_atm_oi⟩_N − ⟨qt_oce⟩_N` (= `recv_ice + qemp_ice`) |
| into the ocean | `qt_ow + runoff_heat` | `⟨qt_oce⟩_N + runoff_heat` (= `recv_ow + snow_ow + evap_rain_ow + cal_latent + runoff_heat`) |
| ice → liquid ocean (melting, freezing, …) | `ice_to_oce` † | `⟨qt_oce_ai⟩_N − ⟨qt_oce⟩_N` |
| heat reaching the liquid ocean | `qt_oce_ai` | `⟨qt_oce_ai⟩_N` |
| geothermal (from below) | `F_geo` | Goutorbe map (mW m⁻²) bilinearly interpolated on the NEMO T points, `⟨·⟩_N / 1000`; 0.0656 W m⁻² per ocean m² (0.0459 global) |
| SI3 storage | `ice_storage` † | `⟨qt_atm_oi⟩_N − ⟨qt_oce_ai⟩_N` |
| NEMO OHC tendency | `ohc_trend` | `(H_y − H_(y−1)) / (365.25 × 86400 s) / A_N`, with `H_y = Σ area × mask × heatc` (annual mean of year y; the first year has no value) |
| **heat uptake** | `heat_uptake` | `ice_storage + ohc_trend` |
| **ε_NEMO/SI3** | `oce_res` | `heat_uptake − (⟨qt_atm_oi⟩_N + runoff_heat + F_geo) = ohc_trend − qt_oce_ai − runoff_heat − F_geo` |

- **`heatc` is the liquid ocean only.** It does not include the sea-ice enthalpy, so the uptake adds the SI3
  storage.
- **The SI3 box closes by construction.** No sea-ice enthalpy is written, so the storage is the box's residual, and
  ε_NEMO/SI3 is in practice the liquid ocean's.
- **The ice → ocean arrow** has a fixed direction (SI3 → NEMO). Its sign is given by its colour, like every other
  arrow.

**Checks:**
- identity: `recv_ice + qemp_ice − ice_to_oce − ice_storage = 0`; with NEMO fields only:
  `qt_ai − ice_to_oce − ice_storage = 0`
- identity: `recv_ow + snow_ow + evap_rain_ow + cal_latent + runoff_heat + ice_to_oce + F_geo − ohc_trend + oce_res = 0`
- identity: `ice_storage + ohc_trend − heat_uptake = 0`

### 3.4 Total

What enters the system at the top (TOA) and from below (geothermal), minus what the ice and the ocean store, is the
energy created inside. It splits into the three frames, and further into each residual (global units):

```text
total_res = heat uptake − TOA − geothermal          (standard output only)
          = oifs_res + band_res + oce_res           (the three frames: OIFS / HTESSEL, OASIS, NEMO / SI3)
          = atm_imb         OIFS: surface − TOA
          + land_res        HTESSEL: calving out − land flux in (storage ≈ 0)
          + oasis_res       OASIS heat: what NEMO receives − what OIFS gives
          + snow_res        OASIS snow: what NEMO books − what OIFS drops
          + water_cycle     water heat content booked by NEMO only (evaporation, rain, runoff)
          + oce_res         NEMO / SI3: heat uptake − inputs
```

`total_res = (ice_storage + ohc_trend − F_geo) × A_N / A_E − toa`. With standard output only it is still exact, split
as `atm_imb + band_res_std + oce_imb` (section 6).

| W m⁻², global mean, years 2–10 | xax0 | ACX0 |
|---|---|---|
| ε_OIFS `atm_imb` | −0.377 | −0.067 |
| ε_HTESSEL `land_res` | −0.028 | −0.013 |
| OASIS heat `oasis_res` | +0.210 | +0.184 |
| OASIS snow `snow_res` (on ice / on open water) | +0.017 (+0.013 / +0.004) | +0.016 (+0.013 / +0.004) |
| water heat content `water_cycle` | −0.124 | −0.111 |
| ε_NEMO/SI3 `oce_res` | −0.052 | −0.036 |
| **total ε `total_res`** | **−0.354** | **−0.027** |
| ε of the OIFS / HTESSEL frame `oifs_res` | −0.405 | −0.080 |
| ε of the OASIS band `band_res` | +0.103 | +0.089 |

**Checks** (real, tolerance 0.001):
- `total_res − oifs_res − band_res − oce_res = 0`
- `total_res − atm_imb − band_res_std − oce_imb = 0`
- `total_res − atm_imb − end_to_end + srf_land_glob = 0`
- `total_res − atm_imb − land_res − oasis_res − snow_res − water_cycle − oce_res = 0`

All four differ from 0 by the surface-split difference (+0.0003), so they test the same thing.

### 3.5 Other terms (tables and time series)

| quantity | term | formula |
|---|---|---|
| water fluxes in `qemp` without snowfall melting | `qemp_sens` (= `nemo_added`) | `qemp − snow_melt = evap_rain_ow + snow_res` |
| water heat content booked by NEMO only | `water_cycle` | `evap_rain_ow + runoff_heat` |
| coupling residual, all | `interface_imb` | `⟨qt_atm_oi⟩_N − F_oifs = oasis_res + qemp_sens + cal_latent` |
| ocean residual, with runoff heat | `oce_imb` | `ohc_trend − qt_oce_ai − F_geo = oce_res + runoff_heat` |
| end to end (OIFS ocean flux → heat content) | `end_to_end` | `interface_imb + oce_imb = oasis_res + cal_latent + snow_res + water_cycle + oce_res` |
| ocean chain without the calving | `e2e_ocean` | `oasis_res + snow_res + water_cycle + oce_res` |
| OASIS band, standard output | `band_in_std`, `band_res_std` | `F_oifs + land_in`; `⟨qt_atm_oi⟩_N − band_in_std` |

**Checks (identities):**
- `F_oifs + interface_imb + runoff_heat + F_geo − heat_uptake + oce_res = 0`
- `end_to_end − oasis_res − cal_latent − snow_res − water_cycle − oce_res = 0`

### 3.6 Derived terms (marked † in the figure)

These numbers have no output field of their own: each is a difference of other terms. They are not budget
residuals like the ε badges, but they rely on how the other fields are defined.

| term | computed as | what it assumes |
|---|---|---|
| `ice_storage` | `qt_atm_oi − qt_oce_ai` | Sea ice stores whatever enters and does not reach the ocean. This is the residual of the SI3 box, so the box closes **by construction** and any SI3 conservation error ends up in the storage. |
| `qt_ai` | `qt_atm_oi − qt_oce` | the part over sea ice = total minus open water |
| `ice_to_oce` | `qt_oce_ai − qt_oce` | what reaches the liquid ocean beyond the open-water flux comes through the ice |
| `oasis_recv` | `qt_atm_oi − qemp − cal_latent` | NEMO's total minus what NEMO adds itself |
| `recv_ice`, `recv_ow` | `qt_ai − qemp_ice`; `qt_oce − qemp_oce − cal_latent` | the same, over ice and open water |
| `snow_ow` | `hflx_snow_cea − qemp_ice` | snowfall heat = part on ice + part on open water |
| `evap_rain_ow` | `qemp_oce − snow_ow` | `qemp_oce` = evaporation + rain + snow on open water |
| `snow_melt` | `F_oifs − F_oifs_nosn` | the two OIFS surface fluxes differ only by `L_f · prsn` |
| `snow_ow_oifs` | `snow_melt − snow_ice_oifs` | OIFS snowfall = on its ice + on open water |

Every term without † is an output field, or a sum or area weighting of output fields. The ε badges are the
residuals: `atm_imb`, `land_res`, the row mismatches and `band_res`, `oce_res`, `total_res`.

---

## 4. Closure checks

Every box must close: inputs − outputs − storage = 0, or, for a box with a residual, = −residual (the residual is
then one of the terms of the check). In code: `energy_budget.CHECKS` and `budget_checks(table)`, shown in the
notebook after the budget table.

There are two kinds of check:
- **Identities** (tolerance 10⁻⁶ W m⁻²) are true by construction. They test that the code, the definitions and this
  document agree, and that all terms are averaged over the same years. This is how a water heat cache one year
  behind was found.
- **Real checks** compare independent estimates (larger tolerance). There are two: "2 Surface split" (two
  weightings of the same OIFS fields) and "3 E−P from diagnostics" (NEMO's E − P heat flux against its separate
  diagnostics). The four "Total" checks only repeat the surface-split difference.

| box | check (= 0) | tolerance (W m⁻²) | xax0 | ACX0 |
|---|---|---|---|---|
| 1 Atmosphere | `toa − srf_net + atm_imb` | 10⁻⁶ | ok | ok |
| 2 Surface split | `srf_net − srf_oce_glob − srf_land_glob` | 0.005 (real) | −0.0003 | −0.0003 |
| 2 Snowfall taken out | `F_oifs − F_oifs_nosn − snow_melt` | 10⁻⁶ | ok | ok |
| 3 OASIS, OIFS snowfall | `snow_ice_oifs + snow_ow_oifs − snow_melt` | 10⁻⁶ | ok | ok |
| 3 OASIS, in | `band_in − F_oifs_nosn − snow_ice_oifs − snow_ow_oifs − cal_latent` | 10⁻⁶ | ok | ok |
| 3 OASIS, out | `band_out − oasis_recv − qemp_ice − snow_ow − evap_rain_ow − cal_latent − runoff_heat` | 10⁻⁶ | ok | ok |
| 3 OASIS, rows | `band_res − oasis_res − snow_res_ice − snow_res_ow − evap_rain_ow − runoff_heat` | 10⁻⁶ | ok | ok |
| 1 OIFS / HTESSEL | `oifs_res − atm_imb − land_res` | 10⁻⁶ | ok | ok |
| 3 OASIS, sum | `band_out − band_in − band_res` | 10⁻⁶ | ok | ok |
| 3 E−P from diagnostics | `qemp_oce + hflx_evap_cea − hflx_rain_cea − hflx_snow_cea + qemp_ice` | 0.001 (real) | 0.0000 | 0.0000 |
| 3 E−P heat box | `qemp_oce − evap_rain_ow − snow_ow` | 10⁻⁶ | ok | ok |
| 4 Sea ice | `recv_ice + qemp_ice − ice_to_oce − ice_storage` | 10⁻⁶ | ok | ok |
| 4 Sea ice, NEMO fields | `qt_ai − ice_to_oce − ice_storage` | 10⁻⁶ | ok | ok |
| 4 Liquid ocean | `recv_ow + snow_ow + evap_rain_ow + cal_latent + runoff_heat + ice_to_oce + F_geo − ohc_trend + oce_res` | 10⁻⁶ | ok | ok |
| 5 Ice + ocean | `ice_storage + ohc_trend − heat_uptake` | 10⁻⁶ | ok | ok |
| 2 Snowfall, OIFS vs NEMO | `snow_melt + snow_res − qemp_ice − snow_ow` | 10⁻⁶ | ok | ok |
| Whole chain | `F_oifs + interface_imb + runoff_heat + F_geo − heat_uptake + oce_res` | 10⁻⁶ | ok | ok |
| Land | `srf_land_glob − cal_latent + land_res` | 10⁻⁶ | ok | ok |
| End to end | `end_to_end − oasis_res − cal_latent − snow_res − water_cycle − oce_res` | 10⁻⁶ | ok | ok |
| Total, standard output | `total_res − atm_imb − end_to_end + srf_land_glob` | 0.001 (real) | +0.0003 | +0.0003 |
| Total, three boxes | `total_res − oifs_res − band_res − oce_res` | 0.001 (real) | +0.0003 | +0.0003 |
| Total, three boxes, standard output | `total_res − atm_imb − band_res_std − oce_imb` | 0.001 (real) | +0.0003 | +0.0003 |
| Total, all residuals | `total_res − atm_imb − land_res − oasis_res − snow_res − water_cycle − oce_res` | 0.001 (real) | +0.0003 | +0.0003 |

Values for years 2–10 (xax0 1851–1859, ACX0 1951–1959), in global units. The checks that mix regions (land, total)
hold only there, so `budget_checks` requires a table with `units = 'global'`. "1 Atmosphere" is at 5·10⁻⁷ because
the fields are float32: little margin below 10⁻⁶.

Real checks that could be added with standard output:
- snowfall mass, NEMO `snowpre` against OIFS `prsn` over its ocean points (0.96 in xax0);
- runoff and calving mass, NEMO `runoffs + calving_cea` against OIFS `rnf`.

---

## 5. Why these variables

Each choice below was made from the model code, not from the result; section 6 shows that the end-to-end residual
is not forced to zero.

**Totals `qt_*` instead of the solar and non-solar pairs.** `qt_oce = qsr_oce + qns_oce` and
`qt_atm_oi = qns_tot + qsr_tot` (`iceupdate.F90`). SI³ builds the totals with the ice fraction it actually uses
(`at_i_b`) and with `qemp` already in place, so using them avoids adding pairs by hand and applying the ice fraction
or `qemp` twice. A budget needs only the total; the split would only say whether a residual is solar or not.

**`qt_atm_oi` for what NEMO receives.** It is the total atmosphere → ice + ocean flux over the whole cell,
`qns_tot + qsr_tot`, which NEMO builds from what OASIS delivers (`sbccpl.F90`, `sbc_cpl_ice_flx`). It contains the
calving latent heat (l. 2027) and `qemp_oce + qemp_ice` (l. 2061), each once.

**`qt_oce_ai`, not `qt_oce`, for the liquid ocean.** `qt_oce` is only the atmosphere → open-water flux. It misses the
heat exchanged between ice and ocean under the ice: bottom melt and growth, freezing of open water.
`qt_oce_ai = qt_atm_oi − (what the ice uses or releases)` is the flux that reaches the liquid ocean; NEMO uses it as
its ocean forcing (`qns = qt_oce_ai − qsr`, `iceupdate.F90` l. 158).

**`qt_oce` only for the open-water / ice split.** `qns_oce` is computed (l. 2035) before `qemp` is added to the total
(l. 2061), so `qt_oce` adds `qemp_oce` explicitly: it holds the calving latent heat and `qemp_oce` once each. Hence
`recv_ow = qt_oce − qemp_oce − cal_latent` and `recv_ice = qt_atm_oi − qt_oce − qemp_ice`.

**`qt_atm_oi` and `qt_oce` for NEMO's side, not `qt_ice`.** What reaches the sea ice is `qt_atm_oi − qt_oce`, a
difference of two output fields. `qt_ice` is not used because it is out of step with `qt_atm_oi`:
- `qt_ice` is computed with `qns_ice` after the dQ/dT correction to the SI3 surface temperature
  (`icethd_zdf_bl99.F90` l. 382), while `qt_atm_oi` keeps the flux as received (`iceupdate.F90` l. 111).
- SI3 computes `qt_oce_ai` from `qt_atm_oi` (l. 136), so `qt_atm_oi − qt_oce` is the input consistent with
  `ice_storage` and `ice_to_oce`.
- `qt_ice + qt_oce` exceeds `qt_atm_oi` by 0.04–0.05 W m⁻² per ocean m² in xax0. This is the dQ/dT correction,
  which the `hfxerr` diagnostic only partly includes, and SI3 takes it back out of the ocean.

**`heatc` for the ocean heat content.** The vertically integrated heat content gives the full-depth change in one 2D
field. Only monthly means exist, so the tendency is the difference of consecutive annual means.

**Surface flux with snowfall melting (`− L_f · prsn`) for the atmosphere.** Snow leaves the atmosphere with an energy
deficit `L_f` per kg that the surface must supply: without this term `atm_imb` would be off by about 0.7 W m⁻².

**Surface flux without snowfall melting for the OASIS heat row.** OASIS carries only the radiative and turbulent
fluxes; NEMO adds the snowfall itself, inside `qemp` (`snow × (c_p·T − L_f)`). Comparing with the flux with snowfall
melting would count it twice. The snowfall crosses the band in its own rows.

**OASIS areas and masks for the OIFS ocean / land split.** They are the land–sea masks and cell areas of the coupled
grids, so the split matches the coupling geometry. They are not conservation weights: the remapping is bilinear
(section 3.2). The global mean keeps the cos(latitude) weights of `check_run`; the two differ by 0.0003 W m⁻² (check
"2 Surface split").

**Per NEMO ocean area from the coupling downward.** NEMO's fields are naturally per unit NEMO cell area. OIFS's
ocean flux is rescaled by `A_o / A_N` so that the two integrals are compared: the land–sea masks differ and the OIFS
ocean is 2 % larger on TL63–ORCA2.

**`qemp_oce` and `qemp_ice` (⚠ for `qemp_oce`).** They are what NEMO adds to the flux it receives; without them what
OASIS delivered cannot be separated from what NEMO adds. `qemp_ice` alone (standard) is not enough and is not used
without `qemp_oce`.

**What the water heat content is, and what it is not.** The latent heat of evaporation and condensation is in OIFS:
the surface pays it through `hfls` = L_v · E and the atmosphere gets it back when the vapour condenses. It is not in
`qemp`. What NEMO books is the heat content c_p·T, relative to 0 °C, of the liquid water that leaves and enters the
ocean (`sbccpl.F90` l. 1791, 2042, 2053; `sbcrnf.F90`):

- evaporation: `−E · c_p · SST`;
- rain: `+P · c_p · SST` over open water (ice temperature over ice): NEMO assigns the local SST, OASIS passes no
  rain temperature;
- snow: `snow · (c_p · T_snow − L_f)`, whose `− L_f` part is the snowfall melting already counted by OIFS
  (`− L_f · prsn`): it crosses the band in the snow rows;
- runoff: `+R · c_p · max(SST, 0 °C)` at the river mouths, outside the heat flux;
- calving: `− L_f` per kg of calved ice.

NEMO has to book these because its heat content `heatc` is ρ c_p T over a volume that changes with E − P − R. OIFS's
energy budget does not carry the heat content of the water it exchanges (as far as known, IFS neglects the sensible
heat of precipitation and water vapour). Over the ocean the water mass about balances (E ≈ P + R), so the 0 °C
reference drops out. `water_cycle` (−0.124 / −0.111 W m⁻² global in xax0 / ACX0, i.e. −0.177 / −0.159 per ocean m²)
says that the water leaves warm, evaporating at the SST of the evaporation regions, and comes back colder, as rain
and runoff at colder local SSTs. In NEMO's bookkeeping the water cycle therefore cools the ocean model. It is a heat
flux booked by one component and not by the other, not an error of either, and it is part of the OASIS band.

**`hflx_rnf_cea` for the runoff heat.** With `ln_rnf_tem = .false.` runoff enters at max(SST, 0 °C) through a
separate term (`rnf_tsc`, `sbcrnf.F90`), not through the heat flux: it shows up in `heatc` but not in `qt_oce_ai`.

**Calving: a flux from the OIFS land to the ocean.** The runoff mapper moves the excess snow of the glaciers from the
OIFS land to NEMO as calving. NEMO subtracts its latent heat from `qns_tot` (`sbccpl.F90` l. 2027–2030), so it is
part of `qt_atm_oi`. It is not heat booked by NEMO only: it is the counterpart of the snowfall melting
(`− L_f · prsn`) that OIFS counts over the land. It is drawn as an arrow from HTESSEL through the OASIS band to the
ocean, is not part of `water_cycle`, and its imbalance is in ε_HTESSEL.

**No heat storage on the land.** Checked with the 1 January snapshots of the OIFS surface state (`saveic_sfc`, written
in xc09 and AC01, the same simulations as xax0 and ACX0). They cover:
- soil temperature (4 layers, HTESSEL heat capacity);
- the latent heat of soil freezing (HTESSEL's frozen fraction of the field-capacity water);
- snow (−L_f × water equivalent, plus its sensible heat).

Over 1951–1959 (AC01) and 1851–1859 (xc09) the change is +0.003 and +0.004 W m⁻² per land m², and the mean snow water
equivalent stays at 1069 kg m⁻². The monthly snow mass (`sd`) confirms it (section 3.1): the glaciers export their
snowfall, they do not store it. Lakes (FLake) are not included.

**Global units by default.** As contributions to the global mean, all the terms are on the same footing and the total
budget can be closed by summing them; per-region values (option `units = 'region'`) are better for comparing regions
of different size but cannot be added.

**Goutorbe map averaged over the NEMO ocean for `F_geo`.** It is the map NEMO applies at the sea floor. The map
covers land too, so its global mean divided by 0.7 (91.8 mW m⁻², used before) is wrong: the NEMO ocean mean is
65.6 mW m⁻². Runs without the file in their run directory use the ECE4-DATA copy, with a warning, instead of a fixed
value.

**Hemispheric sums.** SI³ fields are integrated per hemisphere; the equator row (`nav_lat` = 0 exactly on ORCA2 and
PALEORCA2) is assigned to the north, so that no row is lost.

**Equal month weights in the annual means.** They are what the saved means of `check_run` use, so all terms are
consistent with the rest of the analysis; the effect of month-length weights is given in section 6.

**Same years for every term.** `budget_terms` keeps only the years that every source has (OIFS, NEMO and SI³ means,
water heat terms). Averaging terms over different years breaks the identities.

---

## 6. Runs with standard output only, and caveats

**Without the ⚠ fields** (AC01, aI09), the terms that need `qemp_oce` or the `hflx_*` fields are n/a:
- `qemp`, `cal_latent`, `runoff_heat`, `snow_ow`, `evap_rain_ow`;
- `snow_res`, `recv_ow`, `recv_ice`, `oasis_recv`, `oasis_res`;
- `land_res`, `oifs_res`, `water_cycle`, `oce_res`, `band_res`.

The diagram then shows (marked `*`):
- **OIFS / HTESSEL:** ε_OIFS; ε_HTESSEL is n/a.
- **OASIS band:** the OIFS column (heat, snow on ice and on open water) and only the sum,
  `band_res_std = ⟨qt_atm_oi⟩_N − (F_oifs + land_in)`. The land flux is taken as calving, so the sum includes
  ε_HTESSEL.
- **NEMO / SI3:** `oce_imb`, which includes the runoff heat. The arrow into the ocean is `qt_oce` alone, without the
  runoff heat.

`atm_imb`, `total_res` and `end_to_end` need only standard output.

Standard output does contain `calving_cea`, `snowpre`, `runoffs` and `qemp_ice`. With `cal_latent = −L_f ×
calving_cea` (exact in xax0), ε_HTESSEL, `oifs_res` and the snow-on-ice row could be computed for these runs too;
this is not implemented yet.

Caveats:

- **Month weights.** The annual means weight all months equally (`check_run._annual`), so February counts as much as
  January. An independent recomputation from the raw output gives the same numbers with equal weights. With weights
  proportional to the month length, `F_oifs`, `qt_atm_oi` and `qt_oce_ai` are lower by 0.05–0.06 W m⁻² per ocean m²,
  `ohc_trend` is unchanged, and `end_to_end` moves by about +0.06 W m⁻² per ocean m² (AC01 and xc09).
- **Sampling.** The yearly `end_to_end` varies by ±0.3 W m⁻² per ocean m² (−0.36 to +0.26 in AC01): the OHC
  tendency comes from consecutive annual means (mid-year to mid-year), the fluxes from calendar years. Over a window
  the differences telescope, but a 9-year mean is good to a few 0.01 W m⁻² only.
- **Not forced to zero.** `end_to_end` uses only standard output: runs with and without the water heat terms give
  the same value (AC01 = ACX0, the same simulation), and other runs do not close (per ocean m²: aI09, TL255–eORCA1,
  −0.40, i.e. −0.28 global; AA05 and AA06, PALEORCA2 OIFS 3600 / NEMO 3600 s, −0.07 to −0.12).
- **OASIS remapping.** All fields are remapped bilinearly without conservation (section 3.2). This affects the heat
  row (`oasis_res`), the snow rows (`snow_res`, mostly lost snow mass), and the calving, whose loss is folded into
  ε_HTESSEL.
- **Geothermal interpolation.** `F_geo` uses a bilinear interpolation of the 1° map; NEMO uses its own weights file
  (`weights_Goutorbe1_2_<grid>_bilinear.nc`). The difference is expected to be below 0.001 W m⁻².
- **Sea-ice storage.** It is computed from the fluxes, so the SI3 box closes by construction. An independent check of
  SI³ would need the ice and snow heat content in the output.
