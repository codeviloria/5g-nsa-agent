"""
5G NSA (EN-DC, option 3x) troubleshooting tools for an LLM agent.

- `nr_link_budget` uses a real propagation model (3GPP TR 38.901 UMa).
- `get_endc_kpis`, `audit_endc_config` and `check_nr_pci_conflicts` read MOCKED OSS data
  (PM counters and configuration): fictional LTE anchors and n78 gNB cells.
- `ssb_beam_report` reads MOCKED beam-level measurements. In a real network these come
  from drive-test scanners or L3 measurement reports with per-SSB results, not from
  per-cell PM counters.

Every tool returns a plain dict so it can be serialized to JSON and handed back
to the model as a `tool` message.
"""
from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Mock network (fictional). LTE anchors (MeNB) + NR n78 cells (SgNB).
# ---------------------------------------------------------------------------
LTE_ANCHORS: dict[str, dict] = {
    "BAQ_021_A": {
        "band": "B4", "endc_enabled": True, "x2_status": "up",
        # measObjectNR configured on the anchor (what the UE is told to measure)
        "meas_nr": {"ssb_nr_arfcn": 630000, "ssb_scs_khz": 30,
                    "smtc": {"periodicity_ms": 20, "offset_ms": 0, "duration_ms": 5},
                    "b1_threshold_dbm": -105},
        "nr_neighbors": ["BAQ_021_N78_A"],
        "kpis": {"endc_capable_ue_pct": 64.0, "b1_reports_per_1000_ue": 3,
                 "sgnb_add_attempts": 41, "sgnb_add_success_pct": 97.6,
                 "scg_failure_pct": 0.9, "endc_time_ratio_pct": 2.1,
                 "nr_dl_thp_mbps_when_on_nr": 310},
    },
    "BAQ_034_A": {
        "band": "B4", "endc_enabled": True, "x2_status": "up",
        "meas_nr": {"ssb_nr_arfcn": 630000, "ssb_scs_khz": 30,
                    "smtc": {"periodicity_ms": 20, "offset_ms": 10, "duration_ms": 5},
                    "b1_threshold_dbm": -124},
        "nr_neighbors": ["BAQ_034_N78_A"],
        "kpis": {"endc_capable_ue_pct": 61.0, "b1_reports_per_1000_ue": 820,
                 "sgnb_add_attempts": 5400, "sgnb_add_success_pct": 96.1,
                 "scg_failure_pct": 7.8, "endc_time_ratio_pct": 38.0,
                 "nr_dl_thp_mbps_when_on_nr": 145},
    },
}

NR_CELLS: dict[str, dict] = {
    "BAQ_021_N78_A": {
        "gnb": "BAQ_021", "band": "n78", "absolute_frequency_ssb": 630000, "ssb_scs_khz": 30,
        "ssb_periodicity_ms": 20, "ssb_burst_offset_ms": 10, "pci": 120, "l_max": 8,
        "height_m": 25, "neighbors": ["BAQ_034_N78_A", "BAQ_022_N78_B"],
        "beams_source": "drive-test scanner (SSB index decoded; few UEs attach on this anchor)",
        "beams": [(-86, 14.2, 9), (-84, 15.0, 14), (-83, 16.1, 18), (-85, 15.4, 17),
                  (-88, 12.9, 15), (-90, 11.8, 12), (-93, 9.7, 9), (-95, 8.8, 6)],
    },
    "BAQ_034_N78_A": {
        "gnb": "BAQ_034", "band": "n78", "absolute_frequency_ssb": 630000, "ssb_scs_khz": 30,
        "ssb_periodicity_ms": 20, "ssb_burst_offset_ms": 10, "pci": 123, "l_max": 8,
        "height_m": 30, "neighbors": ["BAQ_021_N78_A", "BAQ_022_N78_B"],
        "beams_source": "L3 measurement reports with per-SSB results (rsIndexResults)",
        "beams": [(-92, 10.1, 11), (-90, 11.5, 15), (-91, 10.8, 16), (-94, 8.2, 13),
                  (-99, 5.1, 12), (-104, 2.0, 15), (-112, -1.8, 10), (-115, -2.6, 8)],
    },
    "BAQ_022_N78_B": {
        "gnb": "BAQ_022", "band": "n78", "absolute_frequency_ssb": 630000, "ssb_scs_khz": 30,
        "ssb_periodicity_ms": 20, "ssb_burst_offset_ms": 10, "pci": 301, "l_max": 8,
        "height_m": 22, "neighbors": ["BAQ_021_N78_A", "BAQ_034_N78_A"],
        "beams_source": "L3 measurement reports with per-SSB results (rsIndexResults)",
        "beams": [(-89, 12.0, 12)] * 8,
    },
}

N_RB_30KHZ = {20: 51, 40: 106, 50: 133, 60: 162, 80: 217, 100: 273}


def _resolve(kind: str, cell_id: str, known: dict) -> tuple[str, dict | None]:
    """Normalize an ID. A site ID that matches exactly one cell (BAQ_021 -> BAQ_021_A) is resolved;
    otherwise the error tells the model exactly how to retry."""
    cid = cell_id.strip().upper()
    if cid in known:
        return cid, None
    matches = sorted(k for k in known if k.startswith(cid + "_"))
    if len(matches) == 1:
        return matches[0], None
    hint = (f"Did you mean one of {matches}? " if matches else "") + \
        f"Call the tool again with an exact {kind} ID from: {sorted(known)}"
    return cid, {"error": f"Unknown {kind} '{cell_id}'. {hint}"}


# ---------------------------------------------------------------------------
# Physics: 3GPP TR 38.901 Urban Macro path loss
# ---------------------------------------------------------------------------
def _uma_path_loss(d2d_m: float, h_bs: float, h_ut: float, fc_ghz: float, los: bool) -> float:
    d3d = math.hypot(d2d_m, h_bs - h_ut)
    d_bp = 4 * (h_bs - 1.0) * (h_ut - 1.0) * fc_ghz * 1e9 / 3e8   # effective env. height 1 m
    if d2d_m <= d_bp:
        pl_los = 28.0 + 22 * math.log10(d3d) + 20 * math.log10(fc_ghz)
    else:
        pl_los = (28.0 + 40 * math.log10(d3d) + 20 * math.log10(fc_ghz)
                  - 9 * math.log10(d_bp ** 2 + (h_bs - h_ut) ** 2))
    if los:
        return pl_los
    pl_nlos = 13.54 + 39.08 * math.log10(d3d) + 20 * math.log10(fc_ghz) - 0.6 * (h_ut - 1.5)
    return max(pl_los, pl_nlos)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
def nr_link_budget(distance_m: float, bs_height_m: float = 25.0, freq_ghz: float = 3.5,
                   los: bool = False, tx_power_dbm: float = 53.0,
                   ssb_beam_gain_dbi: float = 18.0, bandwidth_mhz: int = 100,
                   indoor: bool = False, b1_threshold_dbm: float = -105.0) -> dict:
    """Estimate n78 SS-RSRP at a distance (3GPP TR 38.901 UMa) and whether it passes the B1 threshold.

    Args:
        distance_m: 2D distance from the gNB to the user in meters (10 to 5000).
        bs_height_m: gNB antenna height in meters.
        freq_ghz: Carrier frequency in GHz (3.5 for n78).
        los: True for line-of-sight, False for non-line-of-sight (typical urban).
        tx_power_dbm: Total gNB transmit power in dBm (53 dBm = 200 W).
        ssb_beam_gain_dbi: Antenna gain of the SSB broadcast beam in dBi.
        bandwidth_mhz: NR carrier bandwidth in MHz at 30 kHz SCS (20, 40, 50, 60, 80 or 100).
        indoor: True to add 20 dB outdoor-to-indoor penetration loss.
        b1_threshold_dbm: B1 threshold configured on the LTE anchor, in dBm.
    """
    if not 10 <= distance_m <= 5000:
        return {"error": "distance_m must be between 10 and 5000"}
    n_rb = N_RB_30KHZ.get(bandwidth_mhz)
    if not n_rb:
        return {"error": f"bandwidth_mhz must be one of {sorted(N_RB_30KHZ)}"}
    pl = _uma_path_loss(distance_m, bs_height_m, 1.5, freq_ghz, los)
    epre = tx_power_dbm - 10 * math.log10(12 * n_rb)          # SSB energy per resource element
    o2i = 20.0 if indoor else 0.0
    ss_rsrp = epre + ssb_beam_gain_dbi - pl - o2i
    return {
        "model": f"3GPP TR 38.901 UMa {'LOS' if los else 'NLOS'}",
        "path_loss_db": round(pl, 1),
        "ssb_epre_dbm": round(epre, 1),
        "ss_rsrp_dbm": round(ss_rsrp, 1),
        "passes_b1": ss_rsrp >= b1_threshold_dbm,
        "margin_to_b1_db": round(ss_rsrp - b1_threshold_dbm, 1),
        "assumptions": f"UE 1.5 m, O2I {o2i:.0f} dB, no shadow fading margin",
    }


def get_endc_kpis(anchor_cell_id: str) -> dict:
    """Get daily EN-DC (5G NSA) KPIs for an LTE anchor cell (mock OSS data).

    KPIs: endc_capable_ue_pct (share of UEs that support EN-DC), b1_reports_per_1000_ue
    (NR measurement reports received), sgnb_add_attempts / sgnb_add_success_pct
    (SgNB addition), scg_failure_pct (secondary cell group failures), endc_time_ratio_pct
    (share of time EN-DC UEs are actually on NR), nr_dl_thp_mbps_when_on_nr.

    Args:
        anchor_cell_id: LTE anchor cell identifier, e.g. "BAQ_021_A".
    """
    anchor_cell_id, err = _resolve("anchor", anchor_cell_id, LTE_ANCHORS)
    if err:
        return err
    cell = LTE_ANCHORS[anchor_cell_id]
    return {"anchor_cell_id": anchor_cell_id.upper(), "nr_neighbors": cell["nr_neighbors"],
            "kpis": cell["kpis"]}


def audit_endc_config(anchor_cell_id: str) -> dict:
    """Audit the anchor's NR measurement config against the real SSB config of its NR neighbors.

    Checks SSB frequency (NR-ARFCN), SSB subcarrier spacing, SMTC window vs SSB burst
    timing, B1 threshold range and X2 status.

    Args:
        anchor_cell_id: LTE anchor cell identifier, e.g. "BAQ_021_A".
    """
    anchor_cell_id, err = _resolve("anchor", anchor_cell_id, LTE_ANCHORS)
    if err:
        return err
    cell = LTE_ANCHORS[anchor_cell_id]
    m, findings = cell["meas_nr"], []
    smtc = m["smtc"]

    for nr_id in cell["nr_neighbors"]:
        nr = NR_CELLS[nr_id]
        if nr["absolute_frequency_ssb"] != m["ssb_nr_arfcn"]:
            findings.append({"check": "ssb_frequency", "nr_cell": nr_id, "severity": "critical",
                             "anchor_value": m["ssb_nr_arfcn"], "gnb_value": nr["absolute_frequency_ssb"],
                             "impact": "UE searches the wrong frequency: no B1, no SgNB addition"})
        if nr["ssb_scs_khz"] != m["ssb_scs_khz"]:
            findings.append({"check": "ssb_scs", "nr_cell": nr_id, "severity": "critical",
                             "anchor_value": m["ssb_scs_khz"], "gnb_value": nr["ssb_scs_khz"]})
        # SSB bursts occur at burst_offset + k * ssb_periodicity; SMTC window = [offset, offset + duration)
        delta = (nr["ssb_burst_offset_ms"] - smtc["offset_ms"]) % nr["ssb_periodicity_ms"]
        window_ok = delta < smtc["duration_ms"]
        period_ok = smtc["periodicity_ms"] % nr["ssb_periodicity_ms"] == 0
        if not (window_ok and period_ok):
            findings.append({
                "check": "smtc_alignment", "nr_cell": nr_id, "severity": "critical",
                "anchor_smtc": smtc,
                "gnb_ssb": {"periodicity_ms": nr["ssb_periodicity_ms"],
                            "burst_offset_ms": nr["ssb_burst_offset_ms"]},
                "impact": "SMTC window misses the SSB burst: UE cannot measure NR, B1 rarely triggers",
                "fix": {"smtc_offset_ms": nr["ssb_burst_offset_ms"] % smtc["periodicity_ms"]},
            })

    b1 = m["b1_threshold_dbm"]
    if b1 < -115:
        findings.append({"check": "b1_threshold", "severity": "major", "value_dbm": b1,
                         "direction": "too low (too permissive)",
                         "impact": "Threshold too low: SgNB added at weak NR coverage -> SCG failures",
                         "fix": "Raise toward -110..-105 dBm (typical starting range; tune with drive tests)"})
    elif b1 > -95:
        findings.append({"check": "b1_threshold", "severity": "minor", "value_dbm": b1,
                         "direction": "too high (too strict)",
                         "impact": "Threshold too high: UEs with usable NR coverage stay on LTE only"})
    if cell["x2_status"] != "up":
        findings.append({"check": "x2", "severity": "critical", "value": cell["x2_status"]})

    return {"anchor_cell_id": anchor_cell_id.upper(), "anchor_meas_nr": m,
            "findings": findings, "status": "issues found" if findings else "config consistent"}


def check_nr_pci_conflicts(nr_cell_id: str) -> dict:
    """Check NR PCI collision, mod-3 (PSS) and mod-30 (UL DMRS/SRS) conflicts with co-channel neighbors.

    Args:
        nr_cell_id: NR cell identifier, e.g. "BAQ_034_N78_A".
    """
    nr_cell_id, err = _resolve("NR cell", nr_cell_id, NR_CELLS)
    if err:
        return err
    cell = NR_CELLS[nr_cell_id]
    pci, issues = cell["pci"], []
    co = [n for n in cell["neighbors"]
          if NR_CELLS[n]["absolute_frequency_ssb"] == cell["absolute_frequency_ssb"]]
    for n in co:
        p = NR_CELLS[n]["pci"]
        if p == pci:
            issues.append({"type": "collision", "neighbor": n, "pci": p, "severity": "critical"})
            continue
        if p % 3 == pci % 3:
            issues.append({"type": "mod3", "neighbor": n, "pci": p, "severity": "major",
                           "detail": "Same PSS sequence: degrades SSB detection and SS-SINR"})
        if p % 30 == pci % 30:
            issues.append({"type": "mod30", "neighbor": n, "pci": p, "severity": "minor",
                           "detail": "Same UL DMRS/SRS sequence group"})
    used = {NR_CELLS[n]["pci"] for n in co}
    bad3, bad30 = {p % 3 for p in used}, {p % 30 for p in used}
    suggestions = [p for p in range(1008)
                   if p not in used and p % 3 not in bad3 and p % 30 not in bad30][:3]
    return {"nr_cell_id": nr_cell_id.upper(), "pci": pci, "co_channel_neighbors": co,
            "issues": issues, "suggested_pcis": suggestions if issues else []}


def ssb_beam_report(nr_cell_id: str) -> dict:
    """Per-SSB-beam SS-RSRP, SS-SINR and share of samples for an n78 cell (L_max = 8 beams).

    Beam-level data is not in standard per-cell PM counters. It comes from drive-test
    scanners that decode the SSB index, or from UE L3 measurement reports configured to
    include per-SSB results (rsIndexResults), collected via MR/MDT or vendor traces.
    The `source` field says which one was used.

    Args:
        nr_cell_id: NR cell identifier, e.g. "BAQ_021_N78_A".
    """
    nr_cell_id, err = _resolve("NR cell", nr_cell_id, NR_CELLS)
    if err:
        return err
    cell = NR_CELLS[nr_cell_id]
    beams = [{"ssb_index": i, "ss_rsrp_p50_dbm": r, "ss_sinr_p50_db": s, "samples_pct": sh}
             for i, (r, s, sh) in enumerate(cell["beams"])]
    weak = [b for b in beams if b["ss_rsrp_p50_dbm"] < -110 or b["ss_sinr_p50_db"] < 0]
    return {"nr_cell_id": nr_cell_id.upper(), "source": cell["beams_source"], "beams": beams,
            "weak_beams": [b["ssb_index"] for b in weak],
            "samples_on_weak_beams_pct": sum(b["samples_pct"] for b in weak)}


TOOLS = [nr_link_budget, get_endc_kpis, audit_endc_config, check_nr_pci_conflicts, ssb_beam_report]
