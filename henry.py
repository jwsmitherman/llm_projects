"""Henry Industries - port of henry_industries_pipeline.ipynb.

route + stat sheets. Route legs are aggregated per
(PHARM_CODE, PICKUP_TIME, ROUTE_NAME); per-stop charges use (STOP_NUMBER-1).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .base import COLS_LST, CarrierProcessor, CarrierResult, clean_rate_cols

MERGE_COLS = ["PHARM_CODE"] + COLS_LST

_TOLL_KEYS = ("tolls", "toll", "tollcharge", "tollcharges")
_FUEL_KEYS = ("fuel", "fsc", "fuelsurcharge", "fscs")


def _optional(df, keys):
    """Numeric Series for the first column whose name matches, else zeros.
    Lets pass-through columns (Toll, Fuel) be present or absent without error."""
    for c in df.columns:
        if re.sub(r"[^a-z0-9]", "", str(c).lower()) in keys:
            return pd.to_numeric(df[c], errors="coerce").fillna(0)
    return pd.Series(0.0, index=df.index)


class HenryProcessor(CarrierProcessor):
    slug = "henry_industries"
    display_name = "Henry Industries"
    courier_filter = "HENRY"

    def process(self, xlsx: pd.ExcelFile, rate_df: pd.DataFrame) -> CarrierResult:
        rate_hi = clean_rate_cols(self.filter_rate(rate_df))
        rate_hi["PHARM_CODE"] = rate_hi["LOCATION"]

        route_sheet = next((s for s in xlsx.sheet_names if "details" in s.lower()),
                           xlsx.sheet_names[0])
        stat_sheet = next((s for s in xlsx.sheet_names if "stat" in s.lower()),
                          xlsx.sheet_names[0])
        route_df = xlsx.parse(route_sheet)
        stat_df = xlsx.parse(stat_sheet)

        merge_cols = [c for c in MERGE_COLS if c in rate_hi.columns]

        # ---- route ----
        # Tolls and Fuel are optional pass-throughs: use the column if present, else 0.
        route_df["Tolls"] = _optional(route_df, _TOLL_KEYS)
        route_df["Fuel"] = _optional(route_df, _FUEL_KEYS)

        route_df["row_id"] = route_df.index
        r = route_df.merge(rate_hi[merge_cols], on="PHARM_CODE", how="left")
        r["ROUTED_BASE_COST"] = r["TOTAL_MILES"] * r["ROUTEDCPM"]

        agg_keys = ["PHARM_CODE", "PICKUP_TIME", "ROUTE_NAME"]
        agg = r.groupby(agg_keys).agg(
            ROUTED_BASE_COST=("ROUTED_BASE_COST", "sum"),
            STOP_NUMBER=("STOP_NUMBER", "max"),
            ROUTEDCPS=("ROUTEDCPS", "max"),
            ROUTEMIN=("ROUTEMIN", "max"),
            AMOUNT_CHARGES=("AMOUNT_CHARGES", "sum"),
            Tolls=("Tolls", "sum"),
            Fuel=("Fuel", "sum"),
        ).reset_index()

        agg["CPS_TOTAL"] = round((agg["STOP_NUMBER"] - 1) * agg["ROUTEDCPS"], 2)
        agg["ROUTED_BASE_COST"] = agg["ROUTED_BASE_COST"] + agg["CPS_TOTAL"]
        agg["ROUTED_BASE_COST_MIN"] = pd.Series(np.where(
            agg["ROUTED_BASE_COST"] < agg["ROUTEMIN"],
            agg["ROUTEMIN"], agg["ROUTED_BASE_COST"])).round(2)
        agg["total_cost"] = agg["ROUTED_BASE_COST_MIN"] + agg["Tolls"] + agg["Fuel"]
        agg["diff"] = round(agg["AMOUNT_CHARGES"] - round(agg["total_cost"], 2), 0)

        route_final = route_df.merge(
            agg[agg_keys + ["total_cost", "diff"]], on=agg_keys, how="left"
        ).drop(columns=["row_id"])

        # ---- stat ----
        # Fuel (and toll, if present) are pass-throughs added to the stat charge.
        stat_df["Fuel"] = _optional(stat_df, _FUEL_KEYS)
        stat_df["Tolls"] = _optional(stat_df, _TOLL_KEYS)
        stat_df["row_id"] = stat_df.index
        s = stat_df.merge(rate_hi[merge_cols], on="PHARM_CODE", how="left")
        s["STAT_BASE_COST"] = s["TOTAL_MILES"] * s["STATCPM"]
        s["STAT_BASE_COST_MIN"] = pd.Series(np.where(
            s["STAT_BASE_COST"] < s["STATMIN"],
            s["STATMIN"], s["STAT_BASE_COST"])).round(2)
        s["total_cost"] = (s["STAT_BASE_COST_MIN"] + s["Fuel"] + s["Tolls"]).round(2)
        s["diff"] = round(s["AMOUNT_CHARGES"] - round(s["total_cost"], 2), 0)

        stat_final = stat_df.merge(
            s[["row_id", "total_cost", "diff"]], on="row_id", how="left"
        ).drop(columns=["row_id"])

        return CarrierResult(sheets={"Route": route_final, "Stats": stat_final}, ext="xlsx")
