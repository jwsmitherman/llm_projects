import re
import numpy as np
import pandas as pd

HA_LEVELS = {1, 2}
AMB_LOOSE = r"ambulance|bls|als|911"
AMB_STRICT = r"(?<![a-z])(ambulance|bls|als)(?![a-z])|(?<!\d)911(?!\d)"
TOKENS = ["ambulance", "bls", "als", "911"]

dispo_raw = df[DISPO].fillna("").astype(str)
dispo_low = dispo_raw.str.lower()
nmtara_raw = df[NMTARA].fillna("").astype(str)
nmtara_low = nmtara_raw.str.lower()
ha_mask = df["nmtara_level"].isin(HA_LEVELS)
ha = df[ha_mask]

print("=" * 70)
print("1. DENOMINATOR")
print("=" * 70)
print("rows in df:", f"{len(df):,}")
print("high-acuity rows (nmtara_level in 1,2):", f"{len(ha):,}")
print("share recomputed:", round(len(ha) / len(df) * 100, 3), "percent")
print("total implied by a 20.5 percent share:", f"{round(len(ha) / 0.205):,}")
print()
print("nmtara_level distribution")
print(df["nmtara_level"].value_counts(dropna=False).sort_index().to_string())
print("nmtara_level null:", int(df["nmtara_level"].isna().sum()))

print()
print("=" * 70)
print("2. RAW COUNTS BEHIND THE 100.0 / 0.0 SPLIT")
print("=" * 70)
n_amb = int(ha["is_ambulance"].sum())
n_nonamb = int((~ha["is_ambulance"]).sum())
print("high acuity ending in ambulance:", f"{n_amb:,}")
print("high acuity NOT ending in ambulance:", f"{n_nonamb:,}")
print("unrounded non-ambulance share:", f"{n_nonamb / max(len(ha), 1) * 100:.6f} percent")
print()
print(pd.crosstab(df["nmtara_level"], df["is_ambulance"], dropna=False).to_string())

print()
print("=" * 70)
print("3. IS THE DISPOSITION FIELD MULTI-VALUED")
print("=" * 70)
multi = dispo_low.str.contains(r"[;,|]", regex=True, na=False)
print("rows whose response field holds more than one value:", f"{int(multi.sum()):,}",
      f"({multi.mean() * 100:.1f} percent)")
print("max values in one row:", int(dispo_low.str.count(r"[;,|]").max()) + 1)
print()
print("high-acuity rows flagged BOTH self-care and ambulance:",
      f"{int((ha['is_self_care'] & ha['is_ambulance']).sum()):,}")
print("high-acuity rows flagged BOTH urgent and ambulance:",
      f"{int((ha['is_urgent'] & ha['is_ambulance']).sum()):,}")

print()
print("=" * 70)
print("4. WHICH TOKEN FIRED is_ambulance")
print("=" * 70)


def firing_tokens(s):
    hit = [t for t in TOKENS if t in s]
    return "|".join(hit) if hit else "none"


print(dispo_low[ha_mask].apply(firing_tokens).value_counts().to_string())

print()
print("=" * 70)
print("5. SUBSTRING FALSE POSITIVES")
print("=" * 70)
loose = dispo_low.str.contains(AMB_LOOSE, regex=True, na=False)
strict = dispo_low.str.contains(AMB_STRICT, regex=True, na=False)
print("matched by the current pattern:", f"{int(loose.sum()):,}")
print("matched with word boundaries:", f"{int(strict.sum()):,}")
print("matched only without boundaries:", f"{int((loose & ~strict).sum()):,}")
print()
if int((loose & ~strict).sum()):
    print("response values that matched only as a substring")
    print(dispo_raw[loose & ~strict].value_counts().head(20).to_string())
else:
    print("no substring-only matches")
print()
ha_strict = strict[ha_mask]
print("high acuity, strict pattern, ambulance:", f"{int(ha_strict.sum()):,}")
print("high acuity, strict pattern, NOT ambulance:", f"{int((~ha_strict).sum()):,}")

print()
print("=" * 70)
print("6. HOW MANY DISTINCT RESPONSES EXIST INSIDE HIGH ACUITY")
print("=" * 70)
print("distinct response values:", int(ha[DISPO].nunique(dropna=False)))
print()
print(ha[DISPO].fillna("(null)").value_counts().head(25).to_string())
print()
print("distinct nmtara source values:", int(ha[NMTARA].nunique(dropna=False)))
print()
print(ha[NMTARA].fillna("(null)").value_counts().head(25).to_string())

print()
print("=" * 70)
print("7. DO THE TWO FIELDS ENCODE THE SAME THING")
print("=" * 70)
nm_has_amb = nmtara_low.str.contains(AMB_STRICT, regex=True, na=False)
print("rows whose NMTARA text itself names a transport mode:",
      f"{int(nm_has_amb.sum()):,}", f"({nm_has_amb.mean() * 100:.1f} percent)")
print()
pair = (pd.DataFrame({"nmtara": nmtara_raw[ha_mask].str.slice(0, 45),
                      "response": dispo_raw[ha_mask].str.slice(0, 45)})
        .value_counts().rename("calls").reset_index().head(20))
print(pair.to_string(index=False))

print()
print("=" * 70)
print("8. BUCKET COMPOSITION")
print("=" * 70)
print(ha["bucket"].value_counts(dropna=False).to_string())
print()
print("override flag inside high acuity:", f"{int(ha['is_amb_override'].sum()):,}",
      "of", f"{len(ha):,}")

print()
print("=" * 70)
print("9. THE INVERSE POPULATION")
print("=" * 70)
low_ac = df[df["nmtara_level"].between(3, 5)]
print("nmtara 3-5 calls:", f"{len(low_ac):,}")
print("of those, ended in ambulance:", f"{int(low_ac['is_ambulance'].sum()):,}",
      f"({low_ac['is_ambulance'].mean() * 100:.1f} percent)")
print("with the strict pattern:", f"{int(strict[df['nmtara_level'].between(3, 5)].sum()):,}",
      f"({strict[df['nmtara_level'].between(3, 5)].mean() * 100:.1f} percent)")

print()
print("=" * 70)
print("10. THE CALLS THAT BREAK THE PATTERN")
print("=" * 70)
exceptions = ha[~ha["is_ambulance"]]
print("count:", f"{len(exceptions):,}")
if len(exceptions):
    cols = [c for c in [NMTARA, DISPO, MARKET, CAUSE] if c]
    print(exceptions[cols].head(25).to_string(index=False))
else:
    print("none, which is the result that needs explaining")
