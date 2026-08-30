import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PARTNER_DONOR_MAP_PATH = SCRIPT_DIR / "partner_donor_map.csv"
PARTNER_DONOR_MAP_PATH = Path(
    os.getenv("PARTNER_DONOR_MAP_PATH", DEFAULT_PARTNER_DONOR_MAP_PATH)
)
VALID_MODES = {"absorb", "skip"}


def normalize_location_id(value):
    value_str = str(value).strip()
    if value_str.endswith(".0"):
        candidate = value_str[:-2]
        if candidate.isdigit():
            return candidate
    return value_str


@dataclass(frozen=True)
class PartnerGrouping:
    mapping_df: pd.DataFrame
    location_to_planning: dict
    planning_to_members: dict
    partner_name_by_id: dict
    donor_name_by_id: dict

    def planning_location_id(self, location_id):
        normalized = normalize_location_id(location_id)
        return self.location_to_planning.get(normalized, normalized)


def _empty_mapping():
    return pd.DataFrame(
        columns=[
            "partner_location_id",
            "partner_name",
            "kfm_donor_id",
            "kfm_donor_name",
            "mode",
            "notes",
        ]
    )


def load_partner_donor_map(path=PARTNER_DONOR_MAP_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Partner/donor mapping not found: {path}")

    mapping = pd.read_csv(path, dtype=str).fillna("")
    required_columns = set(_empty_mapping().columns)
    missing_columns = sorted(required_columns - set(mapping.columns))
    if missing_columns:
        raise ValueError(
            f"Partner/donor mapping is missing columns: {missing_columns}"
        )

    mapping = mapping[list(_empty_mapping().columns)].copy()
    mapping["partner_location_id"] = mapping["partner_location_id"].map(
        normalize_location_id
    )
    mapping["kfm_donor_id"] = mapping["kfm_donor_id"].map(normalize_location_id)
    mapping["mode"] = mapping["mode"].str.strip().str.lower()
    mapping["partner_name"] = mapping["partner_name"].str.strip()
    mapping["kfm_donor_name"] = mapping["kfm_donor_name"].str.strip()
    mapping["notes"] = mapping["notes"].str.strip()

    empty_partner = mapping["partner_location_id"] == ""
    if empty_partner.any():
        rows = mapping.index[empty_partner].tolist()
        raise ValueError(f"Empty partner_location_id in mapping rows: {rows}")

    invalid_modes = sorted(set(mapping["mode"]) - VALID_MODES)
    if invalid_modes:
        raise ValueError(f"Unsupported partner mapping modes: {invalid_modes}")

    missing_donor = (mapping["mode"] == "absorb") & (
        mapping["kfm_donor_id"] == ""
    )
    if missing_donor.any():
        partners = mapping.loc[missing_donor, "partner_location_id"].tolist()
        raise ValueError(f"Absorb partners without donor: {partners}")

    duplicate_conflicts = []
    for partner_id, group in mapping.groupby("partner_location_id", sort=False):
        definitions = set(zip(group["mode"], group["kfm_donor_id"]))
        if len(definitions) > 1:
            duplicate_conflicts.append(partner_id)
    if duplicate_conflicts:
        raise ValueError(
            "Partners have conflicting mapping rows: "
            + ", ".join(duplicate_conflicts)
        )

    return mapping.drop_duplicates(
        subset=["partner_location_id", "mode", "kfm_donor_id"],
        keep="first",
    ).reset_index(drop=True)


def build_partner_grouping(mapping_df):
    absorb_direct = {
        str(row.partner_location_id): str(row.kfm_donor_id)
        for row in mapping_df.itertuples(index=False)
        if row.mode == "absorb"
    }

    resolved = {}

    def resolve(location_id, stack):
        if location_id in resolved:
            return resolved[location_id]
        if location_id not in absorb_direct:
            return location_id
        if location_id in stack:
            cycle = " -> ".join(stack + [location_id])
            raise ValueError(f"Cycle in partner/donor mapping: {cycle}")
        target = resolve(absorb_direct[location_id], stack + [location_id])
        resolved[location_id] = target
        return target

    for source_id in absorb_direct:
        resolve(source_id, [])

    planning_to_members = {}
    for source_id, planning_id in resolved.items():
        members = planning_to_members.setdefault(planning_id, set())
        members.add(planning_id)
        members.add(source_id)

    partner_name_by_id = {
        str(row.partner_location_id): str(row.partner_name)
        for row in mapping_df.itertuples(index=False)
    }
    donor_name_by_id = {
        str(row.kfm_donor_id): str(row.kfm_donor_name)
        for row in mapping_df.itertuples(index=False)
        if row.mode == "absorb" and str(row.kfm_donor_id)
    }

    return PartnerGrouping(
        mapping_df=mapping_df.copy(),
        location_to_planning=resolved,
        planning_to_members={
            planning_id: sorted(members, key=lambda value: (not value.isdigit(), value))
            for planning_id, members in planning_to_members.items()
        },
        partner_name_by_id=partner_name_by_id,
        donor_name_by_id=donor_name_by_id,
    )


def load_partner_grouping(path=PARTNER_DONOR_MAP_PATH):
    return build_partner_grouping(load_partner_donor_map(path))


def apply_partner_grouping(df, grouping, location_column="location_id"):
    result = df.copy()
    result[location_column] = (
        result[location_column]
        .map(normalize_location_id)
        .map(grouping.planning_location_id)
    )
    return result


def build_grouping_config_audit(grouping):
    audit = grouping.mapping_df.copy()
    audit["planning_location_id"] = audit["partner_location_id"].map(
        grouping.planning_location_id
    )
    audit["action"] = audit["mode"].map(
        {"absorb": "group_before_slot_calculation", "skip": "keep_independent"}
    )
    return audit
