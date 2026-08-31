"""Named-region tactile proxy derived from MuJoCo contacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

FEATURE_NAMES = (
    "contact_occupancy",
    "normal_force",
    "tangential_force_magnitude",
    "cop_x",
    "cop_y",
    "cop_z",
)


@dataclass(frozen=True)
class ContactSample:
    region_name: str
    normal_force: float
    tangential_force_magnitude: float
    position: np.ndarray


@dataclass(frozen=True)
class RegionDefinition:
    region_name: str
    hand_side: str
    body_names: tuple[str, ...]
    geom_names: tuple[str, ...] = ()


class ContactRegionMap:
    """Resolve stable body/geom names to runtime geom IDs."""

    def __init__(self, regions: Sequence[RegionDefinition], object_body_names: Sequence[str]):
        if not regions:
            raise ValueError("at least one contact region is required")
        names = [region.region_name for region in regions]
        if len(names) != len(set(names)):
            raise ValueError("region names must be unique")
        self.regions = tuple(regions)
        self.object_body_names = frozenset(object_body_names)
        self.geom_to_region: dict[int, str] = {}
        self._audit: dict[str, Any] | None = None

    @classmethod
    def from_config(cls, value: Mapping[str, Any]) -> "ContactRegionMap":
        regions = [
            RegionDefinition(
                region_name=row["region_name"],
                hand_side=row["hand_side"],
                body_names=tuple(row.get("body_names", ())),
                geom_names=tuple(row.get("geom_names", ())),
            )
            for row in value["regions"]
        ]
        return cls(regions, value.get("object_body_names", ()))

    @property
    def region_names(self) -> tuple[str, ...]:
        return tuple(region.region_name for region in self.regions)

    @property
    def tactile_dim(self) -> int:
        return len(self.regions) * len(FEATURE_NAMES)

    def resolve(self, model: Any, mujoco_module: Any) -> dict[str, Any]:
        body_to_region: dict[str, str] = {}
        geom_name_to_region: dict[str, str] = {}
        for region in self.regions:
            for body_name in region.body_names:
                if body_name in body_to_region:
                    raise ValueError(f"body {body_name!r} is assigned to multiple regions")
                body_to_region[body_name] = region.region_name
            for geom_name in region.geom_names:
                if geom_name in geom_name_to_region:
                    raise ValueError(f"geom {geom_name!r} is assigned to multiple regions")
                geom_name_to_region[geom_name] = region.region_name

        geom_to_region: dict[int, str] = {}
        relevant: list[str] = []
        mapped: list[str] = []
        unmapped: list[str] = []
        for geom_id in range(model.ngeom):
            body_id = int(model.geom_bodyid[geom_id])
            body_name = mujoco_module.mj_id2name(model, mujoco_module.mjtObj.mjOBJ_BODY, body_id)
            geom_name = mujoco_module.mj_id2name(model, mujoco_module.mjtObj.mjOBJ_GEOM, geom_id)
            region = geom_name_to_region.get(geom_name) or body_to_region.get(body_name)
            collision_enabled = bool(model.geom_contype[geom_id] or model.geom_conaffinity[geom_id])
            descriptor = geom_name or f"body:{body_name}#collision-geom"
            if collision_enabled and body_name in body_to_region:
                relevant.append(descriptor)
                if region is None:
                    unmapped.append(descriptor)
                else:
                    mapped.append(descriptor)
            if region is not None and collision_enabled:
                if geom_id in geom_to_region:
                    raise ValueError(f"runtime geom {geom_id} is assigned twice")
                geom_to_region[geom_id] = region

        if not geom_to_region:
            raise ValueError("contact region map resolved no collision geoms")
        self.geom_to_region = geom_to_region
        self._audit = {
            "all_hand_collision_geoms": relevant,
            "mapped_geoms": mapped,
            "unmapped_relevant_geoms": unmapped,
            "overlapping_assignments": [],
            "mapped_geom_count": len(mapped),
            "name_based_resolution": True,
            "status": "PASS" if not unmapped else "FAIL",
        }
        return dict(self._audit)

    def region_for_geom(self, geom_id: int) -> str | None:
        return self.geom_to_region.get(int(geom_id))


def aggregate_contact_samples(
    samples: Iterable[ContactSample], region_names: Sequence[str], epsilon: float = 1e-9
) -> tuple[np.ndarray, dict[str, Any]]:
    """Aggregate positive-force contacts into one canonical vector per region."""

    grouped: dict[str, list[ContactSample]] = {name: [] for name in region_names}
    for sample in samples:
        if sample.region_name not in grouped:
            raise ValueError(f"unknown contact region {sample.region_name!r}")
        if sample.normal_force > epsilon:
            grouped[sample.region_name].append(sample)

    rows: list[np.ndarray] = []
    counts: dict[str, int] = {}
    for name in region_names:
        contacts = grouped[name]
        counts[name] = len(contacts)
        if not contacts:
            rows.append(np.zeros(len(FEATURE_NAMES), dtype=np.float32))
            continue
        weights = np.asarray([max(row.normal_force, 0.0) for row in contacts], dtype=np.float64)
        positions = np.stack([np.asarray(row.position, dtype=np.float64) for row in contacts])
        normal = float(weights.sum())
        tangential = float(sum(max(row.tangential_force_magnitude, 0.0) for row in contacts))
        cop = (weights[:, None] * positions).sum(axis=0) / (normal + epsilon)
        rows.append(np.asarray([1.0, normal, tangential, *cop], dtype=np.float32))
    return np.concatenate(rows), {
        "contact_count_by_region": counts,
        "contact_count": sum(counts.values()),
    }


class SimulatedTactileExtractor:
    """Extract the S4.1 physics-derived contact proxy from MuJoCo data."""

    def __init__(self, region_map: ContactRegionMap, epsilon: float = 1e-9):
        self.region_map = region_map
        self.epsilon = float(epsilon)

    def extract(
        self, model: Any, data: Any, mujoco_module: Any
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not self.region_map.geom_to_region:
            self.region_map.resolve(model, mujoco_module)
        samples: list[ContactSample] = []
        matched_pairs: list[dict[str, Any]] = []
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            region1 = self.region_map.region_for_geom(geom1)
            region2 = self.region_map.region_for_geom(geom2)
            body1 = mujoco_module.mj_id2name(
                model, mujoco_module.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom1])
            )
            body2 = mujoco_module.mj_id2name(
                model, mujoco_module.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom2])
            )
            if region1 is not None and body2 in self.region_map.object_body_names:
                region = region1
            elif region2 is not None and body1 in self.region_map.object_body_names:
                region = region2
            else:
                continue
            wrench = np.zeros(6, dtype=np.float64)
            mujoco_module.mj_contactForce(model, data, contact_id, wrench)
            normal = max(float(wrench[0]), 0.0)
            tangential = float(np.linalg.norm(wrench[1:3]))
            samples.append(
                ContactSample(
                    region_name=region,
                    normal_force=normal,
                    tangential_force_magnitude=tangential,
                    position=np.asarray(contact.pos, dtype=np.float64).copy(),
                )
            )
            matched_pairs.append(
                {
                    "region": region,
                    "hand_is_geom1": region1 is not None,
                    "normal_force": normal,
                    "tangential_force_magnitude": tangential,
                }
            )
        tactile, diagnostics = aggregate_contact_samples(
            samples, self.region_map.region_names, epsilon=self.epsilon
        )
        diagnostics["matched_pairs"] = matched_pairs
        return tactile, diagnostics
