"""Physical site measure for atomic moving-front events.

For the baseline event volume ``b**3`` and normal step ``b``, one independent
front site occupies area ``b**2``.  The extensive expected event count is
therefore interface area divided by ``b**2`` times the per-site Poisson rate.
This measure is independent of mesh cells and patch partitioning.  It scales
linearly with *physical* represented thickness, while count and swept volume
per interface area are thickness independent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class PhysicalFrontEventMeasure:
    burgers_m: float
    event_volume_m3: float
    event_length_m: float
    site_area_m2: float
    interface_length_m: float
    represented_thickness_m: float
    interface_area_m2: float
    physical_site_count: float
    rate_a_to_b_per_site_s: float
    rate_b_to_a_per_site_s: float
    dt_s: float
    expected_events_a_to_b: float
    expected_events_b_to_a: float
    expected_signed_event_count: float
    expected_absolute_event_count: float
    expected_signed_swept_volume_m3: float
    expected_absolute_swept_volume_m3: float
    expected_normal_velocity_m_s: float
    site_count_per_interface_area_m2: float
    event_count_per_interface_area: float
    swept_volume_per_interface_area_m: float

    def to_dict(self):
        return asdict(self)


def physical_front_event_measure(
        *, interface_length_m, represented_thickness_m, burgers_m,
        rate_a_to_b_per_site_s, rate_b_to_a_per_site_s, dt_s,
        event_volume_m3=None, event_length_m=None):
    """Return the physical Poisson site/event measure for a front patch."""
    b = float(burgers_m)
    volume = b**3 if event_volume_m3 is None else float(event_volume_m3)
    length = b if event_length_m is None else float(event_length_m)
    interface_length = float(interface_length_m)
    thickness = float(represented_thickness_m)
    rate_ab = float(rate_a_to_b_per_site_s)
    rate_ba = float(rate_b_to_a_per_site_s)
    dt = float(dt_s)
    values = (b, volume, length, interface_length, thickness,
              rate_ab, rate_ba, dt)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("front event measure inputs must be finite")
    if min(b, volume, length, thickness, dt) <= 0.0 or interface_length < 0.0:
        raise ValueError(
            "front event scales, thickness, and dt must be positive; "
            "interface length must be nonnegative")
    if min(rate_ab, rate_ba) < 0.0:
        raise ValueError("front event rates must be nonnegative")
    site_area = volume/length
    interface_area = interface_length*thickness
    sites = interface_area/site_area
    events_ab = sites*rate_ab*dt
    events_ba = sites*rate_ba*dt
    signed_events = events_ab-events_ba
    absolute_events = events_ab+events_ba
    signed_volume = signed_events*volume
    absolute_volume = absolute_events*volume
    velocity = length*(rate_ab-rate_ba)
    return PhysicalFrontEventMeasure(
        b, volume, length, site_area, interface_length, thickness,
        interface_area, sites, rate_ab, rate_ba, dt, events_ab, events_ba,
        signed_events, absolute_events, signed_volume, absolute_volume,
        velocity, 1.0/site_area, (rate_ab+rate_ba)*dt/site_area,
        length*(rate_ab-rate_ba)*dt)


def combine_front_event_measures(measures):
    """Add patch measures after checking one shared microscopic event law."""
    items = tuple(measures)
    if not items:
        raise ValueError("at least one physical front patch is required")
    reference = items[0]
    for item in items[1:]:
        for name in ("burgers_m", "event_volume_m3", "event_length_m",
                     "represented_thickness_m", "rate_a_to_b_per_site_s",
                     "rate_b_to_a_per_site_s", "dt_s"):
            if getattr(item, name) != getattr(reference, name):
                raise ValueError("front patches do not share one event law")
    return physical_front_event_measure(
        interface_length_m=sum(item.interface_length_m for item in items),
        represented_thickness_m=reference.represented_thickness_m,
        burgers_m=reference.burgers_m,
        rate_a_to_b_per_site_s=reference.rate_a_to_b_per_site_s,
        rate_b_to_a_per_site_s=reference.rate_b_to_a_per_site_s,
        dt_s=reference.dt_s,
        event_volume_m3=reference.event_volume_m3,
        event_length_m=reference.event_length_m)
