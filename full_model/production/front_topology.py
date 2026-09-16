"""Topology-aware periodic zero-contour components for a two-phase front.

The zero set of ``phi = eta_receiver - eta_donor`` is represented by a
marching-triangle graph.  Graph connectivity, not ray-crossing order, owns
component identity.  Receiver area is computed from the same piecewise-linear
cut cells in inverse-tanh profile coordinates, so profile-width relaxation at
a fixed zero set has no material sweep.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class FrontComponent:
    component_id: int
    winding: tuple[int, int]
    centroid_grid: tuple[float, float]
    interface_length_cells: float
    enclosed_area_cells2: float | None
    endpoint_count: int
    closed: bool
    active_window_relationship: str
    representative_tangent: tuple[float, float]
    receiver_normal: tuple[float, float]
    signed_receiver_swept_area_cells2: float
    points_grid: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class TopologySnapshot:
    shape: tuple[int, int]
    components: tuple[FrontComponent, ...]
    next_component_id: int
    ray_crossing_count: int = 0
    filtered_subcell_component_count: int = 0
    filtered_subcell_area_cells2: float = 0.0


@dataclass(frozen=True)
class TopologyMatch:
    snapshot: TopologySnapshot
    event: str | None
    event_record: dict | None
    signed_receiver_area_cells2: float
    receiver_fraction_before: np.ndarray
    receiver_fraction_after: np.ndarray
    maximum_component_distance_cells: float


def component_to_dict(component):
    return asdict(component)


def component_from_dict(value):
    return FrontComponent(
        int(value["component_id"]), tuple(map(int, value["winding"])),
        tuple(map(float, value["centroid_grid"])),
        float(value["interface_length_cells"]),
        (None if value.get("enclosed_area_cells2") is None else
         float(value["enclosed_area_cells2"])),
        int(value["endpoint_count"]), bool(value["closed"]),
        str(value["active_window_relationship"]),
        tuple(map(float, value.get("representative_tangent", (0.0, 0.0)))),
        tuple(map(float, value.get("receiver_normal", (0.0, 0.0)))),
        float(value.get("signed_receiver_swept_area_cells2", 0.0)),
        tuple(tuple(map(float, point)) for point in value["points_grid"]))


def snapshot_to_dict(snapshot):
    return {
        "shape": list(snapshot.shape),
        "next_component_id": snapshot.next_component_id,
        "ray_crossing_count": snapshot.ray_crossing_count,
        "filtered_subcell_component_count": (
            snapshot.filtered_subcell_component_count),
        "filtered_subcell_area_cells2": snapshot.filtered_subcell_area_cells2,
        "components": [component_to_dict(item) for item in snapshot.components],
    }


def snapshot_from_dict(value):
    return TopologySnapshot(
        tuple(map(int, value["shape"])),
        tuple(component_from_dict(item) for item in value["components"]),
        int(value["next_component_id"]), int(value.get("ray_crossing_count", 0)),
        int(value.get("filtered_subcell_component_count", 0)),
        float(value.get("filtered_subcell_area_cells2", 0.0)))


def _profile_coordinate(phi):
    eps = 8.0*np.finfo(float).eps
    return np.arctanh(np.clip(np.asarray(phi, dtype=float),
                              -1.0+eps, 1.0-eps))


def _topology_coordinate(phi, periodic):
    """Return profile coordinates with sign-symmetric exact-zero tie breaks.

    A zero contour that lies exactly on a complete row or column otherwise
    presents marching triangles with zero--zero edges.  Skipping those
    degenerate edges can turn one member of a periodic winding pair into an
    open component.  Move exact-zero samples by one floating-point ulp toward
    the sign selected by the first non-zero centered directional derivative.
    Reversing ``phi`` reverses that derivative and therefore preserves phase
    exchange symmetry.  The cut-cell area calculation continues to use the
    unmodified profile coordinate; this helper affects topology ownership
    only.
    """
    q = _profile_coordinate(phi)
    # Algebraically exact zero levels commonly arrive as a few ulps after the
    # partition-of-unity phase construction (for example -6.7e-15).  Treat
    # only that roundoff neighbourhood as a tie; resolved small values remain
    # untouched.
    zero_tolerance = 64.0*np.finfo(float).eps*max(
        1.0, float(np.max(np.abs(q))))
    zero = np.abs(q) <= zero_tolerance
    if not np.any(zero):
        return q
    direction = np.zeros_like(q)
    for axis in range(q.ndim):
        if periodic:
            derivative = np.roll(q, -1, axis=axis)-np.roll(q, 1, axis=axis)
        else:
            derivative = np.zeros_like(q)
            interior = [slice(None)]*q.ndim
            plus = [slice(None)]*q.ndim
            minus = [slice(None)]*q.ndim
            interior[axis] = slice(1, -1)
            plus[axis] = slice(2, None)
            minus[axis] = slice(None, -2)
            derivative[tuple(interior)] = (
                q[tuple(plus)]-q[tuple(minus)])
        unresolved = zero & (direction == 0.0) & (derivative != 0.0)
        direction[unresolved] = np.sign(derivative[unresolved])
    # Fully flat exact-zero regions have no locally identifiable interface.
    # Leave them untouched rather than inventing topology.
    resolved = zero & (direction != 0.0)
    q = q.copy()
    # Keep the displacement larger than the 1e-10 contour-node hash used
    # below; an ulp at zero would round back onto the degenerate vertex and
    # leave duplicate graph nodes.  This is still nine orders below the
    # O(1) profile coordinate and has no measurable cut-cell-area effect.
    q[resolved] = np.copysign(1.0e-9, direction[resolved])
    return q


def _positive_triangle_fraction(values):
    """Area fraction of a linear triangle on which the scalar is positive."""
    q = np.asarray(values, dtype=float)
    positive = q > 0.0
    count = int(np.count_nonzero(positive))
    if count == 0:
        return 0.0
    if count == 3:
        return 1.0
    if count == 1:
        p = int(np.flatnonzero(positive)[0])
        negative = np.flatnonzero(~positive)
        t0 = q[p]/(q[p]-q[negative[0]])
        t1 = q[p]/(q[p]-q[negative[1]])
        return float(np.clip(t0*t1, 0.0, 1.0))
    n = int(np.flatnonzero(~positive)[0])
    positive_indices = np.flatnonzero(positive)
    t0 = -q[n]/(q[positive_indices[0]]-q[n])
    t1 = -q[n]/(q[positive_indices[1]]-q[n])
    return float(np.clip(1.0-t0*t1, 0.0, 1.0))


def cut_cell_receiver_fraction(phi, *, active_mask=None, periodic=True):
    """Receiver-positive area fraction per primal cell.

    Each square is divided along the same diagonal into two equal triangles.
    Fractions are evaluated in inverse-tanh coordinates, making a tanh profile
    at a fixed zero contour invariant to width.
    """
    value = np.asarray(phi, dtype=float)
    if value.ndim != 2 or not np.all(np.isfinite(value)):
        raise ValueError("cut-cell phi must be a finite 2-D field")
    q = _profile_coordinate(value)
    mask = (np.ones(value.shape, dtype=bool) if active_mask is None
            else np.asarray(active_mask, dtype=bool))
    if mask.shape != value.shape:
        raise ValueError("active mask shape differs from phi")
    nx, ny = value.shape
    fraction = np.zeros_like(value)
    stop_i = nx if periodic else nx-1
    stop_j = ny if periodic else ny-1
    for i in range(stop_i):
        i1 = (i+1) % nx
        for j in range(stop_j):
            j1 = (j+1) % ny
            vertices = ((i, j), (i1, j), (i1, j1), (i, j1))
            if not all(mask[index] for index in vertices):
                continue
            fraction[i, j] = 0.5*(_positive_triangle_fraction(
                (q[i, j], q[i1, j], q[i1, j1]))
                +_positive_triangle_fraction(
                    (q[i, j], q[i1, j1], q[i, j1])))
    return fraction


def _edge_intersections(points, values):
    result = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        qa, qb = float(values[a]), float(values[b])
        if qa == 0.0 and qb == 0.0:
            continue
        if qa == 0.0:
            result.append(np.asarray(points[a], dtype=float))
        elif qb == 0.0:
            result.append(np.asarray(points[b], dtype=float))
        elif qa*qb < 0.0:
            t = qa/(qa-qb)
            result.append((1.0-t)*np.asarray(points[a])+t*np.asarray(points[b]))
    unique = []
    for point in result:
        if not any(np.linalg.norm(point-other) < 1e-10 for other in unique):
            unique.append(point)
    return unique


def _minimum_delta(a, b, shape, periodic):
    delta = np.asarray(b, dtype=float)-np.asarray(a, dtype=float)
    if periodic:
        box = np.asarray(shape, dtype=float)
        delta -= np.round(delta/box)*box
    return delta


def _circular_centroid(points, shape, periodic):
    array = np.asarray(points, dtype=float)
    if not periodic:
        return tuple(np.mean(array, axis=0))
    result = []
    for axis, period in enumerate(shape):
        angle = 2.0*np.pi*array[:, axis]/period
        mean = np.mean(np.exp(1j*angle))
        result.append(float((np.angle(mean) % (2.0*np.pi))*period/(2.0*np.pi)))
    return tuple(result)


def _ordered_component(nodes, adjacency, indices, shape, periodic):
    subset = set(indices)
    endpoints = sorted(index for index in subset
                       if len(adjacency[index] & subset) == 1)
    start = endpoints[0] if endpoints else min(subset)
    order = [start]
    previous = None
    current = start
    while True:
        choices = sorted((adjacency[current] & subset)-({previous} if previous is not None else set()))
        choices = [item for item in choices if item != start or len(order) == len(subset)]
        if not choices:
            break
        following = choices[0]
        if following == start:
            break
        if following in order:
            break
        order.append(following)
        previous, current = current, following
    raw = [np.asarray(nodes[index], dtype=float) for index in order]
    unwrapped = [raw[0]]
    for point in raw[1:]:
        unwrapped.append(unwrapped[-1]+_minimum_delta(
            raw[len(unwrapped)-1], point, shape, periodic))
    return order, np.asarray(unwrapped), len(endpoints)


def _oriented_frame(q, point, tangent, periodic):
    """Orient a representative tangent with receiver-positive on its left."""
    shape = np.asarray(q.shape, dtype=int)
    index = np.rint(point).astype(int)
    if periodic:
        index %= shape
    else:
        index = np.clip(index, 1, shape-2)
    gradient = np.empty(2, dtype=float)
    for axis in range(2):
        plus = index.copy(); minus = index.copy()
        plus[axis] += 1; minus[axis] -= 1
        if periodic:
            plus %= shape; minus %= shape
        else:
            plus = np.clip(plus, 0, shape-1)
            minus = np.clip(minus, 0, shape-1)
        gradient[axis] = 0.5*(q[tuple(plus)]-q[tuple(minus)])
    norm = float(np.linalg.norm(gradient))
    if norm <= 64.0*np.finfo(float).eps:
        unit_tangent = np.asarray(tangent, dtype=float)
        unit_tangent /= max(float(np.linalg.norm(unit_tangent)), 1e-300)
        normal = np.asarray((-unit_tangent[1], unit_tangent[0]))
    else:
        normal = gradient/norm
        unit_tangent = np.asarray((normal[1], -normal[0]))
    # Deterministic orientation: tangent is chosen so the receiver normal is
    # its right normal.  It therefore survives graph traversal reversal.
    return tuple(map(float, unit_tangent)), tuple(map(float, normal))


def extract_front_components(phi, *, active_mask=None, periodic=True,
                             first_component_id=0,
                             minimum_resolved_loop_area_cells2=1.0,
                             minimum_resolved_loop_length_cells=4.0):
    """Extract connected zero-contour components from a periodic cut-cell graph.

    A closed, non-winding contour smaller than both one primal cell and four
    edge lengths has no grid-resolved interior.  It remains present in the
    cut-cell receiver fraction (and therefore in the conservative sweep), but
    is excluded from component ownership and recorded on the snapshot.  Once
    either measure is resolved the loop becomes an explicit topology event.
    """
    value = np.asarray(phi, dtype=float)
    if (value.ndim != 2 or not np.all(np.isfinite(value))
            or np.max(np.abs(value)) > 1.0+32.0*np.finfo(float).eps):
        raise ValueError("front topology requires finite bounded 2-D phi")
    mask = (np.ones_like(value, dtype=bool) if active_mask is None
            else np.asarray(active_mask, dtype=bool))
    if mask.shape != value.shape:
        raise ValueError("active mask shape differs from phi")
    q = _topology_coordinate(value, periodic)
    nx, ny = value.shape
    nodes = []
    node_lookup = {}
    adjacency = []

    def node_index(point):
        normalized = np.asarray(point, dtype=float)
        if periodic:
            box = np.asarray(value.shape, dtype=float)
            normalized %= box
            # Linear interpolation at the periodic seam can return the upper
            # box coordinate minus a few ulps.  Canonicalize that value to
            # zero before hashing; otherwise the same physical node acquires
            # two keys and a winding contour is incorrectly reported open.
            seam = 64.0*np.finfo(float).eps*box
            normalized[(box-normalized) <= seam] = 0.0
            normalized[normalized <= seam] = 0.0
        key = tuple(np.round(normalized, 10))
        if key not in node_lookup:
            node_lookup[key] = len(nodes)
            nodes.append(tuple(map(float, normalized)))
            adjacency.append(set())
        return node_lookup[key]

    stop_i = nx if periodic else nx-1
    stop_j = ny if periodic else ny-1
    for i in range(stop_i):
        i1 = i+1
        wi1 = i1 % nx
        for j in range(stop_j):
            j1 = j+1
            wj1 = j1 % ny
            corners_index = ((i, j), (wi1, j), (wi1, wj1), (i, wj1))
            corners_point = ((i, j), (i1, j), (i1, j1), (i, j1))
            for triangle in ((0, 1, 2), (0, 2, 3)):
                indices = [corners_index[k] for k in triangle]
                if not all(mask[index] for index in indices):
                    continue
                points = [corners_point[k] for k in triangle]
                values = [q[index] for index in indices]
                crossings = _edge_intersections(points, values)
                if len(crossings) != 2:
                    continue
                a, b = node_index(crossings[0]), node_index(crossings[1])
                if a != b:
                    adjacency[a].add(b); adjacency[b].add(a)

    remaining = set(range(len(nodes)))
    raw_components = []
    while remaining:
        seed = min(remaining)
        stack = [seed]; indices = set()
        while stack:
            current = stack.pop()
            if current in indices:
                continue
            indices.add(current); remaining.discard(current)
            stack.extend(adjacency[current]-indices)
        if len(indices) >= 2:
            raw_components.append(indices)

    descriptors = []
    for indices in raw_components:
        order, unwrapped, endpoint_count = _ordered_component(
            nodes, adjacency, indices, value.shape, periodic)
        length = 0.0
        for a, b in zip(order[:-1], order[1:]):
            length += float(np.linalg.norm(_minimum_delta(
                nodes[a], nodes[b], value.shape, periodic)))
        closed = endpoint_count == 0 and len(order) >= 3
        winding = np.zeros(2, dtype=int)
        enclosed = None
        if closed:
            closing = _minimum_delta(nodes[order[-1]], nodes[order[0]],
                                     value.shape, periodic)
            length += float(np.linalg.norm(closing))
            closure = unwrapped[-1]+closing-unwrapped[0]
            winding = np.rint(closure/np.asarray(value.shape)).astype(int)
            for component in winding:
                if component != 0:
                    if component < 0:
                        winding *= -1
                    break
            if not np.any(winding):
                polygon = np.vstack((unwrapped, unwrapped[0]))
                enclosed = abs(float(0.5*np.sum(
                    polygon[:-1, 0]*polygon[1:, 1]
                    -polygon[1:, 0]*polygon[:-1, 1])))
        relationship = (
            "INTERSECTS_ACTIVE_WINDOW" if endpoint_count else
            "PERIODIC_WINDING" if np.any(winding) else "INSIDE_ACTIVE_WINDOW")
        tangent_seed = (unwrapped[1]-unwrapped[0]) if len(unwrapped) > 1 else np.zeros(2)
        tangent, normal = _oriented_frame(q, unwrapped[0], tangent_seed, periodic)
        points = tuple(tuple(map(float, nodes[index])) for index in sorted(indices))
        descriptors.append(FrontComponent(
            -1, tuple(map(int, winding)),
            _circular_centroid(points, value.shape, periodic), float(length),
            enclosed, endpoint_count, closed, relationship, tangent, normal,
            0.0, points))
    # A sign excursion whose entire closed contour is sub-cell in both area
    # and perimeter is not a resolved material component.  It is retained by
    # the cut-cell receiver-area ledger, but cannot own a persistent topology
    # ID.  Either threshold being resolved is sufficient to keep the loop.
    filtered = [item for item in descriptors
                if (item.closed and not any(item.winding)
                    and item.enclosed_area_cells2 is not None
                    and item.enclosed_area_cells2
                    < float(minimum_resolved_loop_area_cells2)
                    and item.interface_length_cells
                    < float(minimum_resolved_loop_length_cells))]
    descriptors = [item for item in descriptors if item not in filtered]
    descriptors.sort(key=lambda item: (
        item.winding, item.active_window_relationship,
        item.centroid_grid, item.interface_length_cells))
    assigned = tuple(replace(item, component_id=first_component_id+index)
                     for index, item in enumerate(descriptors))
    return TopologySnapshot(
        tuple(value.shape), assigned, first_component_id+len(assigned), 0,
        len(filtered), float(sum(
            item.enclosed_area_cells2 or 0.0 for item in filtered)))


def _component_distance(a, b, shape, periodic):
    pa = np.asarray(a.points_grid, dtype=float)
    pb = np.asarray(b.points_grid, dtype=float)
    if pa.size == 0 or pb.size == 0:
        return math.inf
    if periodic:
        tree_a = cKDTree(pa, boxsize=np.asarray(shape, dtype=float))
        tree_b = cKDTree(pb, boxsize=np.asarray(shape, dtype=float))
    else:
        tree_a = cKDTree(pa); tree_b = cKDTree(pb)
    return float(max(np.max(tree_a.query(pb)[0]), np.max(tree_b.query(pa)[0])))


def _component_overlap(a, b, shape, periodic, tolerance_cells=1.5):
    """Symmetric sampled-contour overlap, invariant to point ordering."""
    pa = np.asarray(a.points_grid, dtype=float)
    pb = np.asarray(b.points_grid, dtype=float)
    if pa.size == 0 or pb.size == 0:
        return 0.0
    if periodic:
        tree_a = cKDTree(pa, boxsize=np.asarray(shape, dtype=float))
        tree_b = cKDTree(pb, boxsize=np.asarray(shape, dtype=float))
    else:
        tree_a = cKDTree(pa); tree_b = cKDTree(pb)
    ab = float(np.mean(tree_b.query(pa)[0] <= tolerance_cells))
    ba = float(np.mean(tree_a.query(pb)[0] <= tolerance_cells))
    return min(ab, ba)


def _assign_cut_cell_sweep(components, delta_fraction, shape, periodic):
    """Assign every cut-cell sweep exactly once to its nearest component."""
    if not components:
        return components
    points = []
    owners = []
    for owner, component in enumerate(components):
        points.extend(component.points_grid)
        owners.extend([owner]*len(component.points_grid))
    if not points:
        return components
    points = np.asarray(points, dtype=float)
    tree = (cKDTree(points, boxsize=np.asarray(shape, dtype=float))
            if periodic else cKDTree(points))
    cells = np.argwhere(np.abs(delta_fraction) > 0.0).astype(float)+0.5
    sweep = np.zeros(len(components), dtype=np.longdouble)
    if len(cells):
        nearest = tree.query(cells)[1]
        values = delta_fraction[tuple(np.asarray(cells-.5, dtype=int).T)]
        np.add.at(sweep, np.asarray(owners, dtype=int)[nearest], values)
    return tuple(replace(component,
                         signed_receiver_swept_area_cells2=float(sweep[index]))
                 for index, component in enumerate(components))


def _topology_event(previous, current, phi):
    old_count = len(previous.components); new_count = len(current.components)
    if old_count == 0 and new_count > 0:
        return "PAIR_ENTERED_ACTIVE_WINDOW"
    if old_count > 0 and new_count == 0:
        if all(item.active_window_relationship == "INTERSECTS_ACTIVE_WINDOW"
               for item in previous.components):
            return "PAIR_LEFT_ACTIVE_WINDOW"
        if old_count >= 2 or any(any(item.winding) for item in previous.components):
            return "INTERFACE_PAIR_ANNIHILATED"
        return "GRAIN_CONSUMED"
    if new_count > old_count:
        # A closed island appearing alongside an existing component that
        # intersects the active window is not an ordinary split of that
        # component.  With nucleation disabled it is an unauthorized phase
        # island and must remain a named scientific terminal.  Sub-cell loops
        # have already been handled, conservatively, by extraction.
        if (any(item.active_window_relationship == "INTERSECTS_ACTIVE_WINDOW"
                for item in previous.components)
                and any(item.closed and not any(item.winding)
                        and item.active_window_relationship
                        != "INTERSECTS_ACTIVE_WINDOW"
                        for item in current.components)):
            return "UNAUTHORIZED_PHASE_ISLAND"
        return "FRONT_COMPONENT_SPLIT"
    if new_count < old_count:
        return "FRONT_COMPONENT_MERGE"
    return "PAIR_IDENTITY_LOST"


def match_front_topology(previous, phi_before, phi_after, *, active_mask=None,
                         periodic=True, maximum_match_distance_cells=4.0,
                         ray_crossing_count=0,
                         support_component_reconnection=False):
    """Match persistent components and classify every non-bijective event."""
    before = np.asarray(phi_before, dtype=float)
    after = np.asarray(phi_after, dtype=float)
    if before.shape != after.shape or tuple(before.shape) != previous.shape:
        raise ValueError("topology fields and snapshot shapes differ")
    current = extract_front_components(
        after, active_mask=active_mask, periodic=periodic,
        first_component_id=previous.next_component_id)
    old = previous.components; new = current.components
    event = None; record = None; maximum_distance = 0.0
    count_event = (None if len(old) == len(new)
                   else _topology_event(previous, current, after))
    supported_count_event = bool(
        support_component_reconnection
        and count_event in {"FRONT_COMPONENT_SPLIT", "FRONT_COMPONENT_MERGE"})
    if count_event is not None and not supported_count_event:
        event = count_event
    elif not old or not new:
        matched = current.components
    else:
        cost = np.full((len(old), len(new)), 1e12, dtype=float)
        geometric_distance = np.full_like(cost, np.inf)
        for i, a in enumerate(old):
            for j, b in enumerate(new):
                if tuple(map(abs, a.winding)) != tuple(map(abs, b.winding)):
                    continue
                if bool(a.endpoint_count) != bool(b.endpoint_count):
                    continue
                distance = _component_distance(a, b, previous.shape, periodic)
                overlap = _component_overlap(a, b, previous.shape, periodic)
                length_scale = max(a.interface_length_cells,
                                   b.interface_length_cells, 1.0)
                length_change = abs(a.interface_length_cells-b.interface_length_cells)/length_scale
                cost[i, j] = distance+length_change+(1.0-overlap)
                geometric_distance[i, j] = distance
        rows, columns = linear_sum_assignment(cost)
        distances = [geometric_distance[i, j] for i, j in zip(rows, columns)]
        match_limit = float(maximum_match_distance_cells)
        if supported_count_event:
            # Hausdorff distance necessarily jumps at a pinch-off/reconnection.
            # Bound persistence by half the larger participating contour,
            # rather than making such events unmatchable by the translation
            # threshold used for bijective motion.
            match_limit = max(match_limit, 0.5*max(
                [item.interface_length_cells for item in old+new]))
        required_matches = min(len(old), len(new))
        if (len(rows) != required_matches or any(not np.isfinite(item)
                or item > match_limit for item in distances)):
            event = "PAIR_IDENTITY_LOST"
        else:
            maximum_distance = max(distances, default=0.0)
            assigned = list(new)
            for i, j in zip(rows, columns):
                assigned[j] = replace(new[j], component_id=old[i].component_id)
            matched = tuple(assigned)
            current = replace(current, components=matched)
            if supported_count_event:
                record = {
                    "classification": "SUPPORTED_"+count_event,
                    "old_component_count": len(old),
                    "new_component_count": len(new),
                    "old_components": [component_to_dict(item) for item in old],
                    "new_components": [component_to_dict(item) for item in new],
                    "receiver_all_positive": bool(np.all(after > 0.0)),
                    "receiver_all_negative": bool(np.all(after < 0.0)),
                }
    if event is not None:
        record = {
            "classification": event,
            "old_component_count": len(old),
            "new_component_count": len(new),
            "old_components": [component_to_dict(item) for item in old],
            "new_components": [component_to_dict(item) for item in new],
            "receiver_all_positive": bool(np.all(after > 0.0)),
            "receiver_all_negative": bool(np.all(after < 0.0)),
        }
        current = replace(current, components=new)
    current = replace(current, next_component_id=max(
        previous.next_component_id, current.next_component_id),
        ray_crossing_count=int(ray_crossing_count))
    fraction_before = cut_cell_receiver_fraction(
        before, active_mask=active_mask, periodic=periodic)
    fraction_after = cut_cell_receiver_fraction(
        after, active_mask=active_mask, periodic=periodic)
    signed_area = float(np.sum(
        fraction_after-fraction_before, dtype=np.longdouble))
    if event is None:
        current = replace(current, components=_assign_cut_cell_sweep(
            current.components, fraction_after-fraction_before,
            previous.shape, periodic))
    return TopologyMatch(
        current, event, record, signed_area, fraction_before, fraction_after,
        maximum_distance)


def initialize_front_topology(phi, *, active_mask=None, periodic=True,
                              ray_crossing_count=0):
    snapshot = extract_front_components(
        phi, active_mask=active_mask, periodic=periodic, first_component_id=0)
    return replace(snapshot, ray_crossing_count=int(ray_crossing_count))


def diagnostic_ray_crossing_count(phi, *, normal_axis, active_mask=None,
                                  periodic=True):
    """Return the legacy ray count as a non-authoritative diagnostic."""
    value = np.asarray(phi, dtype=float)
    mask = (np.ones_like(value, dtype=bool) if active_mask is None
            else np.asarray(active_mask, dtype=bool))
    rays = np.moveaxis(value, int(normal_axis), 0)
    valid = np.moveaxis(mask, int(normal_axis), 0)
    total = 0
    stop = rays.shape[0] if periodic else rays.shape[0]-1
    for j in range(rays.shape[1]):
        for i in range(stop):
            k = (i+1) % rays.shape[0]
            if not (valid[i, j] and valid[k, j]):
                continue
            a, b = rays[i, j], rays[k, j]
            total += int((a == 0.0 and b != 0.0) or a*b < 0.0)
    return total
