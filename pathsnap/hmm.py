"""Hidden Markov Model (HMM) and Viterbi Route Reconstruction for PathSnap.

Phase 8 Implementation:
- Hidden States: Candidate road segments (arcs) retrieved per GPS observation.
- Emission Probabilities: Probabilistic likelihood from candidate scoring / Gaussian distance.
- Transition Probabilities: Road-network graph reachability, comparing shortest path
  network distance against straight-line GPS displacement, with gap handling.
- Disconnected Handling: Soft topological fallback penalty for broken links / fragmented components.
- GPS Gap Handling: Automatic chain segmentation / reset for timestamps delta_t > threshold.
- Viterbi Decoding: Dynamic programming in log-space to find the globally optimal route.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from scipy.spatial import KDTree

from pathsnap.config import CONFIG
from pathsnap.data_io import Trip
from pathsnap.logging_utils import get_logger

logger = get_logger("pathsnap.hmm", log_file="hmm.log")


@dataclass
class HMMConfig:
    """Hyperparameters for HMM & Viterbi route decoding."""

    k_candidates: int = 6           # Number of candidate arcs per observation
    search_radius_m: float = 60.0   # Maximum radius to search for candidate arcs
    sigma_z: float = 12.0           # GPS observation noise standard deviation (meters)
    beta: float = 25.0              # Transition distance deviation scale (meters)
    gap_threshold_s: float = 15.0   # Temporal gap threshold to reset Markov transition (seconds)
    disconnect_penalty_m: float = 120.0  # Fallback penalty for disconnected graph transitions
    max_network_hop_m: float = 300.0     # Max shortest path distance before fallback penalty


@dataclass
class RouteReconstructionResult:
    """Output of HMM Viterbi route reconstruction."""

    trip_id: str
    n_observations: int
    pred_arcs: List[int]            # Pointwise arc prediction for each observation (length T)
    reconstructed_route: List[int]  # Collapsed sequence of unique traversed arcs
    classifier_only_arcs: List[int] # Pointwise prediction using classifier / nearest distance
    pointwise_accuracy_hmm: Optional[float] = None
    pointwise_accuracy_classifier: Optional[float] = None
    continuity_hmm: Optional[float] = None
    continuity_classifier: Optional[float] = None
    log_likelihood: float = 0.0


def to_local_cartesian(coords: np.ndarray, lon0: float, lat0: float) -> np.ndarray:
    """Project spherical (lon, lat) to local metric (x, y) in meters."""
    re = 6371000.0
    x = (coords[:, 0] - lon0) * (np.pi / 180.0) * re * np.cos(np.radians(lat0))
    y = (coords[:, 1] - lat0) * (np.pi / 180.0) * re
    return np.column_stack([x, y])


def point_to_segment_projection(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Compute orthogonal distance and normalized projection ratio t in [0, 1]."""
    v = b - a
    u = p - a
    l2 = float(np.dot(v, v))
    if l2 < 1e-10:
        return float(np.linalg.norm(u)), 0.0
    t = float(np.dot(u, v) / l2)
    t_clamped = float(np.clip(t, 0.0, 1.0))
    proj = a + t_clamped * v
    dist = float(np.linalg.norm(p - proj))
    return dist, t


class RoadNetworkGraph:
    """Directed road network graph with shortest path distance caching."""

    def __init__(self, trip: Trip, node_xy: np.ndarray):
        self.trip = trip
        self.node_xy = node_xy
        self.arcs = trip.arcs
        self.n_arcs = len(trip.arcs)

        # Precompute arc lengths
        p1 = node_xy[self.arcs[:, 0]]
        p2 = node_xy[self.arcs[:, 1]]
        self.arc_lengths = np.linalg.norm(p2 - p1, axis=1)

        # Build NetworkX DiGraph
        self.G = nx.DiGraph()
        for aid in range(self.n_arcs):
            u, v = int(self.arcs[aid, 0]), int(self.arcs[aid, 1])
            l = float(self.arc_lengths[aid])
            self.G.add_edge(u, v, weight=l, arc_id=aid)

        self._sp_cache: Dict[Tuple[int, int], Optional[float]] = {}

    def get_shortest_path_distance(self, u_node: int, v_node: int, max_cutoff: float = 300.0) -> Optional[float]:
        """Query shortest path distance with memory caching."""
        pair = (u_node, v_node)
        if pair in self._sp_cache:
            return self._sp_cache[pair]

        if u_node == v_node:
            res = 0.0
        else:
            try:
                res = nx.shortest_path_length(self.G, u_node, v_node, weight="weight")
                if res > max_cutoff:
                    res = None
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                res = None

        self._sp_cache[pair] = res
        return res


class SpatialCandidateIndex:
    """KDTree-based dense spatial index for arc candidate retrieval."""

    def __init__(self, trip: Trip, node_xy: np.ndarray, sample_step_m: float = 30.0):
        self.trip = trip
        self.node_xy = node_xy
        p1 = node_xy[trip.arcs[:, 0]]
        p2 = node_xy[trip.arcs[:, 1]]
        arc_lens = np.linalg.norm(p2 - p1, axis=1)

        sampled_pts = []
        sampled_arc_ids = []

        for aid in range(len(trip.arcs)):
            l = arc_lens[aid]
            if l <= sample_step_m:
                sampled_pts.append(0.5 * (p1[aid] + p2[aid]))
                sampled_arc_ids.append(aid)
            else:
                n_samples = int(np.ceil(l / sample_step_m)) + 1
                for a in np.linspace(0, 1, n_samples):
                    sampled_pts.append((1.0 - a) * p1[aid] + a * p2[aid])
                    sampled_arc_ids.append(aid)

        self.sampled_pts = np.array(sampled_pts)
        self.sampled_arc_ids = np.array(sampled_arc_ids, dtype=int)
        self.tree = KDTree(self.sampled_pts)

    def query_candidates(
        self, point_xy: np.ndarray, k: int = 6, search_radius: float = 60.0
    ) -> Tuple[List[int], List[float], List[float]]:
        """Return top-K candidate arcs, perpendicular distances, and projection fractions."""
        # Query nearest sampled points within radius
        indices = self.tree.query_ball_point(point_xy, r=search_radius)
        if len(indices) == 0:
            # Fallback to nearest neighbors if ball query is empty
            _, nn_indices = self.tree.query(point_xy, k=min(k * 3, len(self.sampled_pts)))
            indices = list(nn_indices) if isinstance(nn_indices, np.ndarray) else [nn_indices]

        # Deduplicate arc IDs and calculate exact perpendicular distance
        seen_arcs: Set[int] = set()
        cands = []
        dists = []
        projs = []

        for idx in indices:
            aid = int(self.sampled_arc_ids[idx])
            if aid not in seen_arcs:
                seen_arcs.add(aid)
                u = int(self.trip.arcs[aid, 0])
                v = int(self.trip.arcs[aid, 1])
                d, t_proj = point_to_segment_projection(point_xy, self.node_xy[u], self.node_xy[v])
                cands.append(aid)
                dists.append(d)
                projs.append(t_proj)

        # Sort by perpendicular distance
        sort_order = np.argsort(dists)
        cands = [cands[i] for i in sort_order[:k]]
        dists = [dists[i] for i in sort_order[:k]]
        projs = [projs[i] for i in sort_order[:k]]

        return cands, dists, projs


def derive_ground_truth_labels(trip: Trip, track_xy: np.ndarray, node_xy: np.ndarray) -> np.ndarray:
    """Project GPS observations monotonically onto the ground-truth route polyline."""
    route = trip.route
    r_idx = 0
    true_arcs = []

    for p in track_xy:
        best_dist = 1e9
        best_r = r_idx
        best_aid = int(route[r_idx])
        max_k = min(len(route), r_idx + 8)

        for k in range(r_idx, max_k):
            aid = int(route[k])
            u, v = int(trip.arcs[aid, 0]), int(trip.arcs[aid, 1])
            dist, _ = point_to_segment_projection(p, node_xy[u], node_xy[v])
            if dist < best_dist:
                best_dist = dist
                best_r = k
                best_aid = aid

        r_idx = best_r
        true_arcs.append(best_aid)

    return np.array(true_arcs, dtype=int)


def calculate_route_continuity(pred_arcs: List[int], graph: RoadNetworkGraph) -> float:
    """Compute the percentage of valid transitions along the reconstructed sequence."""
    total = len(pred_arcs) - 1
    if total <= 0:
        return 1.0

    valid = 0
    for t in range(total):
        a1, a2 = pred_arcs[t], pred_arcs[t + 1]
        if a1 == a2:
            valid += 1
        else:
            u_end = int(graph.arcs[a1, 1])
            v_start = int(graph.arcs[a2, 0])
            if u_end == v_start:
                valid += 1
            else:
                sp = graph.get_shortest_path_distance(u_end, v_start, max_cutoff=150.0)
                if sp is not None:
                    valid += 1

    return float(valid / total)


def viterbi_decode(
    trip: Trip,
    config: Optional[HMMConfig] = None,
    classifier_scores: Optional[List[Dict[int, float]]] = None,
) -> RouteReconstructionResult:
    """Execute HMM + Viterbi route reconstruction on a single trip.

    Parameters
    ----------
    trip:
        Loaded :class:`Trip` dataclass containing track, nodes, arcs, and route.
    config:
        Hyperparameters for candidate retrieval, emission, and transition models.
    classifier_scores:
        Optional precomputed candidate probabilities {arc_id: prob} from trained classifier.
        If None, calibrated distance likelihood is used for emissions.

    Returns
    -------
    RouteReconstructionResult
        Reconstructed route, pointwise arc predictions, and evaluation metrics.
    """
    if config is None:
        config = HMMConfig()

    T = trip.n_observations
    lon0, lat0 = float(trip.track[0, 0]), float(trip.track[0, 1])
    node_xy = to_local_cartesian(trip.nodes, lon0, lat0)
    track_xy = to_local_cartesian(trip.track[:, :2], lon0, lat0)
    timestamps = trip.track[:, 2]

    # Initialize graph and spatial index
    graph = RoadNetworkGraph(trip, node_xy)
    spatial_index = SpatialCandidateIndex(trip, node_xy, sample_step_m=30.0)

    # 1. Retrieve candidates for all observations
    cand_arcs: List[List[int]] = []
    cand_dists: List[List[float]] = []
    cand_projs: List[List[float]] = []

    for t in range(T):
        c, d, pr = spatial_index.query_candidates(
            track_xy[t], k=config.k_candidates, search_radius=config.search_radius_m
        )
        cand_arcs.append(c)
        cand_dists.append(d)
        cand_projs.append(pr)

    # Classifier-only pointwise baseline (nearest / highest score)
    classifier_only_arcs = [cand_arcs[t][int(np.argmin(cand_dists[t]))] for t in range(T)]

    # 2. Viterbi Dynamic Programming Matrices
    # V[t][j] = max log-probability ending at state j at time t
    # BP[t][j] = backpointer to state i at time t-1
    V = [np.zeros(len(cand_arcs[t]), dtype=float) for t in range(T)]
    BP = [np.zeros(len(cand_arcs[t]), dtype=int) for t in range(T)]

    # Initial step t = 0
    n_0 = len(cand_arcs[0])
    for i in range(n_0):
        d = cand_dists[0][i]
        # Emission log probability
        if classifier_scores and cand_arcs[0][i] in classifier_scores[0]:
            p = max(classifier_scores[0][cand_arcs[0][i]], 1e-6)
            log_emit = np.log(p)
        else:
            log_emit = -0.5 * (d / config.sigma_z) ** 2
        V[0][i] = log_emit - np.log(n_0)

    # Recurrence: t = 1 to T-1
    for t in range(1, T):
        dt = float(timestamps[t] - timestamps[t - 1])
        d_gps = float(np.linalg.norm(track_xy[t] - track_xy[t - 1]))
        is_gap = dt > config.gap_threshold_s

        n_prev = len(cand_arcs[t - 1])
        n_curr = len(cand_arcs[t])

        for j in range(n_curr):
            curr_aid = cand_arcs[t][j]
            curr_u, curr_v = int(trip.arcs[curr_aid, 0]), int(trip.arcs[curr_aid, 1])

            # Emission log-prob
            d_emit = cand_dists[t][j]
            if classifier_scores and curr_aid in classifier_scores[t]:
                p = max(classifier_scores[t][curr_aid], 1e-6)
                log_emit = np.log(p)
            else:
                log_emit = -0.5 * (d_emit / config.sigma_z) ** 2

            best_val = -1e12
            best_i = 0

            for i in range(n_prev):
                prev_aid = cand_arcs[t - 1][i]
                prev_u, prev_v = int(trip.arcs[prev_aid, 0]), int(trip.arcs[prev_aid, 1])

                if is_gap:
                    # Reset transition across large temporal gap
                    log_trans = -np.log(n_curr)
                else:
                    if prev_aid == curr_aid:
                        # Same arc transition
                        d_net = d_gps
                    elif prev_v == curr_u:
                        # Direct head-to-tail junction connection
                        d_net = 0.0
                    else:
                        # Shortest path network distance
                        sp = graph.get_shortest_path_distance(prev_v, curr_u, max_cutoff=config.max_network_hop_m)
                        if sp is not None:
                            d_net = sp
                        else:
                            # Soft fallback penalty for broken links / fragmented components
                            node_gap = float(np.linalg.norm(node_xy[prev_v] - node_xy[curr_u]))
                            d_net = node_gap + config.disconnect_penalty_m

                    delta_dist = abs(d_net - d_gps)
                    log_trans = -delta_dist / config.beta

                val = V[t - 1][i] + log_trans
                if val > best_val:
                    best_val = val
                    best_i = i

            V[t][j] = best_val + log_emit
            BP[t][j] = best_i

    # 3. Backtracking
    pred_hmm = [0] * T
    curr_state_idx = int(np.argmax(V[T - 1]))
    pred_hmm[T - 1] = cand_arcs[T - 1][curr_state_idx]

    for t in range(T - 2, -1, -1):
        curr_state_idx = int(BP[t + 1][curr_state_idx])
        pred_hmm[t] = cand_arcs[t][curr_state_idx]

    # 4. Collapse consecutive duplicate arcs to form reconstructed route
    reconstructed_route = []
    for aid in pred_hmm:
        if not reconstructed_route or reconstructed_route[-1] != aid:
            reconstructed_route.append(aid)

    # 5. Evaluate accuracy against ground truth route
    true_labels = derive_ground_truth_labels(trip, track_xy, node_xy)
    acc_hmm = float(np.mean(np.array(pred_hmm) == true_labels))
    acc_clf = float(np.mean(np.array(classifier_only_arcs) == true_labels))

    cont_hmm = calculate_route_continuity(pred_hmm, graph)
    cont_clf = calculate_route_continuity(classifier_only_arcs, graph)

    return RouteReconstructionResult(
        trip_id=trip.trip_id,
        n_observations=T,
        pred_arcs=pred_hmm,
        reconstructed_route=reconstructed_route,
        classifier_only_arcs=classifier_only_arcs,
        pointwise_accuracy_hmm=acc_hmm,
        pointwise_accuracy_classifier=acc_clf,
        continuity_hmm=cont_hmm,
        continuity_classifier=cont_clf,
        log_likelihood=float(np.max(V[T - 1])),
    )
