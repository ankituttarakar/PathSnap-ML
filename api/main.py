"""FastAPI backend application for PathSnap (Laptop 1).

Hosts:
- GET  /health   : Health & liveness probe
- GET  /trips    : List available evaluation trips
- POST /predict  : Execute map-matching pipeline & return reconstructed route + metrics

CORS enabled for distributed Two-Laptop deployment over local network / Wi-Fi.
Ground-truth evaluation is computed strictly server-side.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathsnap import __version__
from pathsnap.config import CONFIG
from pathsnap.data_io import list_trips, load_trip
from pathsnap.hmm import HMMConfig, viterbi_decode
from pathsnap.logging_utils import get_logger

logger = get_logger("pathsnap.api", log_file="api.log")

app = FastAPI(
    title="PathSnap API",
    version=__version__,
    description="Map-Matching & Route Reconstruction Backend (Laptop 1)",
)

# Enable CORS for Laptop 2 (UI client) over LAN / Wi-Fi
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    trip_id: str
    partition: Optional[str] = "student_test"
    include_ground_truth: Optional[bool] = True


class PredictResponse(BaseModel):
    trip_id: str
    status: str
    inference_time_ms: float
    metrics: Dict[str, Any]
    gps_points: List[List[float]]            # [[lat, lon], ...]
    predicted_segments: List[List[List[float]]] # [[[lat1, lon1], [lat2, lon2]], ...]
    reference_segments: Optional[List[List[List[float]]]] = None
    reconstructed_arc_ids: List[int]


@app.get("/health")
def health() -> dict:
    """Liveness probe to confirm backend is running."""
    return {
        "status": "healthy",
        "service": "PathSnap API",
        "version": __version__,
        "role": "Laptop 1 (Model & Pipeline Host)",
    }


@app.get("/trips")
def get_trips(partition: str = "student_test") -> dict:
    """List available evaluation trips."""
    try:
        trips = list_trips(partition)
        return {
            "partition": partition,
            "count": len(trips),
            "trips": trips,
        }
    except Exception as e:
        logger.error("Failed to list trips: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictResponse)
def predict_route(req: PredictRequest) -> PredictResponse:
    """Run full map-matching & route reconstruction on selected trip."""
    trip_id = req.trip_id.strip()
    partition = req.partition or "student_test"

    logger.info("Received inference request for trip_id='%s' in '%s'", trip_id, partition)
    start_time = time.perf_counter()

    try:
        trip = load_trip(trip_id, partition)
    except Exception as exc:
        logger.error("Trip '%s' not found: %s", trip_id, exc)
        raise HTTPException(status_code=404, detail=f"Trip '{trip_id}' not found in partition '{partition}'")

    try:
        # Run HMM Viterbi route reconstruction
        cfg = HMMConfig(
            k_candidates=6,
            search_radius_m=60.0,
            sigma_z=12.0,
            beta=25.0,
            gap_threshold_s=15.0,
            disconnect_penalty_m=120.0,
        )
        result = viterbi_decode(trip, config=cfg)
        inference_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        # 1. Format raw GPS points: [[lat, lon], ...]
        gps_pts = [[float(row[1]), float(row[0])] for row in trip.track]

        # 2. Format reconstructed route line segments: [[[lat1, lon1], [lat2, lon2]], ...]
        predicted_segs = []
        for aid in result.reconstructed_route:
            u, v = int(trip.arcs[aid, 0]), int(trip.arcs[aid, 1])
            p1 = [float(trip.nodes[u, 1]), float(trip.nodes[u, 0])]
            p2 = [float(trip.nodes[v, 1]), float(trip.nodes[v, 0])]
            predicted_segs.append([p1, p2])

        # 3. Format reference ground-truth route
        ref_segs = None
        if req.include_ground_truth and trip.route is not None:
            ref_segs = []
            for aid in trip.route:
                u, v = int(trip.arcs[aid, 0]), int(trip.arcs[aid, 1])
                p1 = [float(trip.nodes[u, 1]), float(trip.nodes[u, 0])]
                p2 = [float(trip.nodes[v, 1]), float(trip.nodes[v, 0])]
                ref_segs.append([p1, p2])

        metrics = {
            "observation_count": trip.n_observations,
            "road_nodes_count": trip.n_nodes,
            "road_arcs_count": trip.n_arcs,
            "reconstructed_arcs_count": len(result.reconstructed_route),
            "ground_truth_arcs_count": len(trip.route) if trip.route is not None else 0,
            "pointwise_accuracy_hmm": round((result.pointwise_accuracy_hmm or 0.0) * 100.0, 2),
            "pointwise_accuracy_baseline": round((result.pointwise_accuracy_classifier or 0.0) * 100.0, 2),
            "continuity_hmm": round((result.continuity_hmm or 0.0) * 100.0, 2),
            "continuity_baseline": round((result.continuity_classifier or 0.0) * 100.0, 2),
        }

        logger.info(
            "Completed inference for trip '%s' in %.2f ms | Accuracy: %.2f%% | Continuity: %.2f%%",
            trip_id,
            inference_time_ms,
            metrics["pointwise_accuracy_hmm"],
            metrics["continuity_hmm"],
        )

        return PredictResponse(
            trip_id=trip_id,
            status="success",
            inference_time_ms=inference_time_ms,
            metrics=metrics,
            gps_points=gps_pts,
            predicted_segments=predicted_segs,
            reference_segments=ref_segs,
            reconstructed_arc_ids=result.reconstructed_route,
        )

    except Exception as err:
        logger.error("Error executing route reconstruction on trip '%s': %s", trip_id, err, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Reconstruction failed: {err}")
