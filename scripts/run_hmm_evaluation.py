"""Evaluate HMM + Viterbi route reconstruction across student-test trips.

Compares:
1. Classifier-Only Pointwise Prediction (nearest distance baseline).
2. Classifier + HMM Viterbi Sequential Reconstruction.

Generates:
- outputs/experiments/hmm_evaluation_results.csv
- outputs/experiments/hmm_performance_comparison.png
- outputs/maps/hmm_reconstruction_sample.html (interactive Folium map)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from pathsnap.config import CONFIG
from pathsnap.data_io import list_trips, load_trip
from pathsnap.hmm import HMMConfig, viterbi_decode
from pathsnap.logging_utils import get_logger

logger = get_logger("pathsnap.run_hmm", log_file="run_hmm.log")


def main() -> int:
    CONFIG.paths.ensure_output_dirs()
    exp_dir = CONFIG.paths.outputs_experiments
    maps_dir = CONFIG.paths.outputs_maps

    test_trips = list_trips("student_test")
    logger.info("Evaluating HMM across %d student_test trips...", len(test_trips))

    results = []
    sample_result = None
    sample_trip = None

    config = HMMConfig(
        k_candidates=6,
        search_radius_m=60.0,
        sigma_z=12.0,
        beta=25.0,
        gap_threshold_s=15.0,
        disconnect_penalty_m=120.0,
    )

    for trip_id in tqdm(test_trips, desc="HMM Route Reconstruction"):
        trip = load_trip(trip_id, "student_test")
        res = viterbi_decode(trip, config=config)

        if sample_result is None:
            sample_result = res
            sample_trip = trip

        results.append(
            {
                "trip_id": trip_id,
                "n_obs": trip.n_observations,
                "ground_truth_route_len": len(trip.route),
                "reconstructed_route_len": len(res.reconstructed_route),
                "acc_classifier_only": res.pointwise_accuracy_classifier,
                "acc_hmm_viterbi": res.pointwise_accuracy_hmm,
                "continuity_classifier_only": res.continuity_classifier,
                "continuity_hmm_viterbi": res.continuity_hmm,
                "accuracy_gain_pct": (res.pointwise_accuracy_hmm - res.pointwise_accuracy_classifier) * 100.0,
                "continuity_gain_pct": (res.continuity_hmm - res.continuity_classifier) * 100.0,
            }
        )

    df = pd.DataFrame(results)
    csv_path = exp_dir / "hmm_evaluation_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Evaluation results saved to %s", csv_path)

    # Print summary table
    print("\n" + "=" * 80)
    print("PATHSNAP HMM + VITERBI ROUTE RECONSTRUCTION EVALUATION (student_test)")
    print("=" * 80)
    cols_to_print = [
        "trip_id",
        "n_obs",
        "acc_classifier_only",
        "acc_hmm_viterbi",
        "continuity_classifier_only",
        "continuity_hmm_viterbi",
    ]
    print(df[cols_to_print].to_string(index=False))
    print("-" * 80)
    print(
        f"MEAN ACCURACY:   Classifier-Only: {df['acc_classifier_only'].mean()*100:.2f}%  -->  HMM + Viterbi: {df['acc_hmm_viterbi'].mean()*100:.2f}%"
    )
    print(
        f"MEAN CONTINUITY: Classifier-Only: {df['continuity_classifier_only'].mean()*100:.2f}%  -->  HMM + Viterbi: {df['continuity_hmm_viterbi'].mean()*100:.2f}%"
    )
    print("=" * 80)

    # Generate Performance Comparison Chart
    plt.figure(figsize=(12, 6))

    # Subplot 1: Pointwise Top-1 Accuracy
    plt.subplot(1, 2, 1)
    x = np.arange(len(df))
    width = 0.35
    plt.bar(x - width / 2, df["acc_classifier_only"] * 100, width, label="Classifier-Only", color="#e74c3c", alpha=0.85)
    plt.bar(x + width / 2, df["acc_hmm_viterbi"] * 100, width, label="Classifier + HMM", color="#2ecc71", alpha=0.85)
    plt.xlabel("Student Test Trip Index", fontsize=11)
    plt.ylabel("Pointwise Accuracy (%)", fontsize=11)
    plt.title("Pointwise Top-1 Accuracy Improvement", fontsize=13, fontweight="bold")
    plt.xticks(x, [f"T{i}" for i in range(len(df))])
    plt.ylim([0, 105])
    plt.legend(frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Subplot 2: Route Continuity
    plt.subplot(1, 2, 2)
    plt.bar(x - width / 2, df["continuity_classifier_only"] * 100, width, label="Classifier-Only", color="#e67e22", alpha=0.85)
    plt.bar(x + width / 2, df["continuity_hmm_viterbi"] * 100, width, label="Classifier + HMM", color="#3498db", alpha=0.85)
    plt.xlabel("Student Test Trip Index", fontsize=11)
    plt.ylabel("Route Continuity (%)", fontsize=11)
    plt.title("Route Topological Continuity", fontsize=13, fontweight="bold")
    plt.xticks(x, [f"T{i}" for i in range(len(df))])
    plt.ylim([70, 105])
    plt.legend(frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    chart_path = exp_dir / "hmm_performance_comparison.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()
    logger.info("Comparison chart saved to %s", chart_path)

    # Generate Interactive Folium Visualisation for the sample trip
    if sample_trip and sample_result:
        m_center = [float(sample_trip.track[:, 1].mean()), float(sample_trip.track[:, 0].mean())]
        fmap = folium.Map(location=m_center, zoom_start=14, tiles="OpenStreetMap")

        # Layer 1: Raw GPS observations (Blue)
        gps_group = folium.FeatureGroup(name="Raw GPS (Noisy)")
        for row in sample_trip.track[::3]:  # sub-sample slightly for smooth rendering
            folium.CircleMarker(
                location=[row[1], row[0]],
                radius=2.5,
                color="#2980b9",
                fill=True,
                fill_color="#3498db",
                fill_opacity=0.7,
                popup=f"GPS t={row[2]}",
            ).add_to(gps_group)
        gps_group.add_to(fmap)

        # Layer 2: Ground Truth Route (Green)
        gt_group = folium.FeatureGroup(name="Ground Truth Route")
        gt_lines = []
        for aid in sample_trip.route:
            u, v = sample_trip.arcs[aid]
            p1 = [sample_trip.nodes[u, 1], sample_trip.nodes[u, 0]]
            p2 = [sample_trip.nodes[v, 1], sample_trip.nodes[v, 0]]
            gt_lines.append([p1, p2])
        for seg in gt_lines:
            folium.PolyLine(seg, color="#27ae60", weight=5, opacity=0.8).add_to(gt_group)
        gt_group.add_to(fmap)

        # Layer 3: HMM Reconstructed Route (Red dashed)
        hmm_group = folium.FeatureGroup(name="HMM Viterbi Reconstructed Route")
        hmm_lines = []
        for aid in sample_result.reconstructed_route:
            u, v = sample_trip.arcs[aid]
            p1 = [sample_trip.nodes[u, 1], sample_trip.nodes[u, 0]]
            p2 = [sample_trip.nodes[v, 1], sample_trip.nodes[v, 0]]
            hmm_lines.append([p1, p2])
        for seg in hmm_lines:
            folium.PolyLine(seg, color="#c0392b", weight=3.5, dash_array="7, 5", opacity=0.9).add_to(hmm_group)
        hmm_group.add_to(fmap)

        folium.LayerControl(collapsed=False).add_to(fmap)
        map_path = maps_dir / f"hmm_reconstruction_trip_{sample_trip.trip_id}.html"
        fmap.save(str(map_path))
        logger.info("Sample interactive map saved to %s", map_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
