// PathSnap Evaluator Frontend Logic
document.addEventListener("DOMContentLoaded", () => {
  const apiUrlInput = document.getElementById("apiUrlInput");
  const testApiBtn = document.getElementById("testApiBtn");
  const apiStatusBadge = document.getElementById("apiStatusBadge");
  const tripSelect = document.getElementById("tripSelect");
  const customTripInput = document.getElementById("customTripInput");
  const runBtn = document.getElementById("runBtn");
  const btnText = document.getElementById("btnText");
  const btnSpinner = document.getElementById("btnSpinner");

  // Metrics elements
  const metricsPlaceholder = document.getElementById("metricsPlaceholder");
  const metricsContent = document.getElementById("metricsContent");
  const metricAccuracy = document.getElementById("metricAccuracy");
  const metricAccBaseline = document.getElementById("metricAccBaseline");
  const metricContinuity = document.getElementById("metricContinuity");
  const metricContBaseline = document.getElementById("metricContBaseline");
  const metricLatency = document.getElementById("metricLatency");
  const metricObsCount = document.getElementById("metricObsCount");
  const metricArcsCount = document.getElementById("metricArcsCount");
  const metricArcsGT = document.getElementById("metricArcsGT");
  const metricGraphSize = document.getElementById("metricGraphSize");
  const metricNodesCount = document.getElementById("metricNodesCount");

  // Layer toggles
  const layerGps = document.getElementById("layerGps");
  const layerReconstructed = document.getElementById("layerReconstructed");
  const layerReference = document.getElementById("layerReference");

  // Initialize Leaflet Map
  const map = L.map("map", {
    center: [51.505, -0.09],
    zoom: 13,
    zoomControl: true,
  });

  // OpenStreetMap Base Tiles
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "© OpenStreetMap contributors",
    maxZoom: 19,
  }).addTo(map);

  // Layer Groups
  const gpsLayerGroup = L.featureGroup().addTo(map);
  const reconstructedLayerGroup = L.featureGroup().addTo(map);
  const referenceLayerGroup = L.featureGroup().addTo(map);

  // Layer Visibility Event Listeners
  layerGps.addEventListener("change", (e) => {
    if (e.target.checked) map.addLayer(gpsLayerGroup);
    else map.removeLayer(gpsLayerGroup);
  });

  layerReconstructed.addEventListener("change", (e) => {
    if (e.target.checked) map.addLayer(reconstructedLayerGroup);
    else map.removeLayer(reconstructedLayerGroup);
  });

  layerReference.addEventListener("change", (e) => {
    if (e.target.checked) map.addLayer(referenceLayerGroup);
    else map.removeLayer(referenceLayerGroup);
  });

  function getApiUrl() {
    let url = apiUrlInput.value.trim();
    if (url.endsWith("/")) url = url.slice(0, -1);
    return url;
  }

  // 1. Connect & Check Health
  async function checkApiConnection() {
    const baseUrl = getApiUrl();
    apiStatusBadge.textContent = "Connecting...";
    apiStatusBadge.className = "badge badge-disconnected";

    try {
      const res = await fetch(`${baseUrl}/health`, { method: "GET" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (data.status === "healthy" || data.status === "ok") {
        apiStatusBadge.textContent = "Connected";
        apiStatusBadge.className = "badge badge-connected";
        loadTripsList();
      } else {
        throw new Error("Invalid status");
      }
    } catch (err) {
      console.error("Health check error:", err);
      apiStatusBadge.textContent = "Offline / Unreachable";
      apiStatusBadge.className = "badge badge-disconnected";
    }
  }

  // 2. Fetch Trips List
  async function loadTripsList() {
    const baseUrl = getApiUrl();
    try {
      const res = await fetch(`${baseUrl}/trips?partition=student_test`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      
      tripSelect.innerHTML = "";
      if (data.trips && data.trips.length > 0) {
        data.trips.forEach((tid, idx) => {
          const opt = document.createElement("option");
          opt.value = tid;
          opt.textContent = `Trip ${tid} (Test #${idx + 1})`;
          tripSelect.appendChild(opt);
        });
      } else {
        tripSelect.innerHTML = `<option value="" disabled>No trips returned</option>`;
      }
    } catch (err) {
      console.warn("Could not load trips automatically:", err);
      tripSelect.innerHTML = `
        <option value="00000000">Trip 00000000</option>
        <option value="00000007">Trip 00000007</option>
        <option value="00000009">Trip 00000009</option>
        <option value="00000040">Trip 00000040</option>
        <option value="00000052">Trip 00000052</option>
        <option value="00000056">Trip 00000056</option>
        <option value="00000065">Trip 00000065</option>
        <option value="00000076">Trip 00000076</option>
        <option value="00000077">Trip 00000077</option>
        <option value="00000091">Trip 00000091</option>
      `;
    }
  }

  testApiBtn.addEventListener("click", checkApiConnection);

  // 3. Execute Route Reconstruction
  runBtn.addEventListener("click", async () => {
    const selectedTrip = customTripInput.value.trim() || tripSelect.value;
    if (!selectedTrip) {
      alert("Please select or enter a valid Trip ID.");
      return;
    }

    const baseUrl = getApiUrl();
    setLoading(true);

    try {
      const payload = {
        trip_id: selectedTrip,
        partition: "student_test",
        include_ground_truth: true,
      };

      const res = await fetch(`${baseUrl}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server responded with code ${res.status}`);
      }

      const data = await res.json();
      renderResults(data);

    } catch (err) {
      console.error("Inference failed:", err);
      alert(`Error during route reconstruction:\n${err.message}`);
    } finally {
      setLoading(false);
    }
  });

  function setLoading(isLoading) {
    if (isLoading) {
      runBtn.disabled = true;
      btnText.textContent = "Reconstructing...";
      btnSpinner.classList.remove("hidden");
    } else {
      runBtn.disabled = false;
      btnText.textContent = "Run Route Reconstruction";
      btnSpinner.classList.add("hidden");
    }
  }

  // 4. Render Map Layers & Update Dashboard
  function renderResults(data) {
    // Clear old layers
    gpsLayerGroup.clearLayers();
    reconstructedLayerGroup.clearLayers();
    referenceLayerGroup.clearLayers();

    const bounds = L.latLngBounds([]);

    // Layer 1: Raw GPS Points (Blue)
    if (data.gps_points && data.gps_points.length > 0) {
      data.gps_points.forEach((pt, idx) => {
        const latLng = [pt[0], pt[1]];
        bounds.extend(latLng);

        // Subsample markers slightly for performance if trip has > 2000 points
        if (data.gps_points.length < 1500 || idx % 2 === 0) {
          L.circleMarker(latLng, {
            radius: 3,
            color: "#2563eb",
            fillColor: "#3b82f6",
            fillOpacity: 0.7,
            weight: 1,
          }).bindPopup(`<b>Observation #${idx + 1}</b><br>Lat: ${pt[0]}<br>Lon: ${pt[1]}`)
            .addTo(gpsLayerGroup);
        }
      });
    }

    // Layer 2: Ground Truth Route (Green)
    if (data.reference_segments && data.reference_segments.length > 0) {
      data.reference_segments.forEach((seg, i) => {
        bounds.extend(seg[0]);
        bounds.extend(seg[1]);
        L.polyline(seg, {
          color: "#10b981",
          weight: 5,
          opacity: 0.75,
        }).bindPopup(`<b>Ground Truth Arc #${i + 1}</b>`)
          .addTo(referenceLayerGroup);
      });
    }

    // Layer 3: Reconstructed HMM Route (Red Dashed)
    if (data.predicted_segments && data.predicted_segments.length > 0) {
      data.predicted_segments.forEach((seg, i) => {
        bounds.extend(seg[0]);
        bounds.extend(seg[1]);
        L.polyline(seg, {
          color: "#ef4444",
          weight: 4,
          opacity: 0.9,
          dashArray: "7, 5",
        }).bindPopup(`<b>Reconstructed Arc #${i + 1}</b> (HMM Viterbi)`)
          .addTo(reconstructedLayerGroup);
      });
    }

    // Zoom map to fit trajectory extent
    if (bounds.isValid()) {
      map.fitBounds(bounds, { padding: [40, 40] });
    }

    // Update Metrics Dashboard
    metricsPlaceholder.classList.add("hidden");
    metricsContent.classList.remove("hidden");

    const m = data.metrics || {};
    metricAccuracy.textContent = `${m.pointwise_accuracy_hmm ?? "--"}%`;
    metricAccBaseline.textContent = `vs ${m.pointwise_accuracy_baseline ?? "--"}% baseline`;

    metricContinuity.textContent = `${m.continuity_hmm ?? "--"}%`;
    metricContBaseline.textContent = `vs ${m.continuity_baseline ?? "--"}% baseline`;

    metricLatency.textContent = `${data.inference_time_ms} ms`;
    metricObsCount.textContent = m.observation_count ?? "--";
    
    metricArcsCount.textContent = `${m.reconstructed_arcs_count ?? "--"} arcs`;
    metricArcsGT.textContent = `${m.ground_truth_arcs_count ?? "--"} ground truth`;

    metricGraphSize.textContent = `${m.road_arcs_count ?? "--"} arcs`;
    metricNodesCount.textContent = `${m.road_nodes_count ?? "--"} nodes`;
  }

  // Initial connection check on page load
  checkApiConnection();
});
