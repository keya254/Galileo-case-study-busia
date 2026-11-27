/* Galileo Crop Intelligence MVP: search, draw, on-demand Sentinel + history. */

const $ = (id) => document.getElementById(id);

const map = L.map("map", { zoomControl: true }).setView([0.4747, 34.2306], 14);

const satellite = L.tileLayer(
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
  { maxZoom: 19, attribution: "Imagery &copy; Esri, Maxar, Earthstar Geographics" }
);
const streets = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: "&copy; OpenStreetMap contributors",
});
satellite.addTo(map);
L.control.layers({ Satellite: satellite, Streets: streets }, null, { position: "topright" }).addTo(map);

const drawn = new L.FeatureGroup().addTo(map);
const resultLayer = L.layerGroup().addTo(map);
let fullOverlay = null;

const state = {
  lastGeometry: null,
  currentResult: null,
  currentAoi: null,
  polling: false,
};

map.addControl(
  new L.Control.Draw({
    position: "topleft",
    draw: {
      polygon: { allowIntersection: false, showArea: true },
      rectangle: {},
      marker: false,
      circle: false,
      circlemarker: false,
      polyline: false,
    },
    edit: { featureGroup: drawn, edit: false, remove: true },
  })
);

map.on(L.Draw.Event.CREATED, (event) => {
  drawn.clearLayers();
  drawn.addLayer(event.layer);
  const geometry = event.layer.toGeoJSON().geometry;
  state.lastGeometry = geometry;
  analyzeGeometry(geometry);
});

map.on(L.Draw.Event.DELETED, () => {
  if (drawn.getLayers().length === 0) {
    resultLayer.clearLayers();
    hideOutput();
  }
});

/* ---------------------------- UI helpers ---------------------------- */

function setStatus(message, kind) {
  const el = $("status");
  if (!message) {
    el.hidden = true;
    el.textContent = "";
    return;
  }
  el.hidden = false;
  el.className = `status ${kind || ""}`;
  el.textContent = message;
}

function hideOutput() {
  $("output").hidden = true;
  $("historySection").hidden = true;
}

function fmt(value, digits = 0) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return new Intl.NumberFormat("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value);
}

function pct(value) {
  return `${fmt(value * 100, 0)}%`;
}

function shortPeriod(result) {
  if (!result.start) return result.period || "-";
  return result.start.slice(0, 4) === result.end.slice(0, 4) ? result.start.slice(0, 4) : `${result.start.slice(0, 7)} to ${result.end.slice(0, 7)}`;
}

function getPeriod() {
  const start = $("startDate")?.value || "2019-01-01";
  const end = $("endDate")?.value || "2019-12-31";
  if (!start || !end) throw new Error("Choose a start and end date.");
  return { start, end };
}

function setPeriod(start, end) {
  const startEl = $("startDate");
  const endEl = $("endDate");
  if (startEl) startEl.value = start;
  if (endEl) endEl.value = end;
}

/* ---------------------------- analysis ---------------------------- */

async function analyzeGeometry(geometry) {
  state.lastGeometry = geometry;
  let period;
  try {
    period = getPeriod();
  } catch (error) {
    setStatus(error.message, "error");
    return;
  }
  setStatus("Submitting analysis...", "loading");
  hideOutput();

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ geometry, start: period.start, end: period.end }),
    });
    const text = await response.text();
    let data;
    try {
      data = JSON.parse(text);
    } catch {
      resultLayer.clearLayers();
      setStatus(
        `Server returned ${response.status} ${response.headers.get("content-type") || ""} instead of JSON. ` +
          "Rebuild the container and hard-refresh the browser.",
        "error"
      );
      return;
    }
    if (!response.ok) {
      resultLayer.clearLayers();
      setStatus(data.detail || "Analysis failed.", "error");
      return;
    }
    if (data.status === "done") {
      renderResult(data.result);
      loadHistory(data.result.aoi_key);
      return;
    }
    pollJob(data.job_id);
  } catch (error) {
    setStatus(`Network error: ${error.message}`, "error");
  }
}

async function pollJob(jobId) {
  const started = Date.now();
  state.polling = true;
  let lastMessage = "";
  while (state.polling) {
    await sleep(2000);
    try {
      const response = await fetch(`/api/jobs/${jobId}`);
      const text = await response.text();
      let job;
      try {
        job = JSON.parse(text);
      } catch {
        state.polling = false;
        setStatus(
          `Job endpoint returned ${response.status} ${response.headers.get("content-type") || ""} ` +
            "instead of JSON. Rebuild the container and hard-refresh.",
          "error"
        );
        return;
      }
      if (!response.ok) {
        setStatus(job.detail || "Job not found.", "error");
        state.polling = false;
        return;
      }
      const elapsed = Math.round((Date.now() - started) / 1000);
      const message = `${job.message || job.status} (${elapsed}s)`;
      if (message !== lastMessage) {
        setStatus(message, "loading");
        lastMessage = message;
      }
      if (job.status === "done") {
        state.polling = false;
        setStatus("");
        renderResult(job.result);
        loadHistory(job.result.aoi_key);
        return;
      }
      if (job.status === "error") {
        state.polling = false;
        setStatus(`Analysis failed: ${job.error || job.message}`, "error");
        return;
      }
    } catch (error) {
      state.polling = false;
      setStatus(`Job polling failed: ${error.message}`, "error");
    }
  }
}

function renderResult(data) {
  state.currentResult = data;
  state.currentAoi = data.aoi_key;
  $("output").hidden = false;

  const source = data.source === "demo" ? "cached demo" : "on-demand Sentinel";
  $("periodBadge").textContent = `${data.start} to ${data.end} · ${source}`;

  $("meanProb").textContent = pct(data.maize.mean_probability);
  $("probBar").style.width = `${Math.max(2, data.maize.mean_probability * 100)}%`;

  $("maizeHa").textContent = `${fmt(data.area.maize_ha, 1)} ha`;
  $("maizeAcres").textContent = `${fmt(data.area.maize_acres, 1)} acres`;
  $("mappedHa").textContent = `${fmt(data.area.mapped_ha, 1)} ha`;
  $("mappedAcres").textContent = `${fmt(data.area.mapped_acres, 1)} acres`;

  $("shareStrict").textContent = pct(data.maize.share_above_strict);
  $("shareCutoff").textContent = `${pct(data.maize.share_above_cutoff)} at cutoff ${data.maize.cutoff}`;

  $("confidence").textContent = data.quality.confidence_band;
  $("coverage").textContent = `${pct(data.quality.confidence)} confident${
    data.source === "demo" ? ` / ${fmt(data.quality.coverage_pct, 1)}% in demo` : ""
  }`;

  $("insuranceScore").textContent = `${data.insurance_proxy.score}/100`;
  $("insuranceBand").textContent = data.insurance_proxy.band;
  $("insuranceNote").textContent = data.insurance_proxy.note;

  resultLayer.clearLayers();
  const overlayUrl = data.overlay?.url || data.overlay?.image;
  if (overlayUrl && data.overlay?.bounds) {
    L.imageOverlay(overlayUrl, data.overlay.bounds, { opacity: 0.75, interactive: false }).addTo(
      resultLayer
    );
  }

  setPeriod(data.start, data.end);
  setStatus("");
}

/* ---------------------------- history / change ---------------------------- */

async function loadHistory(aoiKey) {
  if (!aoiKey) return;
  try {
    const items = await fetch(`/api/analyses?aoi_key=${encodeURIComponent(aoiKey)}`).then((r) =>
      r.json()
    );
    if (!Array.isArray(items) || items.length === 0) {
      $("historySection").hidden = true;
      return;
    }
    $("historySection").hidden = false;
    renderChangeSummary(items);
    renderHistoryChart(items);
    renderHistoryList(items);
  } catch (error) {
    $("historySection").hidden = true;
  }
}

function renderChangeSummary(items) {
  const el = $("changeSummary");
  el.replaceChildren();
  if (items.length < 2) {
    el.textContent = "Run another period for this same area to see what changed.";
    return;
  }
  const first = items[0];
  const last = items[items.length - 1];
  const dMaize = last.area.maize_ha - first.area.maize_ha;
  const dProb = (last.maize.mean_probability - first.maize.mean_probability) * 100;
  const sign = (v) => (v >= 0 ? "+" : "");
  const pctChange = first.area.maize_ha ? (dMaize / first.area.maize_ha) * 100 : 0;
  const span = `${shortPeriod(first)} to ${shortPeriod(last)}`;
  el.textContent = `${span}: maize area ${sign(dMaize)}${fmt(dMaize, 1)} ha (${sign(
    pctChange
  )}${fmt(pctChange, 0)}%), mean probability ${sign(dProb)}${fmt(dProb, 0)} points.`;
}

function renderHistoryChart(items) {
  const chart = $("historyChart");
  chart.replaceChildren();
  const max = Math.max(...items.map((i) => i.area.maize_ha), 0.1);
  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "chart-row";
    if (state.currentResult && item.period === state.currentResult.period) {
      row.classList.add("active");
    }
    const label = document.createElement("span");
    label.className = "chart-label";
    label.textContent = shortPeriod(item);
    const track = document.createElement("div");
    track.className = "chart-track";
    const bar = document.createElement("div");
    bar.className = "chart-bar";
    bar.style.width = `${Math.max(2, (item.area.maize_ha / max) * 100)}%`;
    track.appendChild(bar);
    const value = document.createElement("span");
    value.className = "chart-value";
    value.textContent = `${fmt(item.area.maize_ha, 1)} ha`;
    row.append(label, track, value);
    chart.appendChild(row);
  });
}

function renderHistoryList(items) {
  const list = $("historyList");
  list.replaceChildren();
  items.forEach((item) => {
    const li = document.createElement("li");
    const button = document.createElement("button");
    button.className = "history-item";
    if (state.currentResult && item.period === state.currentResult.period) {
      button.classList.add("active");
    }
    const period = document.createElement("span");
    period.className = "hi-period";
    period.textContent = shortPeriod(item);
    const meta = document.createElement("span");
    meta.className = "hi-meta";
    meta.textContent = `${fmt(item.area.maize_ha, 1)} ha · ${pct(
      item.maize.share_above_strict
    )} maize`;
    const source = document.createElement("span");
    source.className = "hi-source";
    source.textContent = item.source === "demo" ? "cache" : "sentinel";
    button.append(period, meta, source);
    button.addEventListener("click", () => loadAnalysis(item.aoi_key, item.period));
    li.appendChild(button);
    list.appendChild(li);
  });
}

async function loadAnalysis(aoiKey, period) {
  try {
    const item = await fetch(
      `/api/analyses/${encodeURIComponent(aoiKey)}/${encodeURIComponent(period)}`
    ).then((r) => r.json());
    if (item.detail) {
      setStatus(item.detail, "error");
      return;
    }
    renderResult(item);
    loadHistory(aoiKey);
  } catch (error) {
    setStatus(`Could not load saved analysis: ${error.message}`, "error");
  }
}

/* ---------------------------- actions ---------------------------- */

function boundsToPolygon(bounds) {
  const west = bounds.getWest();
  const south = bounds.getSouth();
  const east = bounds.getEast();
  const north = bounds.getNorth();
  return {
    type: "Polygon",
    coordinates: [
      [
        [west, south],
        [east, south],
        [east, north],
        [west, north],
        [west, south],
      ],
    ],
  };
}

async function analyzeCurrentView() {
  const bounds = map.getBounds();
  drawn.clearLayers();
  drawn.addLayer(L.rectangle(bounds, { color: "#1f7a4d", weight: 2 }));
  state.lastGeometry = boundsToPolygon(bounds);
  analyzeGeometry(state.lastGeometry);
}

async function analyzeDemoArea() {
  try {
    const meta = await fetch("/api/meta").then((r) => r.json());
    const bounds = L.latLngBounds(meta.bounds);
    map.fitBounds(bounds);
    drawn.clearLayers();
    drawn.addLayer(L.rectangle(bounds, { color: "#1f7a4d", weight: 2 }));
    state.lastGeometry = boundsToPolygon(bounds);
    analyzeGeometry(state.lastGeometry);
  } catch (error) {
    setStatus(`Could not load demo bounds: ${error.message}`, "error");
  }
}

function clearAll() {
  state.polling = false;
  state.lastGeometry = null;
  state.currentResult = null;
  state.currentAoi = null;
  drawn.clearLayers();
  resultLayer.clearLayers();
  hideOutput();
  setStatus("");
  $("results").hidden = true;
  $("q").value = "";
}

/* ---------------------------- search ---------------------------- */

async function runSearch() {
  const query = $("q").value.trim();
  if (!query) return;
  const list = $("results");
  list.hidden = true;
  list.replaceChildren();
  setStatus("Searching...", "loading");
  try {
    const url = `https://nominatim.openstreetmap.org/search?format=jsonv2&limit=5&q=${encodeURIComponent(
      query
    )}`;
    const items = await fetch(url, { headers: { Accept: "application/json" } }).then((r) => r.json());
    if (!items.length) {
      setStatus(`No place found for "${query}".`, "error");
      return;
    }
    setStatus("");
    items.forEach((item) => {
      const li = document.createElement("li");
      li.textContent = item.display_name;
      li.addEventListener("click", () => {
        const bbox = item.boundingbox.map(Number); // [south, north, west, east]
        map.fitBounds([
          [bbox[0], bbox[2]],
          [bbox[1], bbox[3]],
        ]);
        list.hidden = true;
        $("q").value = item.display_name;
      });
      list.appendChild(li);
    });
    list.hidden = false;
  } catch (error) {
    setStatus(`Search failed: ${error.message}`, "error");
  }
}

/* ---------------------------- wiring ---------------------------- */

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function buildQuickYears() {
  const container = $("quickYears");
  if (!container) return;
  container.replaceChildren();
  for (let year = 2019; year <= 2024; year += 1) {
    const button = document.createElement("button");
    button.textContent = year;
    button.addEventListener("click", () => {
      setPeriod(`${year}-01-01`, `${year}-12-31`);
      container.querySelectorAll("button").forEach((b) => b.classList.remove("active"));
      button.classList.add("active");
      if (state.lastGeometry) analyzeGeometry(state.lastGeometry);
    });
    container.appendChild(button);
  }
  const current = getPeriod();
  container.querySelectorAll("button").forEach((b) => {
    if (b.textContent === current.start.slice(0, 4) && current.start.endsWith("-01-01")) {
      b.classList.add("active");
    }
  });
}

$("searchBtn").addEventListener("click", runSearch);
$("q").addEventListener("keydown", (event) => {
  if (event.key === "Enter") runSearch();
});
$("demoBtn").addEventListener("click", analyzeDemoArea);
$("viewBtn").addEventListener("click", analyzeCurrentView);
$("clearBtn").addEventListener("click", clearAll);

$("startDate").addEventListener("change", () => {
  if (state.lastGeometry) analyzeGeometry(state.lastGeometry);
});
$("endDate").addEventListener("change", () => {
  if (state.lastGeometry) analyzeGeometry(state.lastGeometry);
});

$("showOverlay").addEventListener("change", async (event) => {
  if (event.target.checked) {
    if (!fullOverlay) {
      const meta = await fetch("/api/meta").then((r) => r.json());
      fullOverlay = L.imageOverlay(meta.overlay_url, meta.bounds, {
        opacity: 0.55,
        interactive: false,
      });
    }
    fullOverlay.addTo(map);
  } else if (fullOverlay) {
    map.removeLayer(fullOverlay);
  }
});

buildQuickYears();

fetch("/api/meta")
  .then((r) => r.json())
  .then((meta) => {
    map.fitBounds(meta.bounds);
    if (!meta.on_demand) {
      setStatus(
        "On-demand Sentinel is off (no EE_PROJECT/EE_BUCKET). The cached 2019 Busia tile still works.",
        "error"
      );
    }
  })
  .catch(() => {});