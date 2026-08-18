// In dev the Vite proxy (vite.config.js) forwards /api → http://localhost:8000.
// In production the frontend is served from the same origin as the API,
// so relative /api paths resolve correctly without any hardcoded host.
export const BASE = "/api";

export const resolveApiUrl = (path) => {
  if (!path) return null;

  // Already an absolute URL — keep as-is
  if (path.startsWith("http://") || path.startsWith("https://")) {
    return path;
  }

  // Backend returns paths like /api/storage/... or /api/dataset/image/...
  // These are already relative — return them as-is so the browser resolves
  // them against the current origin (proxy in dev, real host in prod).
  if (path.startsWith("/api") || path.startsWith("/storage")) {
    return path;
  }

  return `${BASE}/${path.replace(/^\/+/, "")}`;
};

async function jsonFetch(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { const j = await res.json(); detail = j.detail || detail; } catch (_) { }
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  health: () => jsonFetch(`${BASE}/health`),
  models: () => jsonFetch(`${BASE}/models`),
  modelsCompare: () => jsonFetch(`${BASE}/models/compare`),
  datasetStats: () => jsonFetch(`${BASE}/dataset/stats`),
  datasetGallery: (split, limit = 24) => jsonFetch(`${BASE}/dataset/gallery/${split}?limit=${limit}`),
  results: (limit = 50) => jsonFetch(`${BASE}/results?limit=${limit}`),
  experiments: (limit = 50) => jsonFetch(`${BASE}/experiments?limit=${limit}`),
  createExperiment: (payload) => jsonFetch(`${BASE}/experiments`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }),
  analyticsSummary: () => jsonFetch(`${BASE}/analytics/summary`),
  analyticsTrends: (modelName) => jsonFetch(`${BASE}/analytics/trends${modelName ? `?model_name=${modelName}` : ""}`),
  trainingHistory: (modelName) => jsonFetch(`${BASE}/analytics/training-history/${modelName}`),
  restore: (formData) => jsonFetch(`${BASE}/restore`, { method: "POST", body: formData }),
  evaluate: (formData) => jsonFetch(`${BASE}/evaluate`, { method: "POST", body: formData }),
  generateReport: (resultId) => jsonFetch(`${BASE}/reports/${resultId}`, { method: "POST" }),
};
