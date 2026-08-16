const BASE = "https://semiconductor-image-restoration-api.onrender.com/api";

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
