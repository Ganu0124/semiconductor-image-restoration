// API base URL
// Local development: Vite proxy forwards /api -> http://localhost:8000/api
// Production: requests go directly to the Render backend.

const API_HOST = import.meta.env.VITE_API_URL || "/api";

export const BASE = `${API_HOST}/api`;

export const resolveApiUrl = (path) => {
  if (!path) return null;

  // Already an absolute URL
  if (path.startsWith("http://") || path.startsWith("https://")) {
    return path;
  }

  // Backend returns paths such as:
  // /api/storage/...
  // /api/dataset/image/...
  //
  // These must also point to the backend in production.
  if (path.startsWith("/api")) {
    return `${API_HOST}${path}`;
  }

  if (path.startsWith("/storage")) {
    return `${API_HOST}${path}`;
  }

  return `${BASE}/${path.replace(/^\/+/, "")}`;
};

async function jsonFetch(url, opts) {
  const res = await fetch(url, opts);

  if (!res.ok) {
    let detail = res.statusText;

    try {
      const j = await res.json();
      detail = j.detail || detail;
    } catch (_) { }

    throw new Error(detail);
  }

  return res.json();
}

export const api = {
  health: () => jsonFetch(`${BASE}/health`),

  models: () => jsonFetch(`${BASE}/models`),

  modelsCompare: () =>
    jsonFetch(`${BASE}/models/compare`),

  datasetStats: () =>
    jsonFetch(`${BASE}/dataset/stats`),

  datasetGallery: (split, limit = 24) =>
    jsonFetch(`${BASE}/dataset/gallery/${split}?limit=${limit}`),

  results: (limit = 50) =>
    jsonFetch(`${BASE}/results?limit=${limit}`),

  experiments: (limit = 50) =>
    jsonFetch(`${BASE}/experiments?limit=${limit}`),

  createExperiment: (payload) =>
    jsonFetch(`${BASE}/experiments`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    }),

  analyticsSummary: () =>
    jsonFetch(`${BASE}/analytics/summary`),

  analyticsTrends: (modelName) =>
    jsonFetch(
      `${BASE}/analytics/trends${modelName ? `?model_name=${modelName}` : ""
      }`
    ),

  trainingHistory: (modelName) =>
    jsonFetch(`${BASE}/analytics/training-history/${modelName}`),

  restore: (formData) =>
    jsonFetch(`${BASE}/restore`, {
      method: "POST",
      body: formData,
    }),

  evaluate: (formData) =>
    jsonFetch(`${BASE}/evaluate`, {
      method: "POST",
      body: formData,
    }),

  generateReport: (resultId) =>
    jsonFetch(`${BASE}/reports/${resultId}`, {
      method: "POST",
    }),
};