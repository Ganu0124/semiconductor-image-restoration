# Frontend tests

Component-level tests use Vitest + React Testing Library. To run them:

```bash
cd frontend
npm install --save-dev vitest @testing-library/react @testing-library/jest-dom jsdom
npx vitest run
```

`smoke.test.jsx` covers the four required areas from the project brief:
Dashboard renders, Image Restoration upload control renders, Result page metric
cards render "No results available" with no data, and metric formatting is correct.

These are not run automatically in this repository's CI-less environment (no
`vitest`/`jsdom` preinstalled) — install the dev dependencies above locally to run them.
The build itself (`npm run build`) is verified working and is the primary
correctness check used during development of this project.
