import { Routes, Route } from 'react-router-dom'
import Sidebar from './components/Sidebar.jsx'
import Dashboard from './pages/Dashboard.jsx'
import ImageRestoration from './pages/ImageRestoration.jsx'
import Evaluation from './pages/Evaluation.jsx'
import ModelComparison from './pages/ModelComparison.jsx'
import DatasetExplorer from './pages/DatasetExplorer.jsx'
import Experiments from './pages/Experiments.jsx'
import Analytics from './pages/Analytics.jsx'
import Reports from './pages/Reports.jsx'
import Settings from './pages/Settings.jsx'

export default function App() {
  return (
    <div className="app-shell">
      <Sidebar />
      <main className="main">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/restoration" element={<ImageRestoration />} />
          <Route path="/evaluation" element={<Evaluation />} />
          <Route path="/comparison" element={<ModelComparison />} />
          <Route path="/dataset" element={<DatasetExplorer />} />
          <Route path="/experiments" element={<Experiments />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/reports" element={<Reports />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </main>
    </div>
  )
}
