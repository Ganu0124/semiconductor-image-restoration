import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard, ScanEye, LineChart, GitCompareArrows,
  Images, FlaskConical, TrendingUp, FileText, Settings as SettingsIcon,
} from 'lucide-react'

const NAV = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard, end: true },
  { to: '/restoration', label: 'Image Restoration', icon: ScanEye },
  { to: '/evaluation', label: 'Evaluation', icon: LineChart },
  { to: '/comparison', label: 'Model Comparison', icon: GitCompareArrows },
  { to: '/dataset', label: 'Dataset Explorer', icon: Images },
  { to: '/experiments', label: 'Experiments', icon: FlaskConical },
  { to: '/analytics', label: 'Analytics', icon: TrendingUp },
  { to: '/reports', label: 'Reports', icon: FileText },
  { to: '/settings', label: 'Settings', icon: SettingsIcon },
]

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark">SV</div>
        <div className="brand-text">
          <div className="name">SEMI-VISION AI</div>
          <div className="tag">Restoration &amp; Inspection</div>
        </div>
      </div>
      <ul className="nav-list">
        {NAV.map(({ to, label, icon: Icon, end }) => (
          <li key={to}>
            <NavLink to={to} end={end} className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}>
              <Icon />
              <span>{label}</span>
            </NavLink>
          </li>
        ))}
      </ul>
      <div className="sidebar-footer">
        v1.0.0 · Local dev instance<br />
        Backend: <span className="mono">localhost:8000</span>
      </div>
    </aside>
  )
}
