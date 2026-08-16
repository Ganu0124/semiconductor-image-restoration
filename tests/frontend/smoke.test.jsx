import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { BrowserRouter } from 'react-router-dom'
import MetricCard from '../../frontend/src/components/MetricCard.jsx'
import Sidebar from '../../frontend/src/components/Sidebar.jsx'

function withRouter(children) {
  return <BrowserRouter>{children}</BrowserRouter>
}

describe('MetricCard', () => {
  it('renders "No results available" when value is null', () => {
    render(<MetricCard label="PSNR" value={null} />)
    expect(screen.getByText(/No results available/i)).toBeInTheDocument()
  })

  it('renders a formatted numeric value', () => {
    render(<MetricCard label="PSNR" value={23.456} unit=" dB" precision={2} />)
    expect(screen.getByText(/23.46/)).toBeInTheDocument()
  })
})

describe('Sidebar', () => {
  it('renders the SEMI-VISION AI brand and all nav items', () => {
    render(withRouter(<Sidebar />))
    expect(screen.getByText('SEMI-VISION AI')).toBeInTheDocument()
    expect(screen.getByText('Dashboard')).toBeInTheDocument()
    expect(screen.getByText('Image Restoration')).toBeInTheDocument()
    expect(screen.getByText('Model Comparison')).toBeInTheDocument()
    expect(screen.getByText('Reports')).toBeInTheDocument()
  })
})
