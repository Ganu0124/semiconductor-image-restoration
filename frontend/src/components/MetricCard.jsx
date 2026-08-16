export default function MetricCard({ label, value, unit, precision = 3, emptyText = "No results available" }) {
  const hasValue = value !== null && value !== undefined;
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      {hasValue ? (
        <div className="metric-value">
          {typeof value === 'number' ? value.toFixed(precision) : value}
          {unit && <span className="unit">{unit}</span>}
        </div>
      ) : (
        <div className="metric-empty">{emptyText}</div>
      )}
    </div>
  );
}
