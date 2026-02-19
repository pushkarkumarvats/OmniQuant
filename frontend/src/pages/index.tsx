import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import Head from 'next/head';

// ─── Types ───────────────────────────────────────────────────────────────────

interface Position {
  symbol: string;
  quantity: number;
  avg_price: number;
  market_price: number;
  unrealized_pnl: number;
  notional: number;
}

interface RiskSummary {
  entity_id: string;
  daily_pnl: number;
  gross_notional: number;
  net_notional: number;
  current_equity: number;
  drawdown_pct: number;
  kill_switch: boolean;
  order_rate_1m: number;
  positions_count: number;
}

interface ReconStats {
  internal_fills: number;
  exchange_fills: number;
  matched_pairs: number;
  open_breaks: number;
  total_breaks: number;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function formatCurrency(value: number): string {
  const abs = Math.abs(value);
  if (abs >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `$${(value / 1e6).toFixed(2)}M`;
  if (abs >= 1e3) return `$${(value / 1e3).toFixed(1)}K`;
  return `$${value.toFixed(2)}`;
}

function formatPct(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

function pnlClass(value: number): string {
  if (value > 0) return 'pnl-positive';
  if (value < 0) return 'pnl-negative';
  return 'pnl-neutral';
}

// ─── Components ──────────────────────────────────────────────────────────────

function MetricCard({ label, value, colorClass }: {
  label: string; value: string; colorClass?: string;
}) {
  return (
    <div className="card">
      <div className="metric-label">{label}</div>
      <div className={`metric-value ${colorClass || ''}`}>{value}</div>
    </div>
  );
}

function PositionsTable({ positions }: { positions: Position[] }) {
  return (
    <div className="card" style={{ gridColumn: 'span 8' }}>
      <h3 className="text-sm font-semibold mb-3 uppercase tracking-wider text-gray-400">
        Positions
      </h3>
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-gray-500 border-b border-gray-700">
            <th className="pb-2">Symbol</th>
            <th className="pb-2 text-right">Qty</th>
            <th className="pb-2 text-right">Avg Price</th>
            <th className="pb-2 text-right">Mkt Price</th>
            <th className="pb-2 text-right">Notional</th>
            <th className="pb-2 text-right">Unrealized P&L</th>
          </tr>
        </thead>
        <tbody>
          {positions.map((pos) => (
            <tr key={pos.symbol} className="border-b border-gray-800">
              <td className="py-2 font-mono font-semibold">{pos.symbol}</td>
              <td className="py-2 text-right font-mono">
                {pos.quantity.toLocaleString()}
              </td>
              <td className="py-2 text-right font-mono">
                ${pos.avg_price.toFixed(2)}
              </td>
              <td className="py-2 text-right font-mono">
                ${pos.market_price.toFixed(2)}
              </td>
              <td className="py-2 text-right font-mono">
                {formatCurrency(pos.notional)}
              </td>
              <td className={`py-2 text-right font-mono font-semibold ${pnlClass(pos.unrealized_pnl)}`}>
                {formatCurrency(pos.unrealized_pnl)}
              </td>
            </tr>
          ))}
          {positions.length === 0 && (
            <tr>
              <td colSpan={6} className="py-4 text-center text-gray-500">
                No positions
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

function KillSwitchPanel({ active, entityId }: { active: boolean; entityId: string }) {
  const [loading, setLoading] = useState(false);

  const toggleKillSwitch = async () => {
    setLoading(true);
    const endpoint = active ? 'deactivate' : 'activate';
    await fetch(`/api/risk/kill-switch/${entityId}/${endpoint}`, { method: 'POST' });
    setLoading(false);
  };

  return (
    <div className="card" style={{ gridColumn: 'span 4' }}>
      <h3 className="text-sm font-semibold mb-3 uppercase tracking-wider text-gray-400">
        Kill Switch
      </h3>
      <div className="flex flex-col items-center gap-4">
        <div className={`w-16 h-16 rounded-full flex items-center justify-center text-2xl font-bold
          ${active ? 'bg-red-900 text-red-400 animate-pulse' : 'bg-green-900 text-green-400'}`}>
          {active ? '⛔' : '✓'}
        </div>
        <div className="text-sm text-gray-400">
          {active ? 'KILL SWITCH ACTIVE - ALL ORDERS BLOCKED' : 'Trading Enabled'}
        </div>
        <button
          className="kill-switch-btn"
          onClick={toggleKillSwitch}
          disabled={loading}
        >
          {loading ? '...' : active ? 'Deactivate' : 'Emergency Stop'}
        </button>
      </div>
    </div>
  );
}

function ReconciliationPanel({ stats }: { stats: ReconStats | null }) {
  if (!stats) return null;
  const matchRate = stats.matched_pairs / Math.max(stats.internal_fills, stats.exchange_fills, 1) * 100;

  return (
    <div className="card" style={{ gridColumn: 'span 4' }}>
      <h3 className="text-sm font-semibold mb-3 uppercase tracking-wider text-gray-400">
        Reconciliation
      </h3>
      <div className="space-y-2 text-sm">
        <div className="flex justify-between">
          <span className="text-gray-400">Match Rate</span>
          <span className={matchRate >= 99 ? 'text-green-400' : 'text-yellow-400'}>
            {matchRate.toFixed(1)}%
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">Matched</span>
          <span>{stats.matched_pairs}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">Open Breaks</span>
          <span className={stats.open_breaks > 0 ? 'text-red-400 font-semibold' : ''}>
            {stats.open_breaks}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">Internal Fills</span>
          <span>{stats.internal_fills}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">Exchange Fills</span>
          <span>{stats.exchange_fills}</span>
        </div>
      </div>
    </div>
  );
}

// ─── Main Dashboard ──────────────────────────────────────────────────────────

export default function Dashboard() {
  const [positions, setPositions] = useState<Position[]>([]);

  // REST queries
  const { data: riskSummary } = useQuery<RiskSummary>({
    queryKey: ['riskSummary'],
    queryFn: () => fetch('/api/risk/summary').then(r => r.json()),
  });

  const { data: reconStats } = useQuery<ReconStats>({
    queryKey: ['reconStats'],
    queryFn: () => fetch('/api/reconciliation/summary').then(r => r.json()),
  });

  // WebSocket for real-time positions
  useEffect(() => {
    let ws: WebSocket | null = null;
    try {
      ws = new WebSocket(`ws://${window.location.hostname}:8000/ws/positions`);
      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === 'positions') {
          setPositions(msg.data);
        }
      };
      ws.onerror = () => console.warn('WebSocket connection failed');
    } catch {
      console.warn('WebSocket not available');
    }
    return () => { ws?.close(); };
  }, []);

  return (
    <>
      <Head>
        <title>HRT Trading Dashboard</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <div className="min-h-screen">
        {/* Header */}
        <header className="border-b border-gray-800 px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <h1 className="text-lg font-bold tracking-tight">HRT Trading</h1>
            <span className="text-xs bg-blue-900 text-blue-300 px-2 py-0.5 rounded">
              LIVE
            </span>
          </div>
          <div className="text-xs text-gray-500 font-mono">
            {new Date().toISOString()}
          </div>
        </header>

        {/* Metrics Row */}
        <div className="dashboard-grid">
          <div style={{ gridColumn: 'span 3' }}>
            <MetricCard
              label="Daily P&L"
              value={formatCurrency(riskSummary?.daily_pnl ?? 0)}
              colorClass={pnlClass(riskSummary?.daily_pnl ?? 0)}
            />
          </div>
          <div style={{ gridColumn: 'span 3' }}>
            <MetricCard
              label="Gross Notional"
              value={formatCurrency(riskSummary?.gross_notional ?? 0)}
            />
          </div>
          <div style={{ gridColumn: 'span 3' }}>
            <MetricCard
              label="Drawdown"
              value={formatPct(riskSummary?.drawdown_pct ?? 0)}
              colorClass={
                (riskSummary?.drawdown_pct ?? 0) > 0.03
                  ? 'pnl-negative'
                  : 'pnl-neutral'
              }
            />
          </div>
          <div style={{ gridColumn: 'span 3' }}>
            <MetricCard
              label="Orders/min"
              value={String(riskSummary?.order_rate_1m ?? 0)}
            />
          </div>
        </div>

        {/* Main Content */}
        <div className="dashboard-grid">
          <PositionsTable positions={positions} />
          <KillSwitchPanel
            active={riskSummary?.kill_switch ?? false}
            entityId={riskSummary?.entity_id ?? 'default'}
          />
          <ReconciliationPanel stats={reconStats ?? null} />
        </div>
      </div>
    </>
  );
}
