import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertTriangle,
  ArrowDown,
  ArrowRight,
  ArrowUp,
  CalendarDays,
  CheckCircle2,
  Clock3,
  Database,
  Gauge,
  Leaf,
  SlidersHorizontal,
  Zap,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import "./styles.css";

const DATA_URL = "/data/dashboard.json";

function App() {
  const [payload, setPayload] = useState(null);
  const [error, setError] = useState(null);
  const [selectedDate, setSelectedDate] = useState("");
  const [selectedScenario, setSelectedScenario] = useState("clean_first");

  useEffect(() => {
    fetch(DATA_URL)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Unable to load ${DATA_URL}`);
        }
        return response.json();
      })
      .then((data) => {
        setPayload(data);
        const dates = data.filters?.dates ?? [];
        const scenarios = data.filters?.scenarios ?? [];
        setSelectedDate(dates[dates.length - 1] ?? "");
        setSelectedScenario(scenarios.includes("clean_first") ? "clean_first" : scenarios[0] ?? "");
      })
      .catch((loadError) => setError(loadError.message));
  }, []);

  const baseRecommendations = useMemo(() => {
    if (!payload || !selectedDate) return [];
    return payload.recommendations
      .filter((row) => row.decision_group === selectedDate)
      .sort((left, right) => left.recommendation_rank - right.recommendation_rank);
  }, [payload, selectedDate]);

  const scenarioRecommendations = useMemo(() => {
    if (!payload || !selectedDate) return [];
    return payload.scenario_recommendations
      .filter(
        (row) =>
          row.decision_group === selectedDate && row.scenario === selectedScenario,
      )
      .sort((left, right) => left.recommendation_rank - right.recommendation_rank);
  }, [payload, selectedDate, selectedScenario]);
  const recommendations = scenarioRecommendations.length > 0 ? scenarioRecommendations : baseRecommendations;

  const championMetrics = useMemo(() => {
    if (!payload?.champion?.model) return null;
    return payload.champion.models.find((row) => row.model === payload.champion.model);
  }, [payload]);

  if (error) {
    return (
      <main className="status-screen">
        <h1>Clean-Hour Scheduling</h1>
        <p>{error}</p>
        <code>Run make forecast-recommendations and make dashboard-data.</code>
      </main>
    );
  }

  if (!payload) {
    return (
      <main className="status-screen">
        <Gauge className="spin" size={24} />
        <p>Loading clean-hour recommendations...</p>
      </main>
    );
  }

  const topRecommendation = recommendations[0];
  const carbonChart = recommendations.map((row) => ({
    hour: formatHour(row.timestamp_utc),
    carbon: row.predicted_avg_carbon_intensity_g_co2e_per_kwh,
    confidence: row.confidence_score == null ? null : Math.round(row.confidence_score * 100),
  }));
  const modelScores = payload.champion.models.slice(0, 6).map((row) => ({
    model: shortModel(row.model),
    score: row.champion_score,
  }));
  const healthSummary = payload.summary?.pipeline_health;
  const monitor = payload.forecast_monitoring ?? {};
  const monitorSummary = payload.summary?.forecast_monitoring ?? {};
  const staleFutureRecommendations = Boolean(payload.summary?.stale_future_recommendations);
  const staleFutureScenarios = Boolean(payload.summary?.stale_future_scenarios);
  const activeRecommendationCount = payload.summary?.active_future_scenario_count
    ?? payload.summary?.active_future_recommendation_count
    ?? 0;

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">France electricity workload scheduler</p>
          <h1>Clean-Hour Decision Dashboard</h1>
        </div>
        <div className="champion-pill">
          <CheckCircle2 size={18} />
          <span>Model</span>
          <strong>{payload.champion.display_model_name ?? "Production Model V1"}</strong>
        </div>
      </header>

      <section className="controls-band">
        <label>
          <CalendarDays size={16} />
          <span>Decision date</span>
          <select value={selectedDate} onChange={(event) => setSelectedDate(event.target.value)}>
            {payload.filters.dates.map((date) => (
              <option key={date} value={date}>
                {date}
              </option>
            ))}
          </select>
        </label>
        <label>
          <SlidersHorizontal size={16} />
          <span>Scenario</span>
          <select
            value={selectedScenario}
            onChange={(event) => setSelectedScenario(event.target.value)}
          >
            {payload.filters.scenarios.map((scenario) => (
              <option key={scenario} value={scenario}>
                {formatScenario(scenario)}
              </option>
            ))}
          </select>
        </label>
      </section>

      <section className={`health-band ${healthStatusClass(healthSummary?.status)}`}>
        <div className="health-title">
          {healthSummary?.status === "pass" ? <CheckCircle2 size={18} /> : <AlertTriangle size={18} />}
          <div>
            <span>Data Status</span>
            <strong>{formatHealthStatus(healthSummary?.status)}</strong>
          </div>
        </div>
        <HealthItem
          icon={<Database size={16} />}
          label="Latest data"
          value={formatDateTime(healthSummary?.latest_data_timestamp_utc)}
        />
        <HealthItem
          icon={<Clock3 size={16} />}
          label="Checked"
          value={formatDateTime(healthSummary?.generated_at_utc)}
        />
        <HealthItem
          icon={<Activity size={16} />}
          label="Issues"
          value={`${healthSummary?.critical_issue_count ?? 0} critical / ${healthSummary?.warning_count ?? 0} warnings`}
        />
        <HealthItem
          icon={<Zap size={16} />}
          label="Recommendations"
          value={`${activeRecommendationCount} future`}
        />
      </section>

      <section className={`monitor-band ${monitorStatusClass(monitorSummary)}`}>
        <div className="monitor-title">
          {monitorSummary.retraining_recommended || monitorSummary.stale ? <AlertTriangle size={18} /> : <CheckCircle2 size={18} />}
          <div>
            <span>Forecast Monitor</span>
            <strong>{formatMonitorStatus(monitorSummary)}</strong>
          </div>
        </div>
        <HealthItem
          icon={<Clock3 size={16} />}
          label="Checked"
          value={formatDateTime(monitorSummary.generated_at_utc)}
        />
        <HealthItem
          icon={<Database size={16} />}
          label="Latest actual"
          value={formatDateTime(monitorSummary.latest_actual_timestamp_utc)}
        />
        <HealthItem
          icon={<Activity size={16} />}
          label="Reasons"
          value={`${monitorSummary.reason_count ?? 0} triggers / ${monitorSummary.warning_count ?? 0} warnings`}
        />
        <div className="monitor-reasons">
          {(monitor.reasons ?? []).slice(0, 2).map((reason) => (
            <span key={reason}>{formatReason(reason)}</span>
          ))}
          {(!monitor.reasons || monitor.reasons.length === 0) && <span>No retraining trigger</span>}
        </div>
      </section>

      <section className="kpi-grid">
        <Metric
          icon={<Clock3 size={20} />}
          label="Best start"
          value={topRecommendation ? formatHour(topRecommendation.timestamp_utc) : "-"}
          detail={topRecommendation ? `${topRecommendation.duration_hours}h workload` : "No rows"}
        />
        <Metric
          icon={<Leaf size={20} />}
          label="Carbon intensity"
          value={
            topRecommendation
              ? formatFixed(topRecommendation.predicted_avg_carbon_intensity_g_co2e_per_kwh)
              : "-"
          }
          detail="gCO2e/kWh predicted"
        />
        <Metric
          icon={<Zap size={20} />}
          label="Carbon saving"
          value={
            topRecommendation
              ? formatFixed(topRecommendation.carbon_savings_vs_run_now_g_co2e_per_kwh)
              : "-"
          }
          detail="gCO2e/kWh vs run now"
        />
        <Metric
          icon={<Gauge size={20} />}
          label="Confidence"
          value={topRecommendation ? titleCase(topRecommendation.confidence_level) || "-" : "-"}
          detail={
            topRecommendation?.confidence_score != null
              ? `${Math.round(topRecommendation.confidence_score * 100)}% score`
              : "No confidence"
          }
        />
      </section>

      <section className="content-grid">
        <div className="panel recommendations-panel">
          <div className="panel-heading">
            <div>
              <h2>Clean-Hour Recommendations</h2>
              <p>Top 5 future workload start hours. Carbon intensity is predicted operational gCO2e per kWh.</p>
            </div>
          </div>
          <div className="recommendation-list">
            <div className="recommendation-header" aria-hidden="true">
              <span>Rank</span>
              <span>Start time</span>
              <span>Carbon intensity</span>
              <span>Price vs yesterday</span>
              <span>Confidence</span>
            </div>
            {staleFutureRecommendations && (
              <div className="empty-state">
                Future recommendation file is stale. Run <code>make operational-refresh</code>.
              </div>
            )}
            {staleFutureScenarios && (
              <div className="empty-state">
                Future scenario file is stale. Run <code>make operational-refresh</code>.
              </div>
            )}
            {!staleFutureRecommendations && !staleFutureScenarios && recommendations.length === 0 && (
              <div className="empty-state">
                No future recommendation rows are available for the selected date.
              </div>
            )}
            {recommendations.map((row) => (
              <RecommendationRow
                key={`${row.scenario ?? "base"}-${row.decision_group}-${row.recommendation_rank}`}
                row={row}
              />
            ))}
          </div>
        </div>

        <div className="panel">
          <div className="panel-heading">
            <div>
              <h2>Carbon and Confidence</h2>
              <p>Lower carbon intensity is better; confidence combines rank, margin, and model agreement.</p>
            </div>
          </div>
          <div className="chart-area">
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={carbonChart} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
                <CartesianGrid stroke="#e7e3d8" strokeDasharray="4 4" />
                <XAxis dataKey="hour" tickLine={false} axisLine={false} />
                <YAxis tickLine={false} axisLine={false} width={42} />
                <Tooltip />
                <Line
                  type="monotone"
                  dataKey="carbon"
                  stroke="#1f8a70"
                  strokeWidth={3}
                  dot={{ r: 4 }}
                  name="Carbon intensity"
                />
                <Line
                  type="monotone"
                  dataKey="confidence"
                  stroke="#4b5563"
                  strokeWidth={2}
                  dot={false}
                  name="Confidence %"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>

      <section className="content-grid lower-grid">
        <div className="panel">
          <div className="panel-heading">
            <div>
              <h2>Scenario Reranking</h2>
              <p>Compare alternate scheduling preferences without retraining.</p>
            </div>
          </div>
          <div className="scenario-table">
            {scenarioRecommendations.map((row) => (
              <div className="scenario-row" key={`${row.scenario}-${row.recommendation_rank}`}>
                <span className="rank">#{row.recommendation_rank}</span>
                <strong>{formatHour(row.timestamp_utc)}</strong>
                <span>{formatFixed(row.predicted_avg_carbon_intensity_g_co2e_per_kwh)} gCO2e/kWh</span>
                <DirectionBadge value={row.predicted_price_direction_vs_previous_day} />
              </div>
            ))}
          </div>
        </div>

        <div className="panel">
          <div className="panel-heading">
            <div>
              <h2>Model Quality</h2>
              <p>Lower weighted score wins under the carbon-first rule.</p>
            </div>
          </div>
          <div className="chart-area compact">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={modelScores} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
                <CartesianGrid stroke="#e7e3d8" strokeDasharray="4 4" />
                <XAxis dataKey="model" tickLine={false} axisLine={false} />
                <YAxis tickLine={false} axisLine={false} width={36} />
                <Tooltip />
                <Bar dataKey="score" fill="#1f8a70" radius={[4, 4, 0, 0]} name="Champion score" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          {championMetrics && (
            <div className="score-breakdown">
              <span>Carbon MAE {championMetrics.carbon_intensity_mae_g_co2e_per_kwh.toFixed(2)}</span>
              <span>Carbon regret {championMetrics.carbon_regret_g_co2e_per_kwh.toFixed(2)}</span>
              <span>Top-5 F1 {championMetrics.top_5_f1.toFixed(2)}</span>
            </div>
          )}
        </div>
      </section>
    </main>
  );
}

function Metric({ icon, label, value, detail }) {
  return (
    <div className="metric">
      <div className="metric-icon">{icon}</div>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </div>
  );
}

function HealthItem({ icon, label, value }) {
  return (
    <div className="health-item">
      {icon}
      <span>{label}</span>
      <strong>{value || "-"}</strong>
    </div>
  );
}

function RecommendationRow({ row }) {
  const confidenceAvailable = row.confidence_score != null && row.confidence_level;
  const priceRank = row.predicted_price_rank ?? row.recommendation_rank;
  const scenarioRank = row.predicted_scenario_rank ?? row.recommendation_rank;
  return (
    <details className="recommendation-row">
      <summary className="recommendation-summary">
        <span className="rank-cell">#{row.recommendation_rank}</span>
        <span className="time-cell">
          <strong>{formatHour(row.timestamp_utc)}</strong>
          <small>{formatDateTime(row.timestamp_utc)} UTC</small>
        </span>
        <span className="metric-cell">
          <strong>{formatFixed(row.predicted_avg_carbon_intensity_g_co2e_per_kwh)}</strong>
          <small>gCO2e/kWh</small>
        </span>
        <span className="metric-cell">
          <DirectionBadge value={row.predicted_price_direction_vs_previous_day} />
          <small>same hour previous day</small>
        </span>
        <span className="metric-cell">
          <ConfidenceBadge level={row.confidence_level} score={row.confidence_score} />
          <small>{confidenceAvailable ? "rank and margin score" : "scenario rerank"}</small>
        </span>
      </summary>
      <div className="recommendation-details">
        <DetailItem
          label="Predicted total emissions"
          value={`${formatNumber(row.predicted_total_emissions_kg_co2e)} kgCO2e`}
        />
        <DetailItem
          label="Carbon rank"
          value={`${row.predicted_carbon_rank ?? scenarioRank} of ${row.candidate_count ?? "-"} candidate hours`}
        />
        <DetailItem
          label="Price rank"
          value={`${priceRank} of ${row.candidate_count ?? "-"} candidate hours`}
        />
        {row.scenario && (
          <DetailItem
            label="Scenario weights"
            value={`${Math.round((row.scenario_carbon_weight ?? 0) * 100)}% carbon / ${Math.round((row.scenario_price_weight ?? 0) * 100)}% price`}
          />
        )}
        {row.predicted_scenario_score != null && (
          <DetailItem
            label="Scenario score"
            value={formatFixed(row.predicted_scenario_score)}
          />
        )}
        <DetailItem
          label="Carbon saving vs run now"
          value={`${formatFixed(row.carbon_savings_vs_run_now_g_co2e_per_kwh)} gCO2e/kWh`}
        />
        {row.cost_savings_vs_run_now_eur_mwh != null && (
          <DetailItem
            label="Cost saving vs run now"
            value={`${formatFixed(row.cost_savings_vs_run_now_eur_mwh)} EUR/MWh`}
          />
        )}
        {row.empirical_top_n_hit_rate != null && (
          <DetailItem
            label="Historical top-5 hit rate"
            value={`${Math.round(row.empirical_top_n_hit_rate * 100)}%`}
          />
        )}
        {row.expected_carbon_regret_g_co2e_per_kwh != null && (
          <DetailItem
            label="Expected carbon regret"
            value={`${formatFixed(row.expected_carbon_regret_g_co2e_per_kwh)} gCO2e/kWh`}
          />
        )}
        {row.heuristic_confidence_level && row.confidence_level && (
          <DetailItem
            label="Confidence calibration"
            value={`${titleCase(row.heuristic_confidence_level)} raw -> ${titleCase(row.confidence_level)} calibrated`}
          />
        )}
      </div>
    </details>
  );
}

function DetailItem({ label, value }) {
  return (
    <div className="detail-item">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function DirectionBadge({ value }) {
  const icon =
    value === "increase" ? <ArrowUp size={14} /> : value === "decrease" ? <ArrowDown size={14} /> : <ArrowRight size={14} />;
  return (
    <span className={`direction-badge ${value}`}>
      {icon}
      {titleCase(value)}
    </span>
  );
}

function ConfidenceBadge({ level, score }) {
  if (score == null || !level) {
    return (
      <span className="confidence-badge unavailable">
        <Activity size={14} />
        Scenario
      </span>
    );
  }
  return (
    <span className={`confidence-badge ${level}`}>
      <Activity size={14} />
      {titleCase(level)} {Math.round(score * 100)}%
    </span>
  );
}

function formatHour(value) {
  return new Intl.DateTimeFormat("en-GB", {
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "UTC",
  }).format(new Date(value));
}

function formatDateTime(value) {
  if (!value) return "-";
  return new Intl.DateTimeFormat("en-GB", {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "UTC",
  }).format(new Date(value));
}

function formatHealthStatus(value) {
  if (value === "pass") return "Healthy";
  if (value === "warn") return "Warnings";
  if (value === "fail") return "Action needed";
  return "Unknown";
}

function healthStatusClass(value) {
  if (value === "pass") return "pass";
  if (value === "warn") return "warn";
  if (value === "fail") return "fail";
  return "unknown";
}

function monitorStatusClass(summary) {
  if (summary?.stale) return "warn";
  if (summary?.retraining_recommended) return "fail";
  return healthStatusClass(summary?.status);
}

function formatMonitorStatus(summary) {
  if (summary?.stale) return "Monitor stale";
  if (summary?.retraining_recommended) return "Retraining recommended";
  return formatHealthStatus(summary?.status);
}

function shortModel(value) {
  return value
    .replace("hist_gradient_boosting", "HGB")
    .replace("random_forest", "RF")
    .replace("naive_lag_24h", "Naive")
    .replace("lightgbm", "LGBM")
    .replace("xgboost", "XGB")
    .replace("ridge", "Ridge");
}

function formatScenario(value) {
  return value.split("_").map(titleCase).join(" ");
}

function formatReason(value) {
  if (!value) return "";
  return value.split("_").map(titleCase).join(" ");
}

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return new Intl.NumberFormat("en-GB", {
    maximumFractionDigits: 2,
  }).format(Number(value));
}

function formatFixed(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(2);
}

function titleCase(value) {
  if (!value) return "";
  return value.charAt(0).toUpperCase() + value.slice(1);
}

const rootElement = document.getElementById("root");
window.__cleanHourRoot = window.__cleanHourRoot ?? createRoot(rootElement);
window.__cleanHourRoot.render(<App />);
