import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  ArrowDown,
  ArrowRight,
  ArrowUp,
  CalendarDays,
  CheckCircle2,
  Clock3,
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

  const recommendations = useMemo(() => {
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
    confidence: Math.round((row.confidence_score ?? 0) * 100),
  }));
  const modelScores = payload.champion.models.slice(0, 6).map((row) => ({
    model: shortModel(row.model),
    score: row.champion_score,
  }));

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
              ? `${topRecommendation.predicted_avg_carbon_intensity_g_co2e_per_kwh.toFixed(2)}`
              : "-"
          }
          detail="gCO2e/kWh predicted"
        />
        <Metric
          icon={<Zap size={20} />}
          label="Carbon saving"
          value={
            topRecommendation
              ? `${topRecommendation.carbon_savings_vs_run_now_g_co2e_per_kwh.toFixed(2)}`
              : "-"
          }
          detail="gCO2e/kWh vs run now"
        />
        <Metric
          icon={<Gauge size={20} />}
          label="Confidence"
          value={topRecommendation ? titleCase(topRecommendation.confidence_level) : "-"}
          detail={
            topRecommendation
              ? `${Math.round(topRecommendation.confidence_score * 100)}% score`
              : "No confidence"
          }
        />
      </section>

      <section className="content-grid">
        <div className="panel recommendations-panel">
          <div className="panel-heading">
            <div>
              <h2>Production Recommendations</h2>
              <p>Top 5 clean start hours from the active production model.</p>
            </div>
          </div>
          <div className="recommendation-list">
            {recommendations.map((row) => (
              <RecommendationRow key={`${row.decision_group}-${row.recommendation_rank}`} row={row} />
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
                <span>{row.predicted_avg_carbon_intensity_g_co2e_per_kwh.toFixed(2)} gCO2e/kWh</span>
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

function RecommendationRow({ row }) {
  return (
    <article className="recommendation-row">
      <div className="rank-cell">#{row.recommendation_rank}</div>
      <div>
        <strong>{formatHour(row.timestamp_utc)}</strong>
        <span>{formatDateTime(row.timestamp_utc)} UTC</span>
      </div>
      <div>
        <span>Carbon</span>
        <strong>{row.predicted_avg_carbon_intensity_g_co2e_per_kwh.toFixed(2)}</strong>
      </div>
      <DirectionBadge value={row.predicted_price_direction_vs_previous_day} />
      <ConfidenceBadge level={row.confidence_level} score={row.confidence_score} />
    </article>
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
  return new Intl.DateTimeFormat("en-GB", {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "UTC",
  }).format(new Date(value));
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

function titleCase(value) {
  if (!value) return "";
  return value.charAt(0).toUpperCase() + value.slice(1);
}

const rootElement = document.getElementById("root");
window.__cleanHourRoot = window.__cleanHourRoot ?? createRoot(rootElement);
window.__cleanHourRoot.render(<App />);
