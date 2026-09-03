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
  const [selectedBasis, setSelectedBasis] = useState("scenario");

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
  const causalRecommendations = useMemo(() => {
    if (!payload || !selectedDate) return [];
    return (payload.causal_recommendations ?? [])
      .filter((row) => row.decision_group === selectedDate)
      .sort((left, right) => left.recommendation_rank - right.recommendation_rank);
  }, [payload, selectedDate]);
  const hasScenarioMode = (payload?.filters?.scenarios ?? []).length > 0;
  const hasCausalMode = (payload?.causal_recommendations ?? []).length > 0;
  const recommendations =
    selectedBasis === "causal"
      ? causalRecommendations
      : hasScenarioMode
        ? scenarioRecommendations
        : baseRecommendations;

  const championMetrics = useMemo(() => {
    if (!payload?.champion?.model) return null;
    return payload.champion.models.find((row) => row.model === payload.champion.model);
  }, [payload]);
  const isSampleData = payload?.data_state?.mode === "sample";

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
  const selectedScenarioChampion = payload.summary?.scenario_champions?.find(
    (row) => row.scenario === selectedScenario,
  );
  const marginalShift = payload.summary?.marginal_ranking_shift ?? {};
  const carbonChart = recommendations.map((row) => ({
    hour: formatHour(row.timestamp_utc),
    carbon: recommendationCarbonIntensity(row),
    confidence: row.confidence_score == null ? null : Math.round(row.confidence_score * 100),
  }));
  const modelScores = payload.champion.models.slice(0, 6).map((row) => ({
    model: shortModel(row.model),
    score: row.champion_score,
  }));
  const hasRecommendationData = recommendations.length > 0;

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
          <select
            value={selectedDate}
            onChange={(event) => setSelectedDate(event.target.value)}
            disabled={(payload.filters?.dates ?? []).length === 0}
          >
            {(payload.filters?.dates ?? []).length === 0 && (
              <option value="">No dates available</option>
            )}
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
            disabled={(payload.filters?.scenarios ?? []).length === 0}
          >
            {payload.filters.scenarios.map((scenario) => (
              <option key={scenario} value={scenario}>
                {formatScenario(scenario)}
              </option>
            ))}
          </select>
        </label>
        <label>
          <Activity size={16} />
          <span>Basis</span>
          <select
            value={selectedBasis}
            onChange={(event) => setSelectedBasis(event.target.value)}
          >
            <option value="scenario">Scenario</option>
            <option value="causal" disabled={!hasCausalMode}>
              Causal-adjusted MVP
            </option>
          </select>
        </label>
      </section>

      {isSampleData && (
        <section className="deployment-state">
          <AlertTriangle size={20} />
          <div>
            <strong>Dashboard deployed without live recommendation data</strong>
            <p>{payload.data_state.message}</p>
          </div>
        </section>
      )}

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
              ? formatFixed(recommendationCarbonIntensity(topRecommendation))
              : "-"
          }
          detail={`${recommendationCarbonLabel(topRecommendation)} gCO2e/kWh`}
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
        <Metric
          icon={<AlertTriangle size={20} />}
          label="Risk status"
          value={topRecommendation ? formatRecommendationStatus(topRecommendation.recommendation_status) : "-"}
          detail={
            topRecommendation?.decision_uncertainty_score != null
              ? `${Math.round(topRecommendation.decision_uncertainty_score * 100)}% uncertainty`
              : "No uncertainty score"
          }
        />
      </section>

      <section className="content-grid">
        <div className="panel recommendations-panel">
          <div className="panel-heading">
            <div>
              <h2>Clean-Hour Recommendations</h2>
              <p>{recommendationSubtitle(selectedBasis, selectedScenario)}</p>
            </div>
          </div>
          <div className="recommendation-list">
            <div className="recommendation-header" aria-hidden="true">
              <span>Rank</span>
              <span>Start time</span>
              <span>Carbon intensity</span>
              <span>Price vs yesterday</span>
              <span>Confidence</span>
              <span>Risk</span>
            </div>
            {recommendations.length === 0 && (
              <div className="empty-state">
                {isSampleData
                  ? "Live recommendation data has not been published for this deployment yet."
                  : "No future recommendation rows are available for the selected date."}
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
              <LineChart data={hasRecommendationData ? carbonChart : []} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
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
            {!hasRecommendationData && (
              <div className="chart-empty">No recommendation data to chart yet.</div>
            )}
          </div>
        </div>
      </section>

      {hasCausalMode && (
        <section className="causal-band">
          <div>
            <span>Basis</span>
            <strong>{formatCausalMethod(marginalShift.method)}</strong>
          </div>
          <div>
            <span>Top-1 changed</span>
            <strong>{formatPercent(marginalShift.top_1_change_share)}</strong>
          </div>
          <div>
            <span>Top-5 overlap</span>
            <strong>{formatPercent(marginalShift.mean_top_5_overlap_share)}</strong>
          </div>
          <div>
            <span>Avg rank shift</span>
            <strong>{formatFixed(marginalShift.mean_absolute_rank_shift)}</strong>
          </div>
          <div>
            <span>Proxy coverage</span>
            <strong>{formatPercent(marginalShift.mean_causal_adjustment_coverage)}</strong>
          </div>
          <div className={`quality-chip ${marginalShift.quality_status ?? "unknown"}`}>
            <span>Quality guard</span>
            <strong>{titleCase(marginalShift.quality_status ?? "unknown")}</strong>
          </div>
        </section>
      )}

      <section className="content-grid lower-grid">
        <div className="panel">
          <div className="panel-heading">
            <div>
              <h2>Scenario Reranking</h2>
              <p>{formatScenario(selectedScenario)} reranks the same candidate hours using scenario-specific carbon and price weights.</p>
            </div>
          </div>
          <div className="scenario-table">
            {selectedScenarioChampion && (
              <div className="scenario-row scenario-champion">
                <span className="rank">Best</span>
                <strong>{shortModel(selectedScenarioChampion.model)}</strong>
                <span>{formatFixed(selectedScenarioChampion.mean_scenario_regret)} regret</span>
                <span>{formatFixed(selectedScenarioChampion.top_5_f1)} top-5 F1</span>
              </div>
            )}
            {scenarioRecommendations.length === 0 && (
              <div className="empty-state">
                No scenario recommendations are available yet.
              </div>
            )}
            {scenarioRecommendations.map((row) => (
              <div className="scenario-row" key={`${row.scenario}-${row.recommendation_rank}`}>
                <span className="rank">#{row.recommendation_rank}</span>
                <strong>{formatHour(row.timestamp_utc)}</strong>
                <span>{formatFixed(row.predicted_avg_carbon_intensity_g_co2e_per_kwh)} gCO2e/kWh</span>
                {row.recommendation_status === "no_low_risk_recommendation_available" ? (
                  <RiskBadge status={row.recommendation_status} />
                ) : (
                  <DirectionBadge value={row.predicted_price_direction_vs_previous_day} />
                )}
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
            {modelScores.length === 0 && (
              <div className="chart-empty">No model quality metrics have been published yet.</div>
            )}
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
  const confidenceAvailable = row.confidence_score != null && row.confidence_level;
  const priceRank = row.predicted_price_rank ?? row.recommendation_rank;
  const scenarioRank = row.predicted_scenario_rank ?? row.recommendation_rank;
  const carbonIntensity = recommendationCarbonIntensity(row);
  const carbonLabel = recommendationCarbonLabel(row);
  return (
    <details className="recommendation-row">
      <summary className="recommendation-summary">
        <span className="rank-cell">#{row.recommendation_rank}</span>
        <span className="time-cell">
          <strong>{formatHour(row.timestamp_utc)}</strong>
          <small>{formatDateTime(row.timestamp_utc)} UTC</small>
          {row.scenario && (
            <small>{formatScenario(row.scenario)} score {formatFixed(row.predicted_scenario_score)}</small>
          )}
        </span>
        <span className="metric-cell">
          <strong>{formatFixed(carbonIntensity)}</strong>
          <small>{carbonLabel}</small>
        </span>
        <span className="metric-cell">
          <DirectionBadge value={row.predicted_price_direction_vs_previous_day} />
          <small>same hour previous day</small>
        </span>
        <span className="metric-cell">
          <ConfidenceBadge level={row.confidence_level} score={row.confidence_score} />
          <small>{confidenceAvailable ? "rank and margin score" : "scenario rerank"}</small>
        </span>
        <span className="metric-cell">
          <RiskBadge status={row.recommendation_status} />
          <small>
            {row.decision_uncertainty_score != null
              ? `${Math.round(row.decision_uncertainty_score * 100)}% uncertainty`
              : "uncertainty unavailable"}
          </small>
        </span>
      </summary>
      <div className="recommendation-details">
        <DetailItem
          label="Recommendation status"
          value={formatRecommendationStatus(row.recommendation_status)}
        />
        {row.predicted_price_interval_half_width_eur_mwh != null && (
          <DetailItem
            label="Price interval half-width"
            value={`${formatFixed(row.predicted_price_interval_half_width_eur_mwh)} EUR/MWh`}
          />
        )}
        {row.predicted_carbon_interval_half_width_g_co2e_per_kwh != null && (
          <DetailItem
            label="Carbon interval half-width"
            value={`${formatFixed(row.predicted_carbon_interval_half_width_g_co2e_per_kwh)} gCO2e/kWh`}
          />
        )}
        <DetailItem
          label="Predicted total emissions"
          value={`${formatNumber(row.predicted_total_emissions_kg_co2e)} kgCO2e`}
        />
        {row.predicted_avg_carbon_intensity_g_co2e_per_kwh != null && (
          <DetailItem
            label="Average carbon intensity"
            value={`${formatFixed(row.predicted_avg_carbon_intensity_g_co2e_per_kwh)} gCO2e/kWh`}
          />
        )}
        {row.causal_carbon_source && (
          <DetailItem
            label="Causal-adjusted source"
            value={formatCausalSource(row.causal_carbon_source)}
          />
        )}
        {row.causal_adjusted_rank_shift != null && (
          <DetailItem
            label="Average-vs-causal rank shift"
            value={formatSigned(row.causal_adjusted_rank_shift)}
          />
        )}
        {row.predicted_marginal_proxy_confidence && (
          <DetailItem
            label="Marginal proxy confidence"
            value={titleCase(row.predicted_marginal_proxy_confidence)}
          />
        )}
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

function RiskBadge({ status }) {
  const normalizedStatus = status ?? "recommended";
  const isNoLowRisk = normalizedStatus === "no_low_risk_recommendation_available";
  return (
    <span className={`risk-badge ${isNoLowRisk ? "blocked" : "ok"}`}>
      {isNoLowRisk ? <AlertTriangle size={14} /> : <CheckCircle2 size={14} />}
      {formatRecommendationStatus(normalizedStatus)}
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
  if (!value) return "-";
  return value.split("_").map(titleCase).join(" ");
}

function formatRecommendationStatus(value) {
  if (!value) return "Unknown";
  if (value === "no_low_risk_recommendation_available") return "No low-risk hour";
  if (value === "recommended") return "Recommended";
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

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${Math.round(Number(value) * 100)}%`;
}

function formatSigned(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  const numeric = Number(value);
  return numeric > 0 ? `+${formatFixed(numeric)}` : formatFixed(numeric);
}

function formatCausalSource(value) {
  if (value === "marginal_emissions_proxy") return "Marginal emissions proxy";
  if (value === "average_carbon_fallback") return "Average carbon fallback";
  return formatRecommendationStatus(value);
}

function formatCausalMethod(value) {
  if (value === "marginal_proxy_mvp") return "Marginal proxy MVP";
  return value ? formatRecommendationStatus(value) : "-";
}

function recommendationSubtitle(selectedBasis, selectedScenario) {
  if (selectedBasis === "causal") {
    return "Top 5 future workload start hours using the marginal-emissions proxy MVP.";
  }
  return `Top 5 future workload start hours for ${formatScenario(selectedScenario)}. Scenario rank and score update with the selector.`;
}

function recommendationCarbonIntensity(row) {
  return (
    row?.predicted_marginal_carbon_intensity_g_co2e_per_kwh
    ?? row?.predicted_avg_carbon_intensity_g_co2e_per_kwh
  );
}

function recommendationCarbonLabel(row) {
  return row?.predicted_marginal_carbon_intensity_g_co2e_per_kwh == null
    ? "Average predicted"
    : "Marginal proxy";
}

function titleCase(value) {
  if (!value) return "";
  return value.charAt(0).toUpperCase() + value.slice(1);
}

const rootElement = document.getElementById("root");
window.__cleanHourRoot = window.__cleanHourRoot ?? createRoot(rootElement);
window.__cleanHourRoot.render(<App />);
