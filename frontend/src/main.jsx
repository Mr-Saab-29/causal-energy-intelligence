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
  ClipboardCheck,
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
import BenchmarkView from "./BenchmarkView.jsx";

const DATA_URL = "/data/dashboard.json";

function App() {
  const [payload, setPayload] = useState(null);
  const [error, setError] = useState(null);
  const [selectedDate, setSelectedDate] = useState("");
  const [selectedOutcomeDate, setSelectedOutcomeDate] = useState("");
  const [selectedScenario, setSelectedScenario] = useState("emissions_reduction");
  const [selectedBasis, setSelectedBasis] = useState("scenario");
  const [selectedView, setSelectedView] = useState("live");

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
        const outcomeDates = data.filters?.outcome_dates ?? [];
        const scenarios = data.filters?.scenarios ?? [];
        setSelectedDate(dates[dates.length - 1] ?? "");
        setSelectedOutcomeDate(outcomeDates[0] ?? "");
        setSelectedScenario(
          scenarios.includes("emissions_reduction")
            ? "emissions_reduction"
            : scenarios.includes("clean_first")
              ? "clean_first"
              : scenarios[0] ?? "",
        );
      })
      .catch((loadError) => setError(loadError.message));
  }, []);

  const baseRecommendations = useMemo(() => {
    if (!payload || !selectedDate) return [];
    return (payload.recommendations ?? [])
      .filter((row) => row.decision_group === selectedDate)
      .sort((left, right) => left.recommendation_rank - right.recommendation_rank);
  }, [payload, selectedDate]);

  const scenarioRecommendations = useMemo(() => {
    if (!payload || !selectedDate) return [];
    return (payload.scenario_recommendations ?? [])
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

  const causalScenarioRecommendations = useMemo(() => {
    if (!payload || !selectedDate) return [];
    return (payload.causal_scenario_recommendations ?? [])
      .filter(
        (row) =>
          row.decision_group === selectedDate && row.scenario === selectedScenario,
      )
      .sort((left, right) => left.recommendation_rank - right.recommendation_rank);
  }, [payload, selectedDate, selectedScenario]);

  const outcomeRows = useMemo(() => {
    if (!payload || !selectedOutcomeDate) return [];
    return (payload.recommendation_outcomes ?? [])
      .filter((row) => row.decision_group === selectedOutcomeDate)
      .sort((left, right) => {
        const leftGenerated = left.forecast_generated_at_utc ?? "";
        const rightGenerated = right.forecast_generated_at_utc ?? "";
        if (leftGenerated !== rightGenerated) return rightGenerated.localeCompare(leftGenerated);
        return left.recommendation_rank - right.recommendation_rank;
      });
  }, [payload, selectedOutcomeDate]);

  const hasScenarioMode = (payload?.filters?.scenarios ?? []).length > 0;
  const hasCausalMode =
    (payload?.causal_recommendations ?? []).length > 0 ||
    (payload?.causal_scenario_recommendations ?? []).length > 0;
  const recommendations =
    selectedBasis === "causal"
      ? causalScenarioRecommendations.length > 0
        ? causalScenarioRecommendations
        : causalRecommendations
      : hasScenarioMode
        ? scenarioRecommendations
        : baseRecommendations;
  const scenarioPanelRecommendations =
    selectedBasis === "causal" && causalScenarioRecommendations.length > 0
      ? causalScenarioRecommendations
      : scenarioRecommendations;
  const championMetrics = useMemo(() => {
    if (!payload?.champion?.model) return null;
    return (payload.champion?.models ?? []).find((row) => row.model === payload.champion.model);
  }, [payload]);
  const isSampleData = payload?.data_state?.mode === "sample";
  const outcomeSummary = payload?.summary?.recommendation_outcome_audit ?? {};

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
  const policyBacktest = payload.summary?.policy_backtest ?? {};
  const trustSummary = buildTrustSummary(payload);
  const carbonChart = recommendations.map((row) => ({
    hour: formatHour(row.timestamp_utc),
    carbon: recommendationCarbonIntensity(row),
    confidence: row.confidence_score == null ? null : Math.round(row.confidence_score * 100),
  }));
  const modelScores = (payload.champion?.models ?? []).slice(0, 6).map((row) => ({
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
          <strong>{payload.champion?.display_model_name ?? "Production Model V1"}</strong>
        </div>
      </header>

      <section className="view-tabs" aria-label="Dashboard view">
        <button
          className={selectedView === "live" ? "active" : ""}
          type="button"
          onClick={() => setSelectedView("live")}
        >
          <Gauge size={16} />
          Live recommendations
        </button>
        <button
          className={selectedView === "audit" ? "active" : ""}
          type="button"
          onClick={() => setSelectedView("audit")}
        >
          <ClipboardCheck size={16} />
          Previous-day audit
        </button>
        <button
          className={selectedView === "benchmark" ? "active" : ""}
          type="button"
          onClick={() => setSelectedView("benchmark")}
        >
          <Activity size={16} />
          Benchmarks
        </button>
      </section>

      {selectedView !== "benchmark" && <TrustFreshnessBanner payload={payload} trustSummary={trustSummary} />}

      {selectedView === "benchmark" ? <BenchmarkView /> : selectedView === "audit" ? (
        <OutcomeAuditView
          payload={payload}
          outcomeRows={outcomeRows}
          outcomeSummary={outcomeSummary}
          selectedOutcomeDate={selectedOutcomeDate}
          setSelectedOutcomeDate={setSelectedOutcomeDate}
        />
      ) : (
        <>
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
                {(payload.filters?.dates ?? []).map((date) => (
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
                {(payload.filters?.scenarios ?? []).map((scenario) => (
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
                  ? formatFixed(recommendationCarbonSaving(topRecommendation))
                  : "-"
              }
              detail={recommendationReferenceLabel(topRecommendation)}
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
              value={
                topRecommendation
                  ? formatRecommendationStatus(topRecommendation.recommendation_status)
                  : "-"
              }
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
                    key={`${row.scenario ?? selectedBasis}-${row.decision_group}-${row.timestamp_utc}-${row.recommendation_rank}`}
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
                    <Line type="monotone" dataKey="carbon" stroke="#1f8a70" strokeWidth={3} dot={{ r: 4 }} name="Carbon intensity" />
                    <Line type="monotone" dataKey="confidence" stroke="#4b5563" strokeWidth={2} dot={false} name="Confidence %" />
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

          <section className="content-grid analytics-grid">
            <BacktestingSummaryPanel
              policyBacktest={policyBacktest}
              selectedScenario={selectedScenario}
              championModel={payload.champion?.model}
            />
            <CausalAverageComparisonPanel marginalShift={marginalShift} />
          </section>

          <section className="content-grid lower-grid">
            <div className="panel">
              <div className="panel-heading">
                <div>
                  <h2>Scenario Reranking</h2>
                  <p>{scenarioPanelSubtitle(selectedBasis, selectedScenario)}</p>
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
                {scenarioPanelRecommendations.length === 0 && (
                  <div className="empty-state">No scenario recommendations are available yet.</div>
                )}
                {scenarioPanelRecommendations.map((row) => (
                  <div
                    className="scenario-row"
                    key={`${row.scenario}-${row.timestamp_utc}-${row.recommendation_rank}`}
                  >
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
                  <span>Carbon MAE {formatFixed(championMetrics.carbon_intensity_mae_g_co2e_per_kwh)}</span>
                  <span>Carbon regret {formatFixed(championMetrics.carbon_regret_g_co2e_per_kwh)}</span>
                  <span>Top-5 F1 {formatFixed(championMetrics.top_5_f1)}</span>
                </div>
              )}
            </div>
          </section>
        </>
      )}
    </main>
  );
}

function TrustFreshnessBanner({ payload, trustSummary }) {
  const forecast = payload.summary?.forecast_monitoring ?? {};
  const quantileQuality = forecast.quantile_quality ?? {};
  const readiness = payload.summary?.operational_audit_readiness ?? {};
  const reference = payload.current_reference ?? {};
  return (
    <section className={`trust-banner ${trustSummary.level}`}>
      <div className="trust-status">
        {trustSummary.level === "ready" ? <CheckCircle2 size={18} /> : <AlertTriangle size={18} />}
        <div>
          <span>Trust status</span>
          <strong>{trustSummary.label}</strong>
        </div>
      </div>
      <div className="trust-grid">
        <TrustItem label="Latest actual" value={formatDateTime(forecast.latest_actual_timestamp_utc)} />
        <TrustItem label="Audit coverage" value={`${readiness.settled_current_month_recommendation_rows ?? 0} settled`} />
        <TrustItem label="Built" value={formatDateTime(payload.generated_at_utc)} />
        <TrustItem label="Reference" value={formatDateTime(reference.reference_hour_utc)} />
        {quantileQuality.price_coverage_80 != null && (
          <TrustItem label="Price P80 coverage" value={formatPercent(quantileQuality.price_coverage_80)} />
        )}
        {quantileQuality.carbon_coverage_80 != null && (
          <TrustItem label="Carbon P80 coverage" value={formatPercent(quantileQuality.carbon_coverage_80)} />
        )}
      </div>
      {trustSummary.message && <p>{trustSummary.message}</p>}
    </section>
  );
}

function TrustItem({ label, value }) {
  return (
    <div className="trust-item">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function BacktestingSummaryPanel({ policyBacktest, selectedScenario, championModel }) {
  const baseRows = policyBacktest.base_policy ?? [];
  const scenarioRows = policyBacktest.scenario_policy ?? [];
  const championBacktest =
    baseRows.find((row) => row.model === championModel) ?? baseRows[0] ?? null;
  const scenarioBacktest =
    scenarioRows.find(
      (row) => row.scenario === selectedScenario && row.model === championModel,
    ) ??
    scenarioRows.find((row) => row.scenario === selectedScenario) ??
    null;
  const rows = [
    { label: "Champion", row: championBacktest },
    { label: formatScenario(selectedScenario), row: scenarioBacktest },
  ].filter((item) => item.row);

  return (
    <div className="panel">
      <div className="panel-heading">
        <div>
          <h2>Backtesting Summary</h2>
          <p>Historical rank-1 outcomes from emitted recommendation rows.</p>
        </div>
      </div>
      {rows.length === 0 ? (
        <div className="empty-state">No recommendation backtest metrics have been published yet.</div>
      ) : (
        <>
          <div className="summary-table">
            <div className="summary-header" aria-hidden="true">
              <span>Policy</span>
              <span>Top-1 hit</span>
              <span>Top-5 hit</span>
              <span>Carbon regret</span>
              <span>Risk blocks</span>
            </div>
            {rows.map(({ label, row }) => (
              <div className="summary-row" key={`${label}-${row.model ?? "model"}`}>
                <strong>{label}</strong>
                <span>{formatPercent(row.top_1_hit_rate)}</span>
                <span>{formatPercent(row.top_5_hit_rate)}</span>
                <span>{formatFixed(row.mean_carbon_regret_g_co2e_per_kwh)}</span>
                <span>{row.no_low_risk_groups ?? 0}</span>
              </div>
            ))}
          </div>
          <div className="score-breakdown">
            <span>Days {championBacktest?.decision_groups ?? "-"}</span>
            <span>Mean confidence {formatPercent(championBacktest?.mean_confidence_score)}</span>
            <span>Combined regret {formatFixed(championBacktest?.mean_combined_regret)}</span>
          </div>
        </>
      )}
    </div>
  );
}

function CausalAverageComparisonPanel({ marginalShift }) {
  const hasMetrics = Object.keys(marginalShift ?? {}).length > 0 && marginalShift.method;
  return (
    <div className="panel">
      <div className="panel-heading">
        <div>
          <h2>Causal vs Average</h2>
          <p>Ranking movement after replacing average carbon with the marginal proxy.</p>
        </div>
      </div>
      {!hasMetrics ? (
        <div className="empty-state">No causal comparison metrics have been published yet.</div>
      ) : (
        <>
          <div className="comparison-grid">
            <div>
              <span>Average-carbon ranking</span>
              <strong>Baseline</strong>
              <small>Uses predicted average carbon intensity.</small>
            </div>
            <div>
              <span>Causal-adjusted MVP</span>
              <strong>{formatCausalMethod(marginalShift.method)}</strong>
              <small>Uses marginal-emissions proxy where coverage is available.</small>
            </div>
          </div>
          <div className="summary-table causal-comparison-table">
            <div className="summary-header" aria-hidden="true">
              <span>Metric</span>
              <span>Value</span>
              <span>Read</span>
            </div>
            <div className="summary-row">
              <strong>Top-1 changed</strong>
              <span>{formatPercent(marginalShift.top_1_change_share)}</span>
              <span>{comparisonRead(marginalShift.top_1_change_share, "changed")}</span>
            </div>
            <div className="summary-row">
              <strong>Top-5 overlap</strong>
              <span>{formatPercent(marginalShift.mean_top_5_overlap_share)}</span>
              <span>{comparisonRead(marginalShift.mean_top_5_overlap_share, "overlap")}</span>
            </div>
            <div className="summary-row">
              <strong>Avg rank shift</strong>
              <span>{formatFixed(marginalShift.mean_absolute_rank_shift)}</span>
              <span>{comparisonRead(marginalShift.mean_absolute_rank_shift, "shift")}</span>
            </div>
            <div className="summary-row">
              <strong>Proxy coverage</strong>
              <span>{formatPercent(marginalShift.mean_causal_adjustment_coverage)}</span>
              <span>{comparisonRead(marginalShift.mean_causal_adjustment_coverage, "coverage")}</span>
            </div>
          </div>
          <div className="score-breakdown">
            <span>Top-1 regret delta {formatSigned(marginalShift.mean_top_1_regret_delta)}</span>
            <span>Quality guard {titleCase(marginalShift.quality_status ?? "unknown")}</span>
          </div>
        </>
      )}
    </div>
  );
}

function OutcomeAuditView({
  payload,
  outcomeRows,
  outcomeSummary,
  selectedOutcomeDate,
  setSelectedOutcomeDate,
}) {
  const outcomeDates = payload.filters?.outcome_dates ?? [];
  const latestGeneration = outcomeRows[0]?.forecast_generated_at_utc;
  const latestRows = latestGeneration
    ? outcomeRows.filter((row) => row.forecast_generated_at_utc === latestGeneration)
    : outcomeRows;
  const dayTopRows = latestRows.filter((row) => row.recommendation_rank === 1);
  const chartRows = latestRows.map((row) => ({
    hour: formatHour(row.timestamp_utc),
    predicted: row.predicted_avg_carbon_intensity_g_co2e_per_kwh,
    actual: row.actual_carbon_intensity_g_co2e_per_kwh_observed,
  }));
  const topRecommendationOutcome = dayTopRows[0];
  const worstCarbonMiss = latestRows.reduce((current, row) => {
    const rowRegret = Number(row.carbon_regret_g_co2e_per_kwh_observed);
    const currentRegret = Number(current?.carbon_regret_g_co2e_per_kwh_observed);
    if (!Number.isFinite(rowRegret)) return current;
    if (!current || !Number.isFinite(currentRegret) || rowRegret > currentRegret) return row;
    return current;
  }, null);
  const top1HitRate = meanBoolean(dayTopRows.map((row) => row.is_actual_best_observed));
  const top5HitRate = meanBoolean(
    dayTopRows.map((row) => Number(row.actual_decision_rank_observed) <= 5),
  );

  return (
    <>
      <section className="controls-band">
        <label>
          <CalendarDays size={16} />
          <span>Outcome date</span>
          <select
            value={selectedOutcomeDate}
            onChange={(event) => setSelectedOutcomeDate(event.target.value)}
            disabled={outcomeDates.length === 0}
          >
            {outcomeDates.length === 0 && <option value="">No settled dates</option>}
            {outcomeDates.map((date) => (
              <option key={date} value={date}>
                {date}
              </option>
            ))}
          </select>
        </label>
      </section>

      {!outcomeSummary.available && (
        <section className="deployment-state">
          <AlertTriangle size={20} />
          <div>
            <strong>No settled recommendation outcomes yet</strong>
            <p>{outcomeSummary.reason ?? "Run recommendation-outcome-audit after actual data arrives."}</p>
          </div>
        </section>
      )}

      <section className="audit-insights">
        <div>
          <span>Latest snapshot</span>
          <strong>{formatDateTime(latestGeneration)}</strong>
        </div>
        <div>
          <span>Top recommendation</span>
          <strong>{auditVerdict(topRecommendationOutcome).label}</strong>
        </div>
        <div>
          <span>Actual rank</span>
          <strong>{formatFixed(topRecommendationOutcome?.actual_decision_rank_observed)}</strong>
        </div>
        <div>
          <span>Worst carbon regret</span>
          <strong>{formatFixed(worstCarbonMiss?.carbon_regret_g_co2e_per_kwh_observed)} gCO2e/kWh</strong>
        </div>
      </section>

      <section className="kpi-grid audit-kpis">
        <Metric
          icon={<CheckCircle2 size={20} />}
          label="Top-1 hit rate"
          value={formatPercent(top1HitRate ?? outcomeSummary.top_1_hit_rate)}
          detail="recommended first hour was actual best"
        />
        <Metric
          icon={<ClipboardCheck size={20} />}
          label="Top-5 hit rate"
          value={formatPercent(top5HitRate ?? outcomeSummary.top_5_hit_rate)}
          detail="actual best inside recommendation set"
        />
        <Metric
          icon={<Gauge size={20} />}
          label="Actual rank"
          value={formatFixed(mean(dayTopRows.map((row) => row.actual_decision_rank_observed)))}
          detail="mean actual rank of top recommendation"
        />
        <Metric
          icon={<Leaf size={20} />}
          label="Carbon regret"
          value={formatFixed(mean(dayTopRows.map((row) => row.carbon_regret_g_co2e_per_kwh_observed)))}
          detail="gCO2e/kWh above actual best"
        />
        <Metric
          icon={<Zap size={20} />}
          label="Price MAE"
          value={formatFixed(mean(latestRows.map((row) => Math.abs(Number(row.price_error)))))}
          detail="EUR/MWh for settled recommendations"
        />
      </section>

      <section className="content-grid">
        <div className="panel audit-panel">
          <div className="panel-heading">
            <div>
              <h2>Settled Recommendation Outcomes</h2>
              <p>Predictions are compared with actual values after the recommended hours have landed.</p>
            </div>
          </div>
          <div className="audit-table">
            <div className="audit-header" aria-hidden="true">
              <span>Rec</span>
              <span>Verdict</span>
              <span>Start</span>
              <span>Actual rank</span>
              <span>Price pred/actual</span>
              <span>Carbon pred/actual</span>
              <span>Load pred/actual</span>
            </div>
            {latestRows.length === 0 && (
              <div className="empty-state">No settled outcomes are available for this date.</div>
            )}
            {latestRows.map((row) => (
              <div
                className="audit-row"
                key={`${row.forecast_generated_at_utc}-${row.decision_group}-${row.recommendation_rank}`}
              >
                <span className="rank-cell">#{row.recommendation_rank}</span>
                <OutcomeVerdict row={row} />
                <span>
                  <strong>{formatHour(row.timestamp_utc)}</strong>
                  <small>{formatDateTime(row.timestamp_utc)} UTC</small>
                </span>
                <span>
                  <strong>{formatFixed(row.actual_decision_rank_observed)}</strong>
                  <small>{row.is_actual_best_observed ? "actual best" : "not best"}</small>
                </span>
                <span>
                  <strong>{formatFixed(row.predicted_avg_price_eur_mwh)} / {formatFixed(row.actual_price_eur_mwh_observed)}</strong>
                  <small>{formatSigned(row.price_error)} EUR/MWh error</small>
                </span>
                <span>
                  <strong>{formatFixed(row.predicted_avg_carbon_intensity_g_co2e_per_kwh)} / {formatFixed(row.actual_carbon_intensity_g_co2e_per_kwh_observed)}</strong>
                  <small>{formatSigned(row.carbon_intensity_error)} gCO2e/kWh error</small>
                </span>
                <span>
                  <strong>{formatFixed(row.predicted_consumption_mwh)} / {formatFixed(row.actual_consumption_mwh_observed)}</strong>
                  <small>{formatSigned(row.consumption_error)} MWh consumption</small>
                </span>
              </div>
            ))}
          </div>
        </div>

        <div className="panel">
          <div className="panel-heading">
            <div>
              <h2>Carbon Forecast Outcome</h2>
              <p>Predicted vs actual carbon intensity for the latest settled recommendation snapshot.</p>
            </div>
          </div>
          <div className="chart-area">
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={chartRows} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
                <CartesianGrid stroke="#e7e3d8" strokeDasharray="4 4" />
                <XAxis dataKey="hour" tickLine={false} axisLine={false} />
                <YAxis tickLine={false} axisLine={false} width={42} />
                <Tooltip />
                <Line type="monotone" dataKey="predicted" stroke="#4b5563" strokeWidth={2} name="Predicted carbon" />
                <Line type="monotone" dataKey="actual" stroke="#1f8a70" strokeWidth={3} name="Actual carbon" />
              </LineChart>
            </ResponsiveContainer>
            {chartRows.length === 0 && (
              <div className="chart-empty">No settled carbon outcome data yet.</div>
            )}
          </div>
          <div className="score-breakdown">
            <span>Carbon MAE {formatFixed(outcomeSummary.carbon_intensity_mae_g_co2e_per_kwh)}</span>
            <span>Consumption MAE {formatFixed(outcomeSummary.consumption_mae_mwh)}</span>
            <span>Production MAE {formatFixed(outcomeSummary.total_production_mae_mwh)}</span>
          </div>
        </div>
      </section>
    </>
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
  const explanation = recommendationExplanation(row);
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
        <span className="explanation-cell">
          <strong>Why this hour</strong>
          <small>{explanation}</small>
        </span>
      </summary>
      <div className="recommendation-details">
        <DetailItem label="Recommendation status" value={formatRecommendationStatus(row.recommendation_status)} />
        {row.predicted_avg_price_q10_eur_mwh != null && row.predicted_avg_price_q90_eur_mwh != null ? (
          <DetailItem label="Price 80% interval" value={`${formatFixed(row.predicted_avg_price_q10_eur_mwh)} to ${formatFixed(row.predicted_avg_price_q90_eur_mwh)} EUR/MWh`} />
        ) : row.predicted_price_interval_half_width_eur_mwh != null ? (
          <DetailItem label="Price interval half-width" value={`${formatFixed(row.predicted_price_interval_half_width_eur_mwh)} EUR/MWh`} />
        ) : null}
        {row.predicted_avg_carbon_intensity_q10_g_co2e_per_kwh != null && row.predicted_avg_carbon_intensity_q90_g_co2e_per_kwh != null ? (
          <DetailItem label="Carbon 80% interval" value={`${formatFixed(row.predicted_avg_carbon_intensity_q10_g_co2e_per_kwh)} to ${formatFixed(row.predicted_avg_carbon_intensity_q90_g_co2e_per_kwh)} gCO2e/kWh`} />
        ) : row.predicted_carbon_interval_half_width_g_co2e_per_kwh != null ? (
          <DetailItem label="Carbon interval half-width" value={`${formatFixed(row.predicted_carbon_interval_half_width_g_co2e_per_kwh)} gCO2e/kWh`} />
        ) : null}
        <DetailItem label="Predicted total emissions" value={`${formatNumber(row.predicted_total_emissions_kg_co2e)} kgCO2e`} />
        {row.predicted_avg_carbon_intensity_g_co2e_per_kwh != null && (
          <DetailItem label="Average carbon intensity" value={`${formatFixed(row.predicted_avg_carbon_intensity_g_co2e_per_kwh)} gCO2e/kWh`} />
        )}
        {row.causal_carbon_source && (
          <DetailItem label="Causal-adjusted source" value={formatCausalSource(row.causal_carbon_source)} />
        )}
        {row.causal_adjusted_rank_shift != null && (
          <DetailItem label="Average-vs-causal rank shift" value={formatSigned(row.causal_adjusted_rank_shift)} />
        )}
        {row.predicted_marginal_proxy_confidence && (
          <DetailItem label="Marginal proxy confidence" value={titleCase(row.predicted_marginal_proxy_confidence)} />
        )}
        <DetailItem label="Carbon rank" value={`${row.predicted_carbon_rank ?? scenarioRank} of ${row.candidate_count ?? "-"} candidate hours`} />
        <DetailItem label="Price rank" value={`${priceRank} of ${row.candidate_count ?? "-"} candidate hours`} />
        {row.scenario && (
          <DetailItem
            label="Scenario weights"
            value={`${Math.round((row.scenario_carbon_weight ?? 0) * 100)}% carbon / ${Math.round((row.scenario_price_weight ?? 0) * 100)}% price`}
          />
        )}
        {row.predicted_scenario_score != null && (
          <DetailItem label="Scenario score" value={formatFixed(row.predicted_scenario_score)} />
        )}
        <DetailItem
          label="Carbon saving vs active reference"
          value={`${formatFixed(recommendationCarbonSaving(row))} gCO2e/kWh`}
        />
        {recommendationCostSaving(row) != null && (
          <DetailItem
            label="Cost saving vs active reference"
            value={`${formatFixed(recommendationCostSaving(row))} EUR/MWh`}
          />
        )}
        {row.empirical_top_n_hit_rate != null && (
          <DetailItem label="Historical top-5 hit rate" value={`${Math.round(row.empirical_top_n_hit_rate * 100)}%`} />
        )}
        {row.expected_carbon_regret_g_co2e_per_kwh != null && (
          <DetailItem label="Expected carbon regret" value={`${formatFixed(row.expected_carbon_regret_g_co2e_per_kwh)} gCO2e/kWh`} />
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

function OutcomeVerdict({ row }) {
  const verdict = auditVerdict(row);
  return <span className={`verdict-badge ${verdict.level}`}>{verdict.label}</span>;
}

function DirectionBadge({ value }) {
  const icon =
    value === "increase" ? <ArrowUp size={14} /> : value === "decrease" ? <ArrowDown size={14} /> : <ArrowRight size={14} />;
  return (
    <span className={`direction-badge ${value ?? "unknown"}`}>
      {icon}
      {titleCase(value ?? "unknown")}
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

function buildTrustSummary(payload) {
  const pipeline = payload.summary?.pipeline_health ?? {};
  const forecast = payload.summary?.forecast_monitoring ?? {};
  const readiness = payload.summary?.operational_audit_readiness ?? {};
  const outcome = payload.summary?.recommendation_outcome_audit ?? {};
  const reasons = [
    ...(pipeline.critical_issues ?? []),
    ...(readiness.reasons ?? []),
    ...(readiness.warnings ?? []),
  ];
  if (pipeline.status === "fail" || readiness.status === "fail" || forecast.status === "fail") {
    return {
      level: "blocked",
      label: "Needs attention",
      message: formatTrustReason(reasons[0]) ?? "A required operational data check is failing.",
    };
  }
  if (forecast.stale || forecast.status === "stale" || !outcome.available) {
    return {
      level: "warning",
      label: "Usable with caveats",
      message: forecast.stale
        ? "Forecast monitoring is older than the current model artifacts."
        : "Settled recommendation outcomes are not fully available yet.",
    };
  }
  return {
    level: "ready",
    label: "Fresh and auditable",
    message: "Recommendation data, monitoring, and settled outcome checks are available.",
  };
}

function auditVerdict(row) {
  if (!row) return { level: "unknown", label: "-" };
  const actualRank = Number(row.actual_decision_rank_observed);
  const carbonRegret = Number(row.carbon_regret_g_co2e_per_kwh_observed);
  if (Number.isFinite(actualRank) && actualRank <= 1) return { level: "hit", label: "Hit" };
  if (Number.isFinite(actualRank) && actualRank <= 5) return { level: "good", label: "Good" };
  if (Number.isFinite(carbonRegret) && carbonRegret <= 1) return { level: "ok", label: "Close" };
  return { level: "miss", label: "Miss" };
}

function formatTrustReason(value) {
  if (!value) return null;
  return String(value)
    .split(":")
    .map((part) => part.split("_").map(titleCase).join(" "))
    .join(": ");
}

function formatHour(value) {
  if (!value) return "-";
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

function shortModel(value = "") {
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
  const labels = {
    emissions_reduction: "Emissions Reduction",
    balanced_operations: "Balanced Operations",
    budget_control: "Budget Control",
    clean_first: "Clean First",
    balanced: "Balanced",
    cost_aware_clean: "Cost Aware Clean",
  };
  if (labels[value]) return labels[value];
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
  return new Intl.NumberFormat("en-GB", { maximumFractionDigits: 2 }).format(Number(value));
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

function comparisonRead(value, kind) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "-";
  if (kind === "changed") {
    if (numeric >= 0.5) return "often changes the first choice";
    if (numeric > 0) return "sometimes changes the first choice";
    return "same first choice";
  }
  if (kind === "overlap") {
    if (numeric >= 0.8) return "top set mostly stable";
    if (numeric >= 0.5) return "moderate reshuffle";
    return "large top-set reshuffle";
  }
  if (kind === "coverage") {
    if (numeric >= 0.8) return "usable coverage";
    if (numeric >= 0.5) return "partial coverage";
    return "thin coverage";
  }
  if (numeric >= 3) return "material movement";
  if (numeric > 0) return "small movement";
  return "no movement";
}

function mean(values) {
  const numericValues = values
    .map((value) => Number(value))
    .filter((value) => Number.isFinite(value));
  if (numericValues.length === 0) return null;
  return numericValues.reduce((total, value) => total + value, 0) / numericValues.length;
}

function meanBoolean(values) {
  const numericValues = values
    .map((value) => {
      if (value === true || value === "True" || value === "true" || value === 1) return 1;
      if (value === false || value === "False" || value === "false" || value === 0) return 0;
      return Number.NaN;
    })
    .filter((value) => Number.isFinite(value));
  if (numericValues.length === 0) return null;
  return numericValues.reduce((total, value) => total + value, 0) / numericValues.length;
}

function recommendationSubtitle(selectedBasis, selectedScenario) {
  if (selectedBasis === "causal") {
    return `Top 5 future workload start hours for ${formatScenario(selectedScenario)} using the marginal-emissions proxy MVP.`;
  }
  return `Top 5 future workload start hours for ${formatScenario(selectedScenario)}. Scenario rank and score update with the selector.`;
}

function scenarioPanelSubtitle(selectedBasis, selectedScenario) {
  const carbonBasis = selectedBasis === "causal" ? "marginal-carbon proxy" : "average-carbon";
  return `${formatScenario(selectedScenario)} reranks the same candidate hours using ${carbonBasis} and price weights.`;
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

function recommendationCarbonSaving(row) {
  return (
    row?.carbon_savings_vs_current_reference_g_co2e_per_kwh
    ?? row?.carbon_savings_vs_run_now_g_co2e_per_kwh
  );
}

function recommendationCostSaving(row) {
  return row?.cost_savings_vs_current_reference_eur_mwh ?? row?.cost_savings_vs_run_now_eur_mwh;
}

function recommendationReferenceLabel(row) {
  if (row?.current_reference_start_utc) {
    return `gCO2e/kWh vs active reference ${formatHour(row.current_reference_start_utc)} UTC`;
  }
  return "gCO2e/kWh vs active reference";
}

function recommendationExplanation(row) {
  if (!row) return "-";
  const parts = [];
  const carbonRank = row.predicted_carbon_rank;
  const candidateCount = row.candidate_count;
  const carbonIntensity = recommendationCarbonIntensity(row);
  if (carbonRank != null && candidateCount != null) {
    parts.push(
      `Ranked ${formatOrdinal(carbonRank)} for carbon among ${candidateCount} active candidate hours`,
    );
  } else if (carbonIntensity != null) {
    parts.push(`${recommendationCarbonLabel(row)} carbon is ${formatFixed(carbonIntensity)} gCO2e/kWh`);
  }

  const carbonSaving = recommendationCarbonSaving(row);
  if (carbonSaving != null) {
    const referenceHour = row.current_reference_start_utc
      ? ` against the ${formatHour(row.current_reference_start_utc)} UTC active reference`
      : " against the active reference";
    parts.push(`${formatSigned(carbonSaving)} gCO2e/kWh${referenceHour}`);
  }

  if (row.confidence_score != null && row.confidence_level) {
    parts.push(
      `${titleCase(row.confidence_level)} confidence from rank margin and historical calibration`,
    );
  }

  if (row.recommendation_status === "no_low_risk_recommendation_available") {
    parts.push("shown as the best available hour, but uncertainty is elevated");
  } else if (row.decision_uncertainty_score != null && Number(row.decision_uncertainty_score) <= 0.85) {
    parts.push("passes the uncertainty guard");
  }

  if (row.causal_carbon_source) {
    const shift = row.causal_adjusted_rank_shift == null
      ? ""
      : ` with ${formatSigned(row.causal_adjusted_rank_shift)} rank shift`;
    parts.push(`${formatCausalSource(row.causal_carbon_source)}${shift}`);
  } else if (row.scenario) {
    parts.push(`${formatScenario(row.scenario)} weighting sets the final order`);
  }

  const explanation = parts.filter(Boolean).join(". ");
  return explanation ? `${explanation}.` : "No explanation metrics are available for this recommendation.";
}

function formatOrdinal(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "-";
  const rounded = Math.round(numeric);
  const suffix =
    rounded % 100 >= 11 && rounded % 100 <= 13
      ? "th"
      : { 1: "st", 2: "nd", 3: "rd" }[rounded % 10] ?? "th";
  return `${rounded}${suffix}`;
}

function titleCase(value) {
  if (!value) return "";
  return value.charAt(0).toUpperCase() + value.slice(1);
}

const rootElement = document.getElementById("root");
window.__cleanHourRoot = window.__cleanHourRoot ?? createRoot(rootElement);
window.__cleanHourRoot.render(<App />);
