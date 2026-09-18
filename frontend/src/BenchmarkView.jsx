import React, { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import "./benchmark.css";

const names = { lightgbm_quantile: "LightGBM · quantile", ridge_residual: "Ridge · residual quantiles", naive_24_residual: "Daily naive · residual quantiles", naive_168_residual: "Weekly naive · residual quantiles", "t0-alpha": "t0-alpha · zero-shot", lightgbm: "LightGBM", ridge: "Ridge", naive_24: "Daily seasonal naive", naive_168: "Weekly seasonal naive" };
const variants = { calendar_30d: "Calendar · 30d context", target_30d: "Target history only", weather_30d: "Calendar + past weather", related_30d: "Calendar + related history", calendar_7d: "Calendar · 7d context", calendar_90d: "Calendar · 90d context", dropout_10: "10% weather dropout", dropout_30: "30% weather dropout", outage_6h: "6h weather outage", noise_10: "Weather noise · 0.1σ", noise_30: "Weather noise · 0.3σ" };
const descriptions = {
  E1: "Frozen t0 forecasts versus separately trained LightGBM, Ridge and seasonal-naive references. Lower error is better.",
  E2: "Compare target history, calendar, historical weather and related electricity inputs. Untrained baseline configurations are unavailable.",
  E3: "Score the first 6, 12 or 24 hours of the same forecast. Inputs stay fixed at the daily forecast origin.",
  E4: "Change t0’s input history while keeping calendar inputs and evaluation dates fixed. Trained baselines have fixed features and are N/A here.",
  E5: "Corrupt historical weather, then forward-fill up to 24h and use pre-period means if needed. Compare with clean historical weather. Noise σ uses pre-period variability.",
  E6: "Evaluate unusual hours using thresholds fixed from pre-evaluation data. Fewer than ten contributing days is marked limited evidence.",
  E7: "Compare t0 and probabilistic local variants at the same 10th/50th/90th percentiles. The 80% interval targets 80% coverage; width and pinball loss measure precision. Original point-only models remain N/A.",
  E8: "Compare offline scheduling with identical 80% carbon / 20% price weights, a one-hour 1 MWh workload and previous-day price forecasts. Savings can be negative.",
};
const fmt = (value, digits = 2) => value == null || !Number.isFinite(Number(value)) ? "N/A" : Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
const pct = (value) => value == null ? "N/A" : `${fmt(value * 100, 1)}%`;
const targetName = (value) => value.replace(/_mwh$/, "").replaceAll("_", " ");

export default function BenchmarkView() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [experiment, setExperiment] = useState("E1");
  const [windowDays, setWindowDays] = useState(90);
  const [target, setTarget] = useState("consumption_mwh");
  const [model, setModel] = useState("all");
  const [horizon, setHorizon] = useState(24);
  const [regime, setRegime] = useState("all");
  useEffect(() => {
    let active = true;
    fetch("/data/benchmark.json").then(response => {
      if (!response.ok) throw new Error("No benchmark snapshot has been published yet.");
      return response.json();
    }).then(value => { if (active) setData(value); }).catch(reason => { if (active) setError(reason.message); });
    return () => { active = false; };
  }, []);
  const rows = useMemo(() => (data?.metrics ?? []).filter(row =>
    row.experiment === experiment && row.window_days === windowDays &&
    (experiment === "E8" || row.target === target) && (model === "all" || row.model === model) &&
    row.horizon === (["E3", "E7"].includes(experiment) ? horizon : 24) &&
    row.regime === (["E6", "E7"].includes(experiment) ? regime : "all")
  ).map(row => ({ ...row, label: `${names[row.model] ?? row.model}${["E2", "E4", "E5"].includes(experiment) ? ` / ${variants[row.variant] ?? row.variant}` : ""}` })), [data, experiment, windowDays, target, model, horizon, regime]);
  if (error) return <section className="panel benchmark-empty"><h2>Benchmarks</h2><p>{error}</p><p>Results appear here after a manually run evaluation. Opening this tab does not run inference.</p></section>;
  if (!data) return <section className="panel benchmark-empty" aria-live="polite">Loading the saved benchmark…</section>;
  const complete = data.status === "completed";
  const corrected = data.protocol?.config?.energy_aggregation_version === "source_interval_v2";
  const finished = data.availability.reduce((sum, item) => sum + item.completed_days, 0);
  const total = data.availability.reduce((sum, item) => sum + item.expected_days, 0);
  const calibration = [10, 50, 90].map(level => ({ nominal: level, ideal: level, ...Object.fromEntries(rows.map(row => [row.model, row[`cdf${level}`] == null ? null : row[`cdf${level}`] * 100])) }));
  const calibrationColors = ["#1f8a70", "#b06b25", "#536dc0", "#a45279", "#78812d"];
  const metric = experiment === "E8" ? "carbon_regret" : experiment === "E7" ? "pinball" : "mae";
  const metricLabel = experiment === "E8" ? "Carbon regret (gCO₂e/kWh)" : experiment === "E7" ? "Mean pinball loss (MWh)" : "Mean absolute error (MWh)";
  const dayCounts = [...new Set(rows.map(row => row.days))];
  const filteredExamples = (data.decision_examples ?? []).filter(row => model === "all" || row.model === model)
    .filter(row => new Date(row.origin) >= new Date(new Date(data.date_end).getTime() - (windowDays * 24 - 1) * 3600000))
    .sort((a, b) => a.origin.localeCompare(b.origin) || a.model.localeCompare(b.model)).slice(-10);
  return <div className="benchmark-view">
    <section className="panel benchmark-intro">
      <div><p className="eyebrow">Frozen research snapshot</p><h2>How do the models hold up?</h2>
        <p>{data.date_start.slice(0, 10)} — {data.date_end.slice(0, 10)} · daily forecast origins · France</p></div>
      <span className={`benchmark-status ${complete && corrected ? "done" : "pending"}`}>{!corrected ? "Needs recomputing" : complete ? "Complete" : "Partial results"} · {finished}/{total} API requests</span>
    </section>
    {!corrected && <div className="benchmark-notice" role="alert"><strong>Results need recomputing.</strong> Historical energy was undercounted in this snapshot. These scores should not be used to rank models. Corrected results will appear after the data and benchmark are rebuilt.</div>}
    {!complete && <div className="benchmark-notice" role="status">This run is {data.run_status.status.replaceAll("_", " ")}. Missing configurations are pending, not zero-error results. Comparisons use common dates among available configurations.</div>}
    <section className="controls-band benchmark-controls">
      <label><span>Evaluation window</span><select value={windowDays} onChange={e => setWindowDays(Number(e.target.value))}><option value={90}>Full 90 days</option><option value={28}>Final 28 days</option><option value={7}>Final 7 days</option></select></label>
      <label><span>Target</span><select value={target} disabled={experiment === "E8"} onChange={e => setTarget(e.target.value)}>{data.targets.map(item => <option key={item} value={item}>{targetName(item)}</option>)}</select></label>
      <label><span>Model</span><select value={model} onChange={e => setModel(e.target.value)}><option value="all">All models</option>{data.models.map(item => <option key={item} value={item}>{names[item] ?? item}</option>)}</select></label>
      {["E3", "E7"].includes(experiment) && <label><span>Forecast horizon</span><select value={horizon} onChange={e => setHorizon(Number(e.target.value))}>{[6, 12, 24].map(item => <option key={item} value={item}>{item} hours</option>)}</select></label>}
      {["E6", "E7"].includes(experiment) && <label><span>Conditions</span><select value={regime} onChange={e => setRegime(e.target.value)}>{["all", "low_demand", "high_demand", "cold", "hot", "low_wind", "high_wind", "low_generation", "high_generation"].map(item => <option key={item} value={item}>{item.replaceAll("_", " ")}</option>)}</select></label>}
    </section>
    <nav className="benchmark-experiments" aria-label="Benchmark experiments">{Object.entries(data.experiments).map(([key, title]) => <button key={key} type="button" className={experiment === key ? "active" : ""} onClick={() => setExperiment(key)} aria-pressed={experiment === key}><strong>{key}</strong>{title}</button>)}</nav>
    <section className="panel benchmark-results">
      <div className="panel-heading"><div><h2>{experiment} · {data.experiments[experiment]}</h2><p>{descriptions[experiment]}</p></div></div>
      <p className="benchmark-caption">{rows.length ? `${dayCounts.join(" / ")} contributing days · ${experiment === "E8" ? "all generation sources" : targetName(target)} · ${["E3", "E7"].includes(experiment) ? horizon : 24}h horizon` : "No comparable results for this selection."}</p>
      {rows.length > 0 && <>
        <div className="benchmark-chart" style={{ height: Math.max(240, rows.length * 45) }}>
          <ResponsiveContainer width="100%" height="100%"><BarChart data={rows} layout="vertical" margin={{ left: 10, right: 36, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false}/><XAxis type="number" label={{ value: metricLabel, position: "insideBottom", offset: -12 }}/><YAxis type="category" dataKey="label" width={230} tick={{ fontSize: 11 }}/><Tooltip formatter={value => fmt(value)}/><Bar dataKey={metric} name={metricLabel} fill="#1f8a70" radius={[0, 4, 4, 0]}/>
          </BarChart></ResponsiveContainer>
        </div>
        {experiment === "E7" && <div className="benchmark-calibration"><h3>Quantile calibration</h3><p>Observed fractions below each predicted quantile should track the diagonal.</p><ResponsiveContainer width="100%" height={240}><LineChart data={calibration} margin={{ top: 15, right: 30, bottom: 20, left: 5 }}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="nominal" type="number" domain={[0, 100]} unit="%"/><YAxis domain={[0, 100]} unit="%"/><Tooltip formatter={value => `${fmt(value, 1)}%`}/><Legend/><Line dataKey="ideal" stroke="#999" strokeDasharray="4 4" name="Ideal" dot={false}/>{rows.map((row, index) => <Line key={row.model} dataKey={row.model} stroke={calibrationColors[index % calibrationColors.length]} name={names[row.model] ?? row.model}/>)}</LineChart></ResponsiveContainer></div>}
        <div className="benchmark-table-wrap"><table><thead><tr><th>Model / configuration</th><th>Days</th>{experiment === "E8" ? <><th>Carbon regret</th><th>Cost regret €/MWh</th><th>Carbon savings</th><th>Cost savings €/MWh</th><th>Top-5 overlap</th><th>Best-hour capture</th></> : <><th>MAE</th><th>RMSE</th><th>MASE</th><th>80% coverage</th><th>Interval width</th><th>Pinball loss</th></>}</tr></thead><tbody>{rows.map(row => <tr key={`${row.model}-${row.variant}`}><th>{row.label}{row.evidence === "limited" && <small>Limited evidence</small>}</th><td>{row.days}</td>{experiment === "E8" ? <><td>{fmt(row.carbon_regret)}</td><td>{fmt(row.cost_regret)}</td><td>{fmt(row.carbon_savings)}</td><td>{fmt(row.cost_savings)}</td><td>{pct(row.top5_overlap)}</td><td>{pct(row.best_hour_capture)}</td></> : <><td>{fmt(row.mae)}</td><td>{fmt(row.rmse)}</td><td>{fmt(row.mase, 3)}</td><td>{pct(row.coverage80)}</td><td>{fmt(row.width80)}</td><td>{fmt(row.pinball)}</td></>}</tr>)}</tbody></table></div>
        <p className="benchmark-caption">{experiment === "E8" ? "Carbon values are gCO₂e/kWh (numerically kgCO₂e for the fixed 1 MWh workload). Regret is relative to the best observed hour; savings are relative to execution at the forecast origin." : "Errors and interval widths use MWh. MASE divides MAE by pre-period weekly seasonal-naive error; below 1 beats that historical scale. N/A means the model does not provide the required output."}</p>
      </>}
      {rows.length === 0 && <div className="benchmark-empty">This combination is unavailable, still pending, or has no observed hours in the selected regime.</div>}
    </section>
    {experiment === "E8" && filteredExamples.length > 0 && <section className="panel benchmark-results"><h2>Saved scheduling examples</h2><div className="benchmark-table-wrap"><table><thead><tr><th>Origin (UTC)</th><th>Model</th><th>Selected hour (UTC)</th><th>Best observed hour (UTC)</th><th>Selected carbon</th><th>Run-now carbon</th></tr></thead><tbody>{filteredExamples.map(row => <tr key={`${row.origin}-${row.model}`}><td>{row.origin.slice(0, 10)}</td><td>{names[row.model]}</td><td>{row.chosen_hour.slice(11, 16)}</td><td>{row.oracle_hour.slice(11, 16)}</td><td>{fmt(row.selected_carbon)}</td><td>{fmt(row.run_now_carbon)}</td></tr>)}</tbody></table></div></section>}
    <details className="panel benchmark-methodology"><summary>Methodology, coverage and reproducibility</summary><p>Generated {new Date(data.generated_at_utc).toLocaleString()} · {data.snapshot_id}</p><p>The final 7/28-day views are slices of this fixed period, not live updates. Baselines train before May 26, 2026. Foundation-model inference is zero-shot.</p><ul>{data.limitations.map(item => <li key={item}>{item}</li>)}</ul><div className="benchmark-table-wrap"><table><thead><tr><th>t0 configuration</th><th>Completed dates</th><th>Status</th></tr></thead><tbody>{data.availability.map(item => <tr key={item.variant}><td>{variants[item.variant]}</td><td>{item.completed_days} / {item.expected_days}</td><td>{item.completed_days === item.expected_days ? "Complete" : "Pending"}</td></tr>)}</tbody></table></div><h3>Inference timings</h3><div className="benchmark-table-wrap"><table><thead><tr><th>Model / configuration</th><th>Measurement</th><th>Calls</th><th>Median seconds</th><th>p95 seconds</th></tr></thead><tbody>{(data.timing_summary ?? []).map(item => <tr key={`${item.model}-${item.variant}`}><td>{names[item.model]} / {variants[item.variant]}</td><td>{item.unit}</td><td>{item.calls}</td><td>{fmt(item.median_seconds, 4)}</td><td>{fmt(item.p95_seconds, 4)}</td></tr>)}</tbody></table></div><p>Server peak memory is unavailable through the API. Client API times include network and serving overhead.</p><a href="/data/benchmark.json" download>Download the benchmark results and protocol</a></details>
  </div>;
}
