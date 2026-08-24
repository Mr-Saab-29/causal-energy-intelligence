import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";

const dashboardPath = resolve("public/data/dashboard.json");
const fallbackDashboard = {
  data_state: {
    mode: "sample",
    message: "No live recommendation data has been published for this deployment yet.",
    next_step: "Run the ingestion and recommendation refresh, then rebuild the dashboard data contract.",
  },
  generated_from: {
    champion_model_selection: null,
    recommendations: null,
    scenario_recommendations: null,
    future_scenario_recommendations: null,
  },
  champion: {
    model: null,
    display_model_name: "Production Model V1",
    weights: {},
    selection_rule: null,
    models: [],
  },
  summary: {
    pipeline_health: {
      status: "unknown",
      generated_at_utc: null,
      critical_issue_count: 0,
      warning_count: 1,
      latest_data_timestamp_utc: null,
    },
    forecast_monitoring: {
      status: "unknown",
      generated_at_utc: null,
      retraining_recommended: false,
      stale: false,
      reason_count: 0,
      warning_count: 1,
      latest_actual_timestamp_utc: null,
      champion_model: null,
    },
    date_count: 0,
    recommendation_count: 0,
    future_recommendation_file_rows: 0,
    active_future_recommendation_count: 0,
    future_scenario_file_rows: 0,
    active_future_scenario_count: 0,
    stale_future_recommendations: false,
    stale_future_scenarios: false,
    average_confidence_score: null,
    high_confidence_share: null,
  },
  pipeline_health: {
    status: "unknown",
    sources: {},
    warnings: ["dashboard_sample_data"],
    critical: [],
  },
  forecast_monitoring: {
    status: "unknown",
    warnings: ["dashboard_sample_data"],
    reasons: [],
  },
  filters: {
    dates: [],
    scenarios: ["clean_first", "balanced", "cost_aware_clean"],
  },
  recommendations: [],
  scenario_recommendations: [],
};

if (!existsSync(dashboardPath)) {
  mkdirSync(dirname(dashboardPath), { recursive: true });
  writeFileSync(dashboardPath, `${JSON.stringify(fallbackDashboard, null, 2)}\n`);
  console.log("Using sample dashboard data for this build.");
} else {
  console.log("Using existing dashboard data for this build.");
}
