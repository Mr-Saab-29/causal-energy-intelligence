import assert from "node:assert/strict";
import test from "node:test";
import { validateCreate, validateUpdate } from "../api/workload-decisions.js";

const decisionId = "018f3f22-c3bd-4b6f-8c7f-d69bbdb189be";

function createPayload() {
  return {
    decision_id: decisionId,
    dashboard_accessed_at_utc: "2026-10-06T08:00:00Z",
    user_planned_start_utc: "2026-10-06T12:00:00Z",
    baseline_source: "user_planned_start_time",
    baseline_start_utc: "2026-10-06T12:00:00Z",
    recommendation_generated_at_utc: "2026-10-06T07:55:00Z",
    recommendation_basis: "scenario",
    scenario: "emissions_reduction",
    recommended_start_utc: "2026-10-06T14:00:00Z",
    candidate_alternatives: [
      {
        timestamp_utc: "2026-10-06T14:00:00Z",
        rank: 1,
        predicted_carbon_gco2_kwh: 24.5,
        predicted_price_eur_mwh: 40.0,
        confidence_score: 0.72,
        recommendation_status: "recommended",
      },
    ],
    workload_type: "data_center_batch",
    planned_energy_kwh: 1000,
    planned_duration_minutes: 60,
    constraints: {
      earliest_start_utc: "2026-10-06T12:00:00Z",
      latest_completion_utc: "2026-10-07T06:00:00Z",
      max_delay_minutes: 720,
    },
  };
}

test("create validates and keeps the recommendation exposure snapshot", () => {
  const row = validateCreate(createPayload());

  assert.equal(row.decision_id, decisionId);
  assert.equal(row.status, "planned");
  assert.equal(row.candidate_alternatives.length, 1);
  assert.equal(row.candidate_alternatives[0].predicted_carbon_gco2_kwh, 24.5);
  assert.equal(row.constraints.max_delay_minutes, 720);
});

test("create rejects a recommended start absent from shown candidates", () => {
  const payload = createPayload();
  payload.recommended_start_utc = "2026-10-06T15:00:00Z";

  assert.throws(() => validateCreate(payload), /must be present/);
});

test("create rejects a baseline inconsistent with its declared source", () => {
  const payload = createPayload();
  payload.baseline_start_utc = "2026-10-06T08:00:00Z";

  assert.throws(() => validateCreate(payload), /must match the selected baseline source/);
});

test("selection captures recommended, alternative, or custom timing", () => {
  const result = validateUpdate({
    decision_id: decisionId,
    operation: "select",
    user_selected_start_utc: "2026-10-06T15:00:00Z",
    selection_source: "candidate_alternative",
  });

  assert.equal(result.updates.status, "selected");
  assert.equal(result.updates.selection_source, "candidate_alternative");
  assert.deepEqual(result.expectedStatuses, ["planned", "selected"]);
});

test("completion rejects an inverted execution interval", () => {
  assert.throws(
    () =>
      validateUpdate({
        decision_id: decisionId,
        operation: "complete",
        actual_start_utc: "2026-10-06T15:00:00Z",
        actual_completion_utc: "2026-10-06T14:00:00Z",
        actual_energy_kwh: 900,
      }),
    /must be after/,
  );
});
