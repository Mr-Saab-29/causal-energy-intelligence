const WORKLOAD_TYPES = new Set([
  "data_center_batch",
  "ev_charging",
  "battery_charging",
  "industrial_batch",
  "other",
]);
const BASELINE_SOURCES = new Set([
  "dashboard_access_time",
  "user_planned_start_time",
]);
const SELECTION_SOURCES = new Set([
  "recommended",
  "candidate_alternative",
  "custom",
]);
const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export default async function handler(request, response) {
  response.setHeader("Cache-Control", "no-store");
  if (request.method === "OPTIONS") {
    response.status(204).end();
    return;
  }
  if (!sameOriginRequest(request)) {
    sendError(response, 403, "Cross-origin treatment writes are not allowed.");
    return;
  }
  const config = getSupabaseConfig();
  if (!config) {
    sendError(response, 503, "Treatment collection is not configured.");
    return;
  }
  try {
    const body = parseBody(request.body);
    if (request.method === "POST") {
      const row = validateCreate(body);
      await writeSupabase(config, "POST", "", row, "return=minimal");
      response.status(201).json({ decision_id: row.decision_id, status: row.status });
      return;
    }
    if (request.method === "PATCH") {
      const { decisionId, expectedStatuses, updates } = validateUpdate(body);
      const rows = await writeSupabase(
        config,
        "PATCH",
        `?decision_id=eq.${encodeURIComponent(decisionId)}&status=in.(${expectedStatuses.join(",")})`,
        updates,
        "return=representation",
      );
      if (!Array.isArray(rows) || rows.length === 0) {
        const error = new Error("The workload observation is missing or in the wrong stage.");
        error.statusCode = 409;
        throw error;
      }
      response.status(200).json({ decision_id: decisionId, status: updates.status });
      return;
    }
    response.setHeader("Allow", "POST, PATCH, OPTIONS");
    sendError(response, 405, "Method not allowed.");
  } catch (error) {
    const status = error.statusCode ?? 400;
    sendError(response, status, status >= 500 ? "Treatment write failed." : error.message);
  }
}

function validateCreate(body) {
  requireUuid(body.decision_id);
  const dashboardAccess = requireIso(
    body.dashboard_accessed_at_utc,
    "dashboard_accessed_at_utc",
  );
  const baselineStart = requireIso(body.baseline_start_utc, "baseline_start_utc");
  const recommendedStart = requireIso(
    body.recommended_start_utc,
    "recommended_start_utc",
  );
  if (!BASELINE_SOURCES.has(body.baseline_source)) {
    fail("baseline_source is invalid.");
  }
  const plannedStart = optionalIso(body.user_planned_start_utc, "user_planned_start_utc");
  if ((body.baseline_source === "user_planned_start_time") !== Boolean(plannedStart)) {
    fail("A planned baseline timestamp is required only for a planned-start baseline.");
  }
  const expectedBaseline = plannedStart ?? dashboardAccess;
  if (baselineStart !== expectedBaseline) {
    fail("baseline_start_utc must match the selected baseline source.");
  }
  if (!WORKLOAD_TYPES.has(body.workload_type)) {
    fail("workload_type is invalid.");
  }
  const candidates = sanitizeCandidates(body.candidate_alternatives);
  if (!candidates.some((item) => item.timestamp_utc === recommendedStart)) {
    fail("recommended_start_utc must be present in candidate_alternatives.");
  }
  return {
    decision_id: body.decision_id,
    schema_version: "observed_workload_decision_v1",
    dashboard_accessed_at_utc: dashboardAccess,
    user_planned_start_utc: plannedStart,
    baseline_source: body.baseline_source,
    baseline_start_utc: baselineStart,
    recommendation_generated_at_utc: requireIso(
      body.recommendation_generated_at_utc,
      "recommendation_generated_at_utc",
    ),
    recommendation_basis: requireShortText(body.recommendation_basis, "recommendation_basis"),
    scenario: requireShortText(body.scenario, "scenario"),
    recommended_start_utc: recommendedStart,
    candidate_alternatives: candidates,
    planned_energy_kwh: requirePositiveNumber(body.planned_energy_kwh, "planned_energy_kwh"),
    planned_duration_minutes: requirePositiveInteger(
      body.planned_duration_minutes,
      "planned_duration_minutes",
    ),
    workload_type: body.workload_type,
    constraints: sanitizeConstraints(body.constraints),
    status: "planned",
  };
}

function validateUpdate(body) {
  const decisionId = requireUuid(body.decision_id);
  const now = new Date().toISOString();
  if (body.operation === "select") {
    if (!SELECTION_SOURCES.has(body.selection_source)) {
      fail("selection_source is invalid.");
    }
    return {
      decisionId,
      expectedStatuses: ["planned", "selected"],
      updates: {
        user_selected_start_utc: requireIso(
          body.user_selected_start_utc,
          "user_selected_start_utc",
        ),
        selection_source: body.selection_source,
        selection_recorded_at_utc: now,
        status: "selected",
        updated_at_utc: now,
      },
    };
  }
  if (body.operation === "complete") {
    const actualStart = requireIso(body.actual_start_utc, "actual_start_utc");
    const actualCompletion = requireIso(
      body.actual_completion_utc,
      "actual_completion_utc",
    );
    if (new Date(actualCompletion) <= new Date(actualStart)) {
      fail("actual_completion_utc must be after actual_start_utc.");
    }
    return {
      decisionId,
      expectedStatuses: ["selected", "completed"],
      updates: {
        actual_start_utc: actualStart,
        actual_completion_utc: actualCompletion,
        actual_energy_kwh: requirePositiveNumber(body.actual_energy_kwh, "actual_energy_kwh"),
        status: "completed",
        updated_at_utc: now,
      },
    };
  }
  fail("operation must be select or complete.");
}

function sanitizeCandidates(value) {
  if (!Array.isArray(value) || value.length < 1 || value.length > 20) {
    fail("candidate_alternatives must contain between 1 and 20 rows.");
  }
  return value.map((item) => ({
    timestamp_utc: requireIso(item.timestamp_utc, "candidate timestamp_utc"),
    rank: requirePositiveInteger(item.rank, "candidate rank"),
    predicted_carbon_gco2_kwh: optionalFiniteNumber(item.predicted_carbon_gco2_kwh),
    predicted_price_eur_mwh: optionalFiniteNumber(item.predicted_price_eur_mwh),
    confidence_score: optionalBoundedNumber(item.confidence_score, 0, 1),
    recommendation_status: requireShortText(
      item.recommendation_status ?? "recommended",
      "candidate recommendation_status",
    ),
  }));
}

function sanitizeConstraints(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  return {
    earliest_start_utc: optionalIso(value.earliest_start_utc, "earliest_start_utc"),
    latest_completion_utc: optionalIso(
      value.latest_completion_utc,
      "latest_completion_utc",
    ),
    max_delay_minutes: optionalNonNegativeInteger(value.max_delay_minutes),
  };
}

function getSupabaseConfig() {
  const projectUrl = process.env.SUPABASE_PROJECT_URL?.replace(/\/$/, "");
  const serviceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  return projectUrl && serviceKey ? { projectUrl, serviceKey } : null;
}

async function writeSupabase(config, method, query, body, prefer) {
  const upstream = await fetch(
    `${config.projectUrl}/rest/v1/observed_workload_decisions${query}`,
    {
      method,
      headers: {
        apikey: config.serviceKey,
        Authorization: `Bearer ${config.serviceKey}`,
        "Content-Type": "application/json",
        Prefer: prefer,
      },
      body: JSON.stringify(body),
    },
  );
  if (!upstream.ok) {
    const error = new Error(`Supabase treatment write returned HTTP ${upstream.status}.`);
    error.statusCode = upstream.status >= 500 ? 502 : 400;
    throw error;
  }
  if (prefer === "return=representation") return upstream.json();
  return null;
}

function sameOriginRequest(request) {
  const origin = request.headers?.origin;
  if (!origin) return false;
  const allowed = process.env.TREATMENT_ALLOWED_ORIGIN;
  if (allowed && origin === allowed) return true;
  const host = request.headers?.["x-forwarded-host"] ?? request.headers?.host;
  try {
    return Boolean(host) && new URL(origin).host === host;
  } catch {
    return false;
  }
}

function parseBody(body) {
  if (body && typeof body === "object") return body;
  try {
    return JSON.parse(body ?? "{}");
  } catch {
    fail("Request body must be valid JSON.");
  }
}

function requireUuid(value) {
  if (!UUID_PATTERN.test(value ?? "")) fail("decision_id must be a UUID.");
  return value;
}

function requireIso(value, name) {
  if (typeof value !== "string" || !Number.isFinite(Date.parse(value))) {
    fail(`${name} must be an ISO timestamp.`);
  }
  return new Date(value).toISOString();
}

function optionalIso(value, name) {
  return value == null || value === "" ? null : requireIso(value, name);
}

function requirePositiveNumber(value, name) {
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) fail(`${name} must be positive.`);
  return number;
}

function requirePositiveInteger(value, name) {
  const number = Number(value);
  if (!Number.isInteger(number) || number <= 0) fail(`${name} must be a positive integer.`);
  return number;
}

function optionalNonNegativeInteger(value) {
  if (value == null || value === "") return null;
  const number = Number(value);
  if (!Number.isInteger(number) || number < 0) fail("max_delay_minutes must be non-negative.");
  return number;
}

function optionalFiniteNumber(value) {
  if (value == null || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function optionalBoundedNumber(value, minimum, maximum) {
  const number = optionalFiniteNumber(value);
  return number != null && number >= minimum && number <= maximum ? number : null;
}

function requireShortText(value, name) {
  if (typeof value !== "string" || value.length < 1 || value.length > 80) {
    fail(`${name} must be between 1 and 80 characters.`);
  }
  return value;
}

function fail(message) {
  const error = new Error(message);
  error.statusCode = 400;
  throw error;
}

function sendError(response, status, message) {
  response.status(status).json({ error: message });
}

export { validateCreate, validateUpdate };
