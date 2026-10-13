create table if not exists observed_workload_decisions (
    decision_id uuid primary key,
    schema_version text not null default 'observed_workload_decision_v1',
    dashboard_accessed_at_utc timestamptz not null,
    user_planned_start_utc timestamptz,
    baseline_source text not null check (
        baseline_source in ('dashboard_access_time', 'user_planned_start_time')
    ),
    baseline_start_utc timestamptz not null,
    recommendation_generated_at_utc timestamptz,
    recommendation_basis text not null,
    scenario text not null,
    recommended_start_utc timestamptz not null,
    candidate_alternatives jsonb not null check (
        jsonb_typeof(candidate_alternatives) = 'array'
        and jsonb_array_length(candidate_alternatives) between 1 and 20
    ),
    user_selected_start_utc timestamptz,
    selection_source text check (
        selection_source is null
        or selection_source in ('recommended', 'candidate_alternative', 'custom')
    ),
    selection_recorded_at_utc timestamptz,
    actual_start_utc timestamptz,
    actual_completion_utc timestamptz,
    planned_energy_kwh double precision not null check (planned_energy_kwh > 0),
    actual_energy_kwh double precision check (
        actual_energy_kwh is null or actual_energy_kwh > 0
    ),
    planned_duration_minutes integer not null check (planned_duration_minutes > 0),
    workload_type text not null check (
        workload_type in (
            'data_center_batch',
            'ev_charging',
            'battery_charging',
            'industrial_batch',
            'other'
        )
    ),
    constraints jsonb not null default '{}'::jsonb check (
        jsonb_typeof(constraints) = 'object'
    ),
    status text not null default 'planned' check (
        status in ('planned', 'selected', 'completed')
    ),
    created_at_utc timestamptz not null default now(),
    updated_at_utc timestamptz not null default now(),
    check (
        (baseline_source = 'dashboard_access_time' and user_planned_start_utc is null)
        or (baseline_source = 'user_planned_start_time' and user_planned_start_utc is not null)
    ),
    check (
        actual_completion_utc is null
        or (actual_start_utc is not null and actual_completion_utc > actual_start_utc)
    ),
    check (
        (user_selected_start_utc is null and selection_source is null)
        or (user_selected_start_utc is not null and selection_source is not null)
    )
);

create index if not exists idx_observed_workload_decisions_accessed
    on observed_workload_decisions (dashboard_accessed_at_utc);

create index if not exists idx_observed_workload_decisions_status
    on observed_workload_decisions (status, actual_completion_utc);

alter table observed_workload_decisions
    drop constraint if exists observed_workload_baseline_matches;
alter table observed_workload_decisions
    add constraint observed_workload_baseline_matches check (
        (
            baseline_source = 'dashboard_access_time'
            and user_planned_start_utc is null
            and baseline_start_utc = dashboard_accessed_at_utc
        )
        or (
            baseline_source = 'user_planned_start_time'
            and user_planned_start_utc is not null
            and baseline_start_utc = user_planned_start_utc
        )
    );

alter table observed_workload_decisions
    drop constraint if exists observed_workload_status_matches;
alter table observed_workload_decisions
    add constraint observed_workload_status_matches check (
        status = 'planned'
        or (
            status = 'selected'
            and user_selected_start_utc is not null
            and selection_source is not null
        )
        or (
            status = 'completed'
            and user_selected_start_utc is not null
            and selection_source is not null
            and actual_start_utc is not null
            and actual_completion_utc is not null
            and actual_energy_kwh is not null
        )
    );

drop view if exists observed_workload_treatment_analysis;

create view observed_workload_treatment_analysis as
select
    decision_id,
    dashboard_accessed_at_utc,
    user_planned_start_utc,
    baseline_source,
    baseline_start_utc,
    recommendation_generated_at_utc,
    recommended_start_utc,
    user_selected_start_utc,
    selection_recorded_at_utc,
    actual_start_utc,
    actual_completion_utc,
    recommendation_basis,
    scenario,
    workload_type,
    planned_energy_kwh,
    actual_energy_kwh,
    planned_duration_minutes,
    extract(epoch from (actual_completion_utc - actual_start_utc)) / 60.0
        as actual_duration_minutes,
    extract(epoch from (recommended_start_utc - baseline_start_utc)) / 60.0
        as recommended_shift_minutes,
    extract(epoch from (user_selected_start_utc - baseline_start_utc)) / 60.0
        as selected_shift_minutes,
    extract(epoch from (actual_start_utc - baseline_start_utc)) / 60.0
        as realized_shift_minutes,
    case
        when user_selected_start_utc is null then null
        else abs(extract(epoch from (user_selected_start_utc - recommended_start_utc))) < 60
    end as recommendation_accepted,
    candidate_alternatives,
    constraints,
    status,
    created_at_utc,
    updated_at_utc
from observed_workload_decisions;

comment on table observed_workload_decisions is
    'Decision-level recommendation exposure, workload choice, and execution outcome without direct personal identifiers.';

alter table observed_workload_decisions enable row level security;
revoke all on observed_workload_decisions from anon, authenticated;
grant all on observed_workload_decisions to service_role;
revoke all on observed_workload_treatment_analysis from anon, authenticated;
grant select on observed_workload_treatment_analysis to service_role;
