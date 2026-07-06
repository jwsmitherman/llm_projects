-- =============================================================================
-- med_nec_discovery.sql
-- GMR / Transport.net (transbroker) - Medical Necessity POV
--
-- Purpose: locate the clinical / PCS free-text + JSON in the Transport.net data
-- and stand up a first extraction for the medical-necessity dataset.
--
-- HOW TO USE IN DATABRICKS SQL EDITOR:
--   Run ONE block at a time (highlight the block, then "Run selected").
--   Start at Section 1. Use its output to fill in the TODOs in Section 3.
--
-- What is known from the environment (confirmed from your editor):
--   catalog: prod
--   schemas: silver_transbroker         (Transport.net ordering / concierge)
--            silver_transbrokerdocs      (docs, e.g. tripridesharereceipts)
--            silver_operationaldw        (operational DW star schema: FactCall,
--                                         FactCrewManifest, Dim* tables)
-- Everything marked TODO is inferred and must be confirmed with Sections 1-2.
-- =============================================================================


-- =============================================================================
-- SECTION 1 - DISCOVERY  (these run as-is; no edits needed)
-- =============================================================================

-- 1a. List the schemas in the prod catalog (sanity check names).
SHOW SCHEMAS IN prod;

-- 1b. List tables in the Transport.net ordering schema.
SHOW TABLES IN prod.silver_transbroker;

-- 1c. List tables in the docs schema.
SHOW TABLES IN prod.silver_transbrokerdocs;

-- 1d. Find TABLES whose names suggest orders / clinical / PCS / survey content.
SELECT table_schema, table_name
FROM prod.information_schema.tables
WHERE table_schema IN ('silver_transbroker', 'silver_transbrokerdocs')
  AND lower(table_name) RLIKE
      '(order|concierge|clinic|pcs|medical|necess|survey|question|trip|request|narrativ)'
ORDER BY table_schema, table_name;

-- 1e. THE IMPORTANT ONE: find COLUMNS that likely hold the clinical free-text,
--     the giant JSON, requested level of service, and market/area.
SELECT table_schema, table_name, column_name, data_type
FROM prod.information_schema.columns
WHERE table_schema IN ('silver_transbroker', 'silver_transbrokerdocs')
  AND lower(column_name) RLIKE
      '(pcs|clinic|medical|necess|narrativ|blurb|freetext|free_text|comment|note|reason|diagnos|oxygen|ambulan|wheelchair|stretcher|los|levelofservice|servicetype|special|position|mobility|json|survey|question|emergen)'
ORDER BY table_schema, table_name, column_name;


-- =============================================================================
-- SECTION 2 - PROFILE A CANDIDATE TABLE
-- After Section 1 tells you the real order/clinical table name, paste it below.
-- =============================================================================

-- 2a. Full column list for a candidate table (replace the table name).
DESCRIBE TABLE prod.silver_transbroker.o_concierge_trips;   -- TODO: replace with the real table

-- 2b. Peek at a few rows (replace the table name). Small LIMIT = cheap + safe.
SELECT *
FROM prod.silver_transbroker.o_concierge_trips              -- TODO: replace with the real table
LIMIT 20;

-- 2c. Row count + date span so you know how much data exists (replace names).
SELECT
    count(*)                          AS row_count,
    min(RequestDateTime)              AS earliest,          -- TODO: real timestamp column
    max(RequestDateTime)              AS latest             -- TODO: real timestamp column
FROM prod.silver_transbroker.o_concierge_trips;            -- TODO: replace with the real table


-- =============================================================================
-- SECTION 3 - EXTRACTION SKELETON for the medical-necessity dataset
-- Fill the TODOs using the real names from Sections 1-2, then run.
-- This produces exactly the columns the POV scripts expect:
--   order_id, market, requested_los, emergent, free_text (blurb), clinical_json
-- =============================================================================

SELECT
    tc.TripId                       AS order_id,          -- TODO: primary key of the order
    tc.RequestDateTime              AS requested_at,      -- TODO: order timestamp
    an1.AreaName                    AS market,            -- TODO: origin market/area name
    tc.RequestedServiceType         AS requested_los,     -- TODO: requested level of service
    coalesce(tc.IsEmergent, false)  AS emergent,          -- TODO: emergent flag (scope = non-emergent)
    tc.ClinicalNarrative            AS free_text,         -- TODO: the "blurb" narrative column
    tc.ClinicalSurveyJson           AS clinical_json      -- TODO: the giant JSON column
FROM prod.silver_transbroker.o_concierge_trips tc          -- TODO: the order/clinical table
LEFT JOIN prod.silver_transbroker.o_concierge_areas an1
       ON tc.OriginAreaId = an1.AreaId                     -- TODO: confirm area join keys
WHERE coalesce(tc.IsEmergent, false) = false               -- non-emergent IFT only
  AND tc.RequestDateTime >= dateadd(month, -3, current_date())   -- recent slice; adjust
  -- AND an1.AreaName IN ('MUSC', 'Texas Health')          -- TODO: pilot-market filter
LIMIT 200;


-- =============================================================================
-- SECTION 4 - IMMEDIATE WIN (runs against a table confirmed in your editor)
-- tripridesharereceipts exists in silver_transbrokerdocs; use it to validate
-- access + syntax right now while you resolve the clinical table names above.
-- =============================================================================

SELECT
    CadId,
    TripCostDollars,
    TripSurcharge,
    TripDurationSeconds,
    TripDistance,
    EventDate
FROM prod.silver_transbrokerdocs.tripridesharereceipts
ORDER BY EventDate DESC
LIMIT 50;


-- =============================================================================
-- SECTION 5 - OPTIONAL: level of service from the operational DW
-- FactCall / Dim* were visible in your "Create Gold Tables" notebook. If the
-- ordering side does not carry a clean LOS, the operational DW may. Confirm
-- column names first with:
--   DESCRIBE TABLE prod.silver_operationaldw.FactCall;
-- =============================================================================

-- SELECT fc.CallID, fc.RequestDateID, fc.RequestingUnitID, fc.TransportingUnitID
-- FROM prod.silver_operationaldw.FactCall fc
-- LIMIT 20;
