-- carcinofusion PostgreSQL schema
--
-- Five tables. The separation exists so that any model prediction can be
-- traced back to its chemical structure, the experimental evidence behind its
-- label, and the curation decisions that produced that label.
--
--   sources         where a record came from, and when it was retrieved
--   chemicals       standardised molecular identities
--   experiments     experimental metadata, one row per experimental record
--   observations    experimental outcomes and harmonised labels
--   model_datasets  versioned model-ready labels and split assignments
--
-- Apply with:  psql -d carcinofusion -f sql/schema.sql

BEGIN;

CREATE TABLE IF NOT EXISTS sources (
    source_id               TEXT PRIMARY KEY,
    source_database         TEXT NOT NULL,
    source_record_id        TEXT,
    source_url_or_reference TEXT,
    retrieval_date          TIMESTAMPTZ NOT NULL,
    source_version          TEXT,
    license                 TEXT,
    file_sha256             TEXT,
    notes                   TEXT
);

COMMENT ON TABLE sources IS
    'Provenance. Every experimental record traces to a row here.';


CREATE TABLE IF NOT EXISTS chemicals (
    compound_id                 TEXT PRIMARY KEY,
    canonical_smiles            TEXT NOT NULL,
    isomeric_smiles             TEXT,
    inchi_key                   TEXT NOT NULL,
    -- First block of the InChIKey: connectivity skeleton shared by all
    -- stereoisomers. A grouping key, never an exact identity.
    connectivity_key            TEXT,
    cas_number                  TEXT,
    chemical_name               TEXT,
    molecular_formula           TEXT,
    molecular_weight            DOUBLE PRECISION,
    scaffold_smiles             TEXT,
    scaffold_group              TEXT,
    n_stereocenters             INTEGER,
    n_unassigned_stereocenters  INTEGER,
    n_fragments                 INTEGER,
    structure_status            TEXT NOT NULL,
    standardization_version     TEXT NOT NULL,
    input_smiles                TEXT,
    resolution_source           TEXT,
    resolution_retrieved_date   TEXT,
    standardization_notes       TEXT,
    CONSTRAINT chemicals_inchi_key_unique UNIQUE (inchi_key)
);

COMMENT ON COLUMN chemicals.inchi_key IS
    'Full InChIKey. The second block carries the stereochemistry layer, so
     stereoisomers are distinct compounds.';
COMMENT ON COLUMN chemicals.scaffold_group IS
    'Grouping key for leakage-resistant splits. Acyclic compounds, which have
     no Murcko scaffold, form singleton groups.';

CREATE INDEX IF NOT EXISTS chemicals_scaffold_group_idx ON chemicals (scaffold_group);
CREATE INDEX IF NOT EXISTS chemicals_connectivity_idx   ON chemicals (connectivity_key);
CREATE INDEX IF NOT EXISTS chemicals_cas_idx            ON chemicals (cas_number);


CREATE TABLE IF NOT EXISTS experiments (
    experiment_id     TEXT PRIMARY KEY,
    compound_id       TEXT REFERENCES chemicals (compound_id) ON DELETE RESTRICT,
    source_id         TEXT REFERENCES sources (source_id)     ON DELETE RESTRICT,
    source_record_id  TEXT,
    chemcode          TEXT,
    species           TEXT,
    species_raw       TEXT,
    sex               TEXT,
    sex_raw           TEXT,
    strain            TEXT,
    exposure_route    TEXT,
    study_duration    TEXT,
    dose_duration     TEXT,
    study_reference   TEXT,
    notes             TEXT
);

COMMENT ON COLUMN experiments.species_raw IS
    'The value exactly as the source recorded it. Preserved so that a
     conflicting result can be audited after standardisation.';
COMMENT ON COLUMN experiments.compound_id IS
    'Nullable: an experimental record whose structure could not be resolved
     keeps its row and is held out of the model-ready dataset.';

CREATE INDEX IF NOT EXISTS experiments_compound_idx ON experiments (compound_id);
CREATE INDEX IF NOT EXISTS experiments_chemcode_idx ON experiments (chemcode);
CREATE INDEX IF NOT EXISTS experiments_species_idx  ON experiments (species);


CREATE TABLE IF NOT EXISTS observations (
    observation_id       BIGSERIAL PRIMARY KEY,
    experiment_id        TEXT NOT NULL REFERENCES experiments (experiment_id) ON DELETE CASCADE,
    target_organ         TEXT,
    tumor_type           TEXT,
    author_assessment    TEXT,
    author_assessment_raw TEXT,
    endpoint_name        TEXT NOT NULL,
    -- 1 positive, 0 negative, NULL unresolved. NULL is not negative.
    harmonized_label     SMALLINT,
    dose_response        TEXT,
    tumor_incidence      TEXT,
    td50                 DOUBLE PRECISION,
    label_policy_version TEXT NOT NULL,
    curation_status      TEXT NOT NULL DEFAULT 'parsed',
    CONSTRAINT observations_label_values CHECK (harmonized_label IN (0, 1)),
    CONSTRAINT observations_endpoint_unique UNIQUE (experiment_id, endpoint_name)
);

COMMENT ON COLUMN observations.harmonized_label IS
    '1 positive, 0 negative, NULL unresolved (equivocal, insufficient or
     conflicting). NULL never means negative. The CHECK constraint permits
     only 0 and 1 when a value is present; NULL is always allowed.';
COMMENT ON COLUMN observations.label_policy_version IS
    'Version of configs/labels.yaml that produced this label. A policy change
     requires a version bump and a new dataset version.';

CREATE INDEX IF NOT EXISTS observations_experiment_idx ON observations (experiment_id);
CREATE INDEX IF NOT EXISTS observations_endpoint_idx   ON observations (endpoint_name);
CREATE INDEX IF NOT EXISTS observations_label_idx      ON observations (harmonized_label);


CREATE TABLE IF NOT EXISTS model_datasets (
    dataset_version     TEXT NOT NULL,
    compound_id         TEXT NOT NULL REFERENCES chemicals (compound_id) ON DELETE RESTRICT,
    endpoint_definition TEXT NOT NULL,
    endpoint_name       TEXT NOT NULL,
    label               SMALLINT,
    label_confidence    TEXT,
    is_conflicted       BOOLEAN NOT NULL DEFAULT FALSE,
    n_experiments       INTEGER,
    n_positive          INTEGER,
    n_negative          INTEGER,
    n_indeterminate     INTEGER,
    split_group         TEXT,
    split_strategy      TEXT NOT NULL,
    split_assignment    TEXT NOT NULL,
    label_policy_version TEXT NOT NULL,
    split_version       TEXT NOT NULL,
    frozen_at           TIMESTAMPTZ,
    PRIMARY KEY (dataset_version, endpoint_name, split_strategy, compound_id),
    CONSTRAINT model_datasets_label_values CHECK (label IN (0, 1)),
    CONSTRAINT model_datasets_assignment_values
        CHECK (split_assignment IN ('train', 'validation', 'test', 'external', 'excluded'))
);

COMMENT ON TABLE model_datasets IS
    'Versioned, model-ready labels and split assignments. Immutable once
     frozen: a correction produces a new dataset_version with a change log.';

CREATE INDEX IF NOT EXISTS model_datasets_lookup_idx
    ON model_datasets (dataset_version, endpoint_name, split_strategy, split_assignment);


-- Convenience view: the primary training set, with conflicted and unresolved
-- compounds already excluded.
CREATE OR REPLACE VIEW v_training_set AS
SELECT
    md.dataset_version,
    md.endpoint_name,
    md.split_strategy,
    md.split_assignment,
    md.label,
    md.label_confidence,
    c.compound_id,
    c.canonical_smiles,
    c.isomeric_smiles,
    c.inchi_key,
    c.scaffold_group
FROM model_datasets md
JOIN chemicals c USING (compound_id)
WHERE md.label IS NOT NULL
  AND md.is_conflicted = FALSE
  AND md.split_assignment <> 'excluded';

COMMIT;
