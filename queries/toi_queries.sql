-- TESS Objects of Interest (TOI) TAP Query Examples
-- Documentation: https://exoplanetarchive.ipac.caltech.edu/docs/API_toi_columns.html
-- Table: toi

-- ============================================================================
-- Query 1: Get all TOI candidates with dispositions
-- ============================================================================
SELECT
    tid,              -- TIC ID
    toi,              -- TOI number
    toipfx,           -- TOI prefix (planet letter)
    tfopwg_disp,      -- TFOPWG disposition (PC/CP/KP/FP/FA)
    pl_orbper,        -- Orbital period (days)
    pl_orbpererr1,    -- Period error upper
    pl_orbpererr2,    -- Period error lower
    pl_rade,          -- Planet radius (Earth radii)
    pl_radeerr1,      -- Radius error upper
    pl_radeerr2,      -- Radius error lower
    pl_bmasse,        -- Planet mass (Earth masses)
    pl_trandep,       -- Transit depth (ppm)
    pl_trandur,       -- Transit duration (hours)
    st_tmag,          -- TESS magnitude
    st_rad,           -- Stellar radius (Solar radii)
    st_teff,          -- Stellar temperature (K)
    ra,               -- Right ascension
    dec               -- Declination
FROM toi
WHERE
    tfopwg_disp IS NOT NULL
ORDER BY toi;

-- ============================================================================
-- Query 2: Get confirmed planets from TOI (PC, CP, KP)
-- ============================================================================
SELECT
    tid,
    toi,
    toipfx,
    tfopwg_disp,
    pl_orbper,
    pl_rade,
    pl_trandep,
    st_tmag,
    ra,
    dec
FROM toi
WHERE
    tfopwg_disp IN ('PC', 'CP', 'KP')  -- PC=Planet Candidate, CP=Confirmed Planet, KP=Known Planet
ORDER BY pl_rade;

-- ============================================================================
-- Query 3: Get false positives for training negative samples
-- ============================================================================
SELECT
    tid,
    toi,
    tfopwg_disp,
    tfopwg_comment,   -- Comment on disposition
    pl_orbper,
    pl_trandep,
    st_tmag,
    ra,
    dec
FROM toi
WHERE
    tfopwg_disp = 'FP'  -- False Positive
ORDER BY toi;

-- ============================================================================
-- Query 4: Small planet candidates (potential Earth-like)
-- ============================================================================
SELECT
    tid,
    toi,
    toipfx,
    tfopwg_disp,
    pl_orbper,
    pl_rade,
    pl_eqt,           -- Equilibrium temperature
    pl_insol,         -- Insolation flux
    st_teff,
    st_rad,
    st_logg,
    ra,
    dec
FROM toi
WHERE
    pl_rade < 2.0                       -- Smaller than 2 Earth radii
    AND pl_orbper BETWEEN 10 AND 500    -- Reasonable orbital period
    AND tfopwg_disp IN ('PC', 'CP')     -- Candidates or confirmed
    AND pl_rade IS NOT NULL
ORDER BY pl_rade;

-- ============================================================================
-- Query 5: Multi-planet TOI systems
-- ============================================================================
SELECT
    tid,
    COUNT(DISTINCT toipfx) as n_planets,
    MIN(pl_orbper) as min_period,
    MAX(pl_orbper) as max_period,
    MIN(pl_rade) as min_radius,
    MAX(pl_rade) as max_radius,
    AVG(pl_trandep) as avg_transit_depth,
    st_tmag,
    ra,
    dec
FROM toi
WHERE
    tfopwg_disp IN ('PC', 'CP', 'KP')
    AND pl_orbper IS NOT NULL
GROUP BY tid
HAVING COUNT(DISTINCT toipfx) >= 2  -- Systems with 2+ planets
ORDER BY n_planets DESC;

-- ============================================================================
-- Query 6: TOI with TESS sector information
-- ============================================================================
SELECT
    t.tid,
    t.toi,
    t.toipfx,
    t.tfopwg_disp,
    t.pl_orbper,
    t.pl_rade,
    t.sectors,        -- TESS sectors observed
    t.toi_created,    -- Date TOI was created
    t.toi_updated,    -- Date TOI was last updated
    t.st_tmag,
    t.ra,
    t.dec
FROM toi t
WHERE
    t.tfopwg_disp IN ('PC', 'CP')
    AND t.sectors IS NOT NULL
ORDER BY t.toi_created DESC
LIMIT 100;

-- ============================================================================
-- Query 7: Join TOI with confirmed planets for validation
-- ============================================================================
SELECT
    t.tid,
    t.toi,
    t.tfopwg_disp as toi_disposition,
    t.pl_orbper as toi_period,
    t.pl_rade as toi_radius,
    p.pl_name,
    p.pl_orbper as confirmed_period,
    p.pl_rade as confirmed_radius,
    ABS(t.pl_orbper - p.pl_orbper) as period_diff,
    ABS(t.pl_rade - p.pl_rade) as radius_diff
FROM toi t
JOIN pscomppars p ON t.tid = p.tic_id
WHERE
    t.tfopwg_disp IN ('CP', 'KP')
    AND p.default_flag = 1
    AND t.pl_orbper IS NOT NULL
    AND p.pl_orbper IS NOT NULL
ORDER BY period_diff;

-- ============================================================================
-- Query 8: Recent TOI updates (last 30 days)
-- ============================================================================
SELECT
    tid,
    toi,
    toipfx,
    tfopwg_disp,
    pl_orbper,
    pl_rade,
    toi_created,
    toi_updated,
    st_tmag
FROM toi
WHERE
    toi_updated >= CURRENT_DATE - INTERVAL '30' DAY
ORDER BY toi_updated DESC;

-- ============================================================================
-- NOTES:
-- TFOPWG Dispositions:
--   PC = Planet Candidate (active candidate)
--   CP = Confirmed Planet (confirmed by follow-up)
--   KP = Known Planet (previously known)
--   FP = False Positive (ruled out as planet)
--   FA = False Alarm (instrumental/stellar variability)
--
-- API Usage:
--   Synchronous: https://exoplanetarchive.ipac.caltech.edu/TAP/sync
--   Asynchronous: https://exoplanetarchive.ipac.caltech.edu/TAP/async
-- ============================================================================