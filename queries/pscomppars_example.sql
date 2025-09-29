-- NASA Exoplanet Archive TAP Query Examples
-- Documentation: https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html
-- Table Schema: https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html

-- ============================================================================
-- Query 1: Get all confirmed exoplanets discovered by transit method
-- ============================================================================
SELECT
    pl_name,           -- Planet name
    hostname,          -- Host star name
    pl_rade,          -- Planet radius (Earth radii)
    pl_masse,         -- Planet mass (Earth masses)
    pl_orbper,        -- Orbital period (days)
    pl_orbsmax,       -- Semi-major axis (AU)
    pl_eqt,           -- Equilibrium temperature (K)
    st_teff,          -- Stellar effective temperature (K)
    st_rad,           -- Stellar radius (Solar radii)
    discoverymethod,  -- Discovery method
    disc_year,        -- Discovery year
    disc_facility     -- Discovery facility
FROM pscomppars
WHERE
    discoverymethod = 'Transit'
    AND pl_rade IS NOT NULL
    AND pl_orbper IS NOT NULL
    AND default_flag = 1  -- Use only default parameter set
ORDER BY disc_year DESC
LIMIT 1000;

-- ============================================================================
-- Query 2: Find Earth-like planets (habitable zone candidates)
-- ============================================================================
SELECT
    pl_name,
    hostname,
    pl_rade,
    pl_orbper,
    pl_eqt,
    st_teff,
    pl_insol,         -- Insolation flux (Earth flux)
    disc_facility
FROM pscomppars
WHERE
    pl_rade BETWEEN 0.8 AND 1.5     -- Earth-sized (0.8-1.5 Earth radii)
    AND pl_eqt BETWEEN 200 AND 320  -- Temperature range for liquid water
    AND discoverymethod = 'Transit'
    AND default_flag = 1
ORDER BY pl_rade;

-- ============================================================================
-- Query 3: Get all TESS discoveries with full parameters
-- ============================================================================
SELECT
    pl_name,
    tic_id,           -- TESS Input Catalog ID
    toi,              -- TESS Object of Interest number
    pl_orbper,
    pl_rade,
    pl_masse,
    pl_dens,          -- Planet density (g/cm^3)
    pl_trandep,       -- Transit depth (%)
    pl_trandur,       -- Transit duration (hours)
    pl_ratror,        -- Planet-star radius ratio
    st_tmag,          -- TESS magnitude
    ra,               -- Right ascension (degrees)
    dec,              -- Declination (degrees)
    sy_pnum,          -- Number of planets in system
    disc_year
FROM pscomppars
WHERE
    disc_facility = 'Transiting Exoplanet Survey Satellite (TESS)'
    AND default_flag = 1
ORDER BY disc_year DESC, toi;

-- ============================================================================
-- Query 4: Multi-planet systems
-- ============================================================================
SELECT
    hostname,
    COUNT(*) as n_planets,
    MIN(pl_orbper) as shortest_period,
    MAX(pl_orbper) as longest_period,
    AVG(pl_rade) as avg_radius,
    st_teff,
    st_rad
FROM pscomppars
WHERE
    discoverymethod = 'Transit'
    AND default_flag = 1
GROUP BY hostname
HAVING COUNT(*) >= 3  -- Systems with 3+ planets
ORDER BY n_planets DESC;

-- ============================================================================
-- Query 5: Recent discoveries (last 2 years)
-- ============================================================================
SELECT
    pl_name,
    hostname,
    pl_orbper,
    pl_rade,
    pl_masse,
    disc_facility,
    disc_year,
    disc_refname     -- Discovery reference
FROM pscomppars
WHERE
    disc_year >= 2023
    AND discoverymethod = 'Transit'
    AND default_flag = 1
ORDER BY disc_year DESC, pl_name;

-- ============================================================================
-- Query 6: Join with stellar properties for machine learning features
-- ============================================================================
SELECT
    p.pl_name,
    p.pl_orbper,
    p.pl_rade,
    p.pl_masse,
    p.pl_dens,
    p.pl_trandep,
    p.pl_trandur,
    p.pl_eqt,
    p.st_teff,        -- Stellar temperature
    p.st_rad,         -- Stellar radius
    p.st_mass,        -- Stellar mass
    p.st_logg,        -- Stellar surface gravity
    p.st_met,         -- Stellar metallicity
    p.st_age,         -- Stellar age
    p.sy_pnum,        -- Number of planets
    p.sy_snum         -- Number of stars
FROM pscomppars p
WHERE
    p.discoverymethod = 'Transit'
    AND p.pl_rade IS NOT NULL
    AND p.st_teff IS NOT NULL
    AND p.default_flag = 1
ORDER BY p.pl_name;

-- ============================================================================
-- NOTES:
-- 1. Always use default_flag = 1 to get the preferred parameter set
-- 2. Use FORMAT='csv' or FORMAT='json' when executing via API
-- 3. Maximum row limit is typically 50000 for synchronous queries
-- 4. For large queries, use asynchronous TAP service
-- 5. Column descriptions: https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html
-- ============================================================================