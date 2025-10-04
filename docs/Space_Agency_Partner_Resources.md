# Space Agency Partner & Other Data Resources

## Overview

In addition to NASA's primary exoplanet datasets, the Space Apps Challenge references several partner space agencies and complementary data sources that can enhance exoplanet research.

---

## 🇨🇦 Canadian Space Agency (CSA)

### NEOSSat (Near-Earth Object Surveillance Satellite)

**Mission Overview**:
- Dual-purpose microsatellite launched in 2013
- Primary missions: Asteroid detection and space surveillance
- Secondary capability: High-precision photometry for exoplanet studies

**Astronomy Data**:
- **Purpose**: Uninterrupted space-based photometry
- **Advantage**: No atmospheric interference
- **Applications**: Exoplanet transit detection, stellar variability studies

**Data Access**:
```
Primary Portal: Canadian Astronomy Data Centre (CADC)
Website: https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/
```

**NEOSSat Specific Resources**:
```
Mission Homepage: https://www.asc-csa.gc.ca/eng/satellites/neossat/
Data Archive: Available through CADC (registration required)
```

**Data Characteristics**:
- High-cadence photometric observations
- Optical wavelengths
- Complementary to TESS for bright stars
- Suitable for follow-up observations

**Integration with Our Project**:
- **Status**: Optional enhancement
- **Use Case**: Validate TESS/Kepler candidates with independent observations
- **Implementation**: Future phase - cross-mission validation pipeline

---

## 🔭 James Webb Space Telescope (JWST)

### Mission Overview

**Launch**: December 25, 2021
**Primary Focus**: Infrared astronomy and exoplanet characterization
**Capability**: Highest sensitivity for exoplanet atmospheric spectroscopy

**Relevance to Exoplanet Research**:
- Transmission spectroscopy of exoplanet atmospheres
- Direct imaging of exoplanets
- Characterization of planetary systems
- Biosignature detection potential

---

### JWST Data Resources

#### Official Data Archive
```
MAST (Mikulski Archive for Space Telescopes)
Website: https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html
JWST-Specific: https://mast.stsci.edu/search/ui/#/jwst
```

#### Exoplanet-Specific Programs
```
JWST Exoplanet Programs Database
Website: https://www.stsci.edu/jwst/science-execution/approved-programs
Filter: Exoplanet science category
```

**Key JWST Exoplanet Instruments**:

1. **NIRSpec (Near Infrared Spectrograph)**
   - High-resolution spectroscopy
   - Molecular signature detection (H₂O, CO₂, CH₄)
   - Atmospheric composition analysis

2. **NIRISS (Near Infrared Imager and Slitless Spectrograph)**
   - Single Object Slitless Spectroscopy (SOSS) mode
   - Optimized for transiting exoplanet studies
   - Wavelength range: 0.6 - 2.8 μm

3. **NIRCam (Near Infrared Camera)**
   - Transit photometry
   - Direct imaging with coronagraph
   - Wavelength range: 0.6 - 5.0 μm

4. **MIRI (Mid-Infrared Instrument)**
   - Mid-IR spectroscopy (5 - 28 μm)
   - Thermal emission from exoplanets
   - Secondary eclipse observations

---

### JWST Data Access

#### Programmatic Access
```python
# Using astroquery to access JWST data
from astroquery.mast import Observations

# Search for JWST exoplanet observations
obs_table = Observations.query_criteria(
    obs_collection='JWST',
    dataproduct_type='spectrum',
    target_name='TRAPPIST-1'  # Example exoplanet system
)
```

**Direct Download Portal**:
```
https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html
```

**API Documentation**:
```
https://mast.stsci.edu/api/v0/
```

---

### JWST Integration with Our Project

**Current Status**: Reference for future enhancement

**Potential Applications**:

1. **Atmospheric Characterization** (Future Phase 2):
   - Extend AI model to predict atmospheric properties
   - Train on JWST transmission spectra
   - Identify biosignature candidates

2. **Follow-up Prioritization**:
   - AI scores exoplanet candidates by JWST observability
   - Factors: transit depth, stellar brightness, atmospheric scale height
   - Generate ranked list for proposal submissions

3. **Multi-wavelength Analysis**:
   - Combine optical transit (TESS/Kepler) with infrared (JWST)
   - Improved planet radius measurements
   - Atmospheric escape detection

---

## 🌍 European Space Agency (ESA)

### Gaia Mission

**Relevance**: Precise stellar parameters for exoplanet host stars

**Data Access**:
```
Gaia Archive: https://gea.esac.esa.int/archive/
ESA Science Data: https://www.cosmos.esa.int/web/gaia
```

**Use Case**:
- Stellar radius, mass, distance measurements
- Improves exoplanet parameter accuracy
- Essential for planet characterization

---

### CHEOPS (CHaracterising ExOPlanet Satellite)

**Mission Focus**: Precise radius measurements of known exoplanets

**Data Access**:
```
CHEOPS Archive: https://cheops.unige.ch/archive_browser/
ESA CHEOPS Portal: https://www.cosmos.esa.int/web/cheops
```

**Use Case**:
- High-precision transit photometry
- Validate AI detections with independent observations
- Refine planet parameters

---

## 🇯🇵 JAXA (Japan Aerospace Exploration Agency)

### Future Missions

**PLATO (Partnership with ESA)**
- Planned launch: 2026
- Large-scale exoplanet survey
- Will generate massive datasets suitable for AI analysis

---

## Other Complementary Data Sources

### Ground-Based Telescopes

#### 1. Las Cumbres Observatory (LCO)
```
Website: https://lco.global/
Data Archive: https://archive.lco.global/
```
- Global network of robotic telescopes
- Exoplanet transit follow-up
- Open data policy

#### 2. Transiting Exoplanet Survey Network (TESS Follow-up Program)
```
ExoFOP-TESS: https://exofop.ipac.caltech.edu/tess/
```
- Community-contributed follow-up observations
- Ground-based validation of TESS candidates
- Publicly accessible data

---

### Radial Velocity Surveys

#### High Accuracy Radial velocity Planet Searcher (HARPS)
```
ESO Archive: http://archive.eso.org/cms.html
```
- Precise radial velocity measurements
- Complements transit detection
- Mass determination for transiting planets

---

## Data Integration Strategy

### Our Current Implementation

**Primary Data Sources** (Core Project):
✅ NASA Kepler (KOI)
✅ NASA TESS (TOI)
✅ NASA K2 (K2 Candidates)

**Secondary Data Sources** (Optional/Future Enhancements):
🔄 CSA NEOSSat (validation)
🔄 JWST (atmospheric characterization)
🔄 ESA Gaia (stellar parameters)
🔄 Ground-based follow-up (ExoFOP)

---

### Future Expansion Plan

**Phase 1** (Current): NASA primary datasets only
**Phase 2** (3-6 months): Integrate Gaia stellar parameters
**Phase 3** (6-12 months): Add ground-based validation data (ExoFOP)
**Phase 4** (1-2 years): Atmospheric analysis with JWST data
**Phase 5** (2+ years): Multi-mission, multi-wavelength AI system

---

## References and Links Summary

### Canadian Space Agency
- NEOSSat Mission: https://www.asc-csa.gc.ca/eng/satellites/neossat/
- CADC Data Portal: https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/

### James Webb Space Telescope
- MAST Archive: https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html
- JWST Programs: https://www.stsci.edu/jwst/science-execution/approved-programs
- JWST Science: https://www.stsci.edu/jwst/science-execution/scientific-opportunities

### European Space Agency
- Gaia Archive: https://gea.esac.esa.int/archive/
- CHEOPS: https://www.cosmos.esa.int/web/cheops

### Ground-Based
- ExoFOP-TESS: https://exofop.ipac.caltech.edu/tess/
- Las Cumbres Observatory: https://lco.global/

---

## Conclusion

While our current project focuses exclusively on NASA's primary exoplanet datasets (Kepler, TESS, K2), we acknowledge the rich ecosystem of complementary data from partner space agencies. These resources represent exciting opportunities for future enhancements, particularly:

1. **CSA NEOSSat**: Independent validation of candidates
2. **JWST**: Atmospheric characterization and biosignature detection
3. **ESA missions**: Improved stellar and planetary parameters
4. **Ground-based networks**: Community-driven follow-up observations

Our modular architecture is designed to accommodate these additional data sources in future development phases, enabling increasingly sophisticated multi-mission, multi-wavelength exoplanet analysis.

---

*Last Updated: October 2025*
*Project: NASA Space Apps Challenge 2025 - Exoplanet Hunter*
