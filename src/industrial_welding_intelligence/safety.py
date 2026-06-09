"""Safety boundaries, mandatory disclaimers, and pilot-scope definitions."""

DISCLAIMERS: dict = {
    "synthetic_data": (
        "SYNTHETIC DATA — All signals, metrics, and values shown in this demo are "
        "generated from mathematical models. They do not represent real welding data, "
        "real machines, or real production processes."
    ),
    "concept_demo": (
        "CONCEPT DEMO — This application is a product concept demonstration only. "
        "It has not been validated against approved production data and is not intended "
        "for operational, quality, or safety decision-making."
    ),
    "quality": (
        "NO CERTIFIED WELD QUALITY — This system does not certify, assure, or replace "
        "qualified weld inspection. Weld quality must be determined by qualified "
        "inspectors following applicable standards (e.g. ISO 3834, AWS D1.1)."
    ),
    "wps": (
        "NO WPS GENERATION — This system does not generate, validate, or replace "
        "Welding Procedure Specifications (WPS). All welding must be performed "
        "according to approved and qualified WPS documents."
    ),
    "inspection": (
        "NO INSPECTION REPLACEMENT — This system does not replace visual inspection, "
        "non-destructive testing (NDT), or welding supervision by qualified personnel. "
        "All statutory inspection and supervision requirements remain in effect."
    ),
    "pilot_estimates": (
        "PILOT ESTIMATES ONLY — Cost, waste, and efficiency figures are synthetic "
        "approximations for concept illustration. Actual values require validated "
        "production data and site-specific measurement."
    ),
}

PILOT_SCOPE: dict = {
    "machine_scope": "One machine model per pilot",
    "process_scope": "One welding process per pilot (e.g. GMAW/MIG)",
    "data_requirement": "Approved test data required before any operational use",
    "deviation_types": "2–3 measurable deviation types per pilot phase",
    "validation_required": True,
    "demo_phase_only": True,
}

PILOT_KPIS: list = [
    "Scrap / reject rate reduction",
    "Inspection load reduction",
    "Alert precision (true-positive rate)",
    "False alert rate",
    "Arc time ratio improvement",
]


def get_all_disclaimers() -> dict:
    """Return a copy of all mandatory disclaimers."""
    return DISCLAIMERS.copy()


def get_pilot_scope() -> dict:
    """Return pilot-scope constraints."""
    return PILOT_SCOPE.copy()


def get_pilot_kpis() -> list:
    """Return the list of measurable pilot KPIs."""
    return list(PILOT_KPIS)
