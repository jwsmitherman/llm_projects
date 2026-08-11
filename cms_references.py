# CMS reference registry for the medical-necessity scoring notebook.
#
# Each entry is keyed by the CMS source it comes from. The scoring notebook imports this module
# and looks references up by key, so citations live in one place and are named from their source.
#
# Import from the notebook (both files in the same Workspace folder, with Files enabled):
#     from cms_references import CMS_REFERENCES, ref_url, ref_label
#
# If import is not available in your environment, run this file first with %run ./cms_references

CMS_REFERENCES = {
    "BPM10": {
        "title": "Medicare Benefit Policy Manual, Pub. 100-02, Ch. 10 - Ambulance Services",
        "url": "https://www.cms.gov/Regulations-and-Guidance/Guidance/Manuals/Downloads/bp102c10.pdf",
        "note": "10.2.1 general medical necessity test; 10.2.3 bed-confined three-prong test.",
    },
    "BPM10_10_2_1": {
        "title": "Benefit Policy Manual Ch. 10, Section 10.2.1 - General medical necessity",
        "url": "https://www.cms.gov/Regulations-and-Guidance/Guidance/Manuals/Downloads/bp102c10.pdf",
        "note": "Ambulance covered only when other transport is contraindicated.",
    },
    "BPM10_10_2_3": {
        "title": "Benefit Policy Manual Ch. 10, Section 10.2.3 - Bed-confined test",
        "url": "https://www.cms.gov/Regulations-and-Guidance/Guidance/Manuals/Downloads/bp102c10.pdf",
        "note": "Bed-confined = unable to get up, unable to ambulate, and unable to sit. All three.",
    },
    "CFR_410_40": {
        "title": "42 CFR 410.40 - Coverage of ambulance services",
        "url": "https://www.ecfr.gov/current/title-42/chapter-IV/subchapter-B/part-410/subpart-B/section-410.40",
        "note": "Levels of service and physician certification.",
    },
    "CFR_414_605": {
        "title": "42 CFR 414.605 - Definitions (BLS, ALS1, ALS2, SCT, ALS assessment/intervention)",
        "url": "https://www.ecfr.gov/current/title-42/chapter-IV/subchapter-B/part-414/subpart-H/section-414.605",
        "note": "Level-of-service definitions used by the monitoring axis.",
    },
    "CPM15": {
        "title": "Medicare Claims Processing Manual, Pub. 100-04, Ch. 15 - Ambulance",
        "url": "https://www.cms.gov/Regulations-and-Guidance/Guidance/Manuals/Downloads/clm104c15ambulance.pdf",
        "note": "Claims adjudication for ambulance.",
    },
    "RSN": {
        "title": "CMS Ambulance Transport Reason Codes and Statements",
        "url": "https://www.cms.gov/files/document/ambulance-transport-reason-codes-statements.pdf",
        "note": "Denial reason codes, e.g. AM600 (documentation does not indicate other transport contraindicated).",
    },
    "MLN": {
        "title": "CMS MLN Ambulance Services - provider compliance tips",
        "url": "https://www.cms.gov/training-education/medicare-learning-networkr-mln/compliance/medicare-provider-compliance-tips/ambulance-services",
        "note": "Documentation guidance; basis for treating vague terms as filler.",
    },
}


def ref_url(key):
    """URL for a reference key. Accepts a section key (BPM10_10_2_3) or the base doc (BPM10)."""
    if key in CMS_REFERENCES:
        return CMS_REFERENCES[key]["url"]
    base = key.split("_")[0]
    return CMS_REFERENCES.get(base, {}).get("url", "")


def ref_label(key):
    """Short human label for a reference key."""
    return CMS_REFERENCES.get(key, {}).get("title", key)
