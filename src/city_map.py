"""Canonical city names, aliases, and coarse region tags."""

from __future__ import annotations

CANONICAL_TO_REGION = {
    "Agartala": "Northeast",
    "Agra": "IGP",
    "Ahmedabad": "West",
    "Aizawl": "Northeast",
    "Amritsar": "IGP",
    "Bengaluru": "South",
    "Bhopal": "Central",
    "Bhubaneswar": "East",
    "Chandigarh": "IGP",
    "Chennai": "South",
    "Coimbatore": "South",
    "Dehradun": "IGP",
    "Delhi": "IGP",
    "Faridabad": "IGP",
    "Gandhinagar": "West",
    "Ghaziabad": "IGP",
    "Guwahati": "Northeast",
    "Gwalior": "Central",
    "Hyderabad": "South",
    "Imphal": "Northeast",
    "Indore": "Central",
    "Itanagar": "Northeast",
    "Jaipur": "Northwest",
    "Jodhpur": "Northwest",
    "Kanpur": "IGP",
    "Kochi": "South",
    "Kohima": "Northeast",
    "Kolkata": "East",
    "Lucknow": "IGP",
    "Ludhiana": "IGP",
    "Mumbai": "West",
    "Nagpur": "Central",
    "Noida": "IGP",
    "Panaji": "West",
    "Patna": "IGP",
    "Pune": "West",
    "Raipur": "Central",
    "Ranchi": "East",
    "Shillong": "Northeast",
    "Shimla": "Himalaya",
    "Srinagar": "Himalaya",
    "Surat": "West",
    "Thiruvananthapuram": "South",
    "Varanasi": "IGP",
    "Vijayawada": "South",
    "Visakhapatnam": "South",
}

# Maps any alias (already lowercased, stripped) -> canonical name.
_ALIASES = {
    "bangalore": "Bengaluru",
    "bengalooru": "Bengaluru",
    "bengaluru": "Bengaluru",
    "bombay": "Mumbai",
    "mumbai": "Mumbai",
    "calcutta": "Kolkata",
    "kolkata": "Kolkata",
    "madras": "Chennai",
    "chennai": "Chennai",
    "delhi": "Delhi",
    "new delhi": "Delhi",
    "newdelhi": "Delhi",
    "nct of delhi": "Delhi",
    "nct delhi": "Delhi",
    "delhi ncr": "Delhi",
    "gurgaon": "Gurugram",
    "gurugram": "Gurugram",
    "trivandrum": "Thiruvananthapuram",
    "thiruvananthapuram": "Thiruvananthapuram",
    "pondicherry": "Puducherry",
    "puducherry": "Puducherry",
    "baroda": "Vadodara",
    "vadodara": "Vadodara",
    "benares": "Varanasi",
    "varanasi": "Varanasi",
    "vizag": "Visakhapatnam",
    "visakhapatnam": "Visakhapatnam",
    "waltair": "Visakhapatnam",
    "cochin": "Kochi",
    "kochi": "Kochi",
    "ernakulam": "Kochi",
    "allahabad": "Prayagraj",
    "prayagraj": "Prayagraj",
    "cawnpore": "Kanpur",
    "kanpur": "Kanpur",
    "jabalpur": "Jabalpur",
    "indore": "Indore",
    "jaipur": "Jaipur",
    "lucknow": "Lucknow",
    "patna": "Patna",
    "hyderabad": "Hyderabad",
    "secunderabad": "Hyderabad",
    "ahmedabad": "Ahmedabad",
    "amdavad": "Ahmedabad",
    "pune": "Pune",
    "poona": "Pune",
    "bhopal": "Bhopal",
    "chandigarh": "Chandigarh",
    "amritsar": "Amritsar",
    "ludhiana": "Ludhiana",
    "faridabad": "Faridabad",
    "ghaziabad": "Ghaziabad",
    "noida": "Noida",
    "greater noida": "Noida",
    "agra": "Agra",
    "gwalior": "Gwalior",
    "nagpur": "Nagpur",
    "raipur": "Raipur",
    "ranchi": "Ranchi",
    "bhubaneswar": "Bhubaneswar",
    "bhubaneshwar": "Bhubaneswar",
    "guwahati": "Guwahati",
    "gauhati": "Guwahati",
    "shillong": "Shillong",
    "imphal": "Imphal",
    "aizawl": "Aizawl",
    "aizwal": "Aizawl",
    "kohima": "Kohima",
    "itanagar": "Itanagar",
    "agartala": "Agartala",
    "panaji": "Panaji",
    "panjim": "Panaji",
    "goa": "Panaji",
    "shimla": "Shimla",
    "srinagar": "Srinagar",
    "dehradun": "Dehradun",
    "dehra dun": "Dehradun",
    "surat": "Surat",
    "jodhpur": "Jodhpur",
    "gandhinagar": "Gandhinagar",
    "coimbatore": "Coimbatore",
    "vijayawada": "Vijayawada",
    "bezwada": "Vijayawada",
}

# Auto-add canonical names as identity aliases.
for _name in list(CANONICAL_TO_REGION):
    _ALIASES[_name.lower()] = _name
_ALIASES["gurugram"] = "Gurugram"
CANONICAL_TO_REGION.setdefault("Gurugram", "IGP")
CANONICAL_TO_REGION.setdefault("Puducherry", "South")
CANONICAL_TO_REGION.setdefault("Vadodara", "West")
CANONICAL_TO_REGION.setdefault("Prayagraj", "IGP")
CANONICAL_TO_REGION.setdefault("Jabalpur", "Central")


def normalize_city(raw: object) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    key = " ".join(text.lower().replace("-", " ").replace("_", " ").split())
    if key in _ALIASES:
        return _ALIASES[key]
    # Title-case unknown cities so joins still work within a dataset.
    return text.title()


def region_for(city: str | None) -> str:
    if not city:
        return "Unknown"
    return CANONICAL_TO_REGION.get(city, "Other")
