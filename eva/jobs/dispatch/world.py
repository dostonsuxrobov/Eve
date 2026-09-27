"""The fake world of Red Oak Transport, a fictional truckload carrier: a SQLite database big
enough that the dispatcher has to search and join, not recite.

Deterministic for a seed and an hour: ``build(path, now=...)`` makes the same fleet, brokers,
load board and history every time, anchored at ``now`` (the active loads are on the road
*now*). A few rows are planted for the scenarios in bench/dispatch_eval.py (``PLANTED``): a
lowball reefer load out of Dallas, a late load, a do-not-use broker with a rate too good to be
true, a hazmat load whose nearest driver has no endorsement, a truck broken down with a
reefer load due tomorrow, and an empty truck in Memphis looking for a load. Every name,
number and company is invented.

    python -m eva.jobs.dispatch.world            # (re)build data/dispatch.db and print its size
"""
from __future__ import annotations

import math
import os
import random
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from ...config import ROOT

# EVA_DISPATCH_DB: another file, so two evals can run side by side without rebuilding each other's world
DB_PATH = Path(os.environ.get("EVA_DISPATCH_DB") or ROOT / "data" / "dispatch.db")
COMPANY = "Red Oak Transport"
SEED = 7
ROAD_FACTOR = 1.18  # straight-line miles to road miles
AVG_MPH = 50.0  # door to door, with fuel stops and traffic

# (city, state, lat, lon): the freight markets of the lower 48
CITIES: list[tuple[str, str, float, float]] = [
    ("Atlanta", "GA", 33.749, -84.388), ("Savannah", "GA", 32.081, -81.091), ("Macon", "GA", 32.841, -83.632),
    ("Gainesville", "GA", 34.298, -83.824), ("Dallas", "TX", 32.777, -96.797), ("Fort Worth", "TX", 32.755, -97.331),
    ("Houston", "TX", 29.760, -95.370), ("Beaumont", "TX", 30.080, -94.126), ("San Antonio", "TX", 29.424, -98.494),
    ("Austin", "TX", 30.267, -97.743), ("El Paso", "TX", 31.762, -106.485), ("Laredo", "TX", 27.506, -99.507),
    ("McAllen", "TX", 26.203, -98.230), ("Amarillo", "TX", 35.222, -101.831), ("Lubbock", "TX", 33.578, -101.855),
    ("Chicago", "IL", 41.878, -87.630), ("Joliet", "IL", 41.525, -88.082), ("Peoria", "IL", 40.694, -89.589),
    ("Indianapolis", "IN", 39.768, -86.158), ("Fort Wayne", "IN", 41.079, -85.139), ("Richmond", "IN", 39.829, -84.890),
    ("Columbus", "OH", 39.961, -82.999), ("Cleveland", "OH", 41.499, -81.694), ("Cincinnati", "OH", 39.103, -84.512),
    ("Toledo", "OH", 41.654, -83.537), ("Dayton", "OH", 39.759, -84.192), ("Detroit", "MI", 42.331, -83.046),
    ("Grand Rapids", "MI", 42.963, -85.668), ("Lansing", "MI", 42.733, -84.555), ("Milwaukee", "WI", 43.039, -87.906),
    ("Green Bay", "WI", 44.519, -88.020), ("Madison", "WI", 43.073, -89.401), ("Minneapolis", "MN", 44.978, -93.265),
    ("Des Moines", "IA", 41.587, -93.625), ("Cedar Rapids", "IA", 41.977, -91.666), ("Omaha", "NE", 41.257, -95.935),
    ("Kansas City", "MO", 39.100, -94.578), ("St. Louis", "MO", 38.627, -90.199), ("Springfield", "MO", 37.209, -93.292),
    ("Memphis", "TN", 35.149, -90.049), ("Nashville", "TN", 36.163, -86.781), ("Knoxville", "TN", 35.961, -83.921),
    ("Chattanooga", "TN", 35.046, -85.310), ("Louisville", "KY", 38.253, -85.759), ("Lexington", "KY", 38.040, -84.504),
    ("Birmingham", "AL", 33.521, -86.803), ("Montgomery", "AL", 32.367, -86.300), ("Mobile", "AL", 30.695, -88.040),
    ("Jackson", "MS", 32.299, -90.185), ("Southaven", "MS", 34.989, -90.013), ("New Orleans", "LA", 29.951, -90.072),
    ("Baton Rouge", "LA", 30.451, -91.187), ("Shreveport", "LA", 32.525, -93.750), ("Little Rock", "AR", 34.746, -92.290),
    ("Fort Smith", "AR", 35.386, -94.398), ("Oklahoma City", "OK", 35.468, -97.516), ("Tulsa", "OK", 36.154, -95.993),
    ("Wichita", "KS", 37.687, -97.330), ("Garden City", "KS", 37.971, -100.873), ("Dodge City", "KS", 37.753, -100.017),
    ("Denver", "CO", 39.739, -104.990), ("Salt Lake City", "UT", 40.761, -111.891), ("Phoenix", "AZ", 33.448, -112.074),
    ("Tucson", "AZ", 32.222, -110.975), ("Nogales", "AZ", 31.340, -110.934), ("Albuquerque", "NM", 35.084, -106.650),
    ("Las Vegas", "NV", 36.170, -115.140), ("Reno", "NV", 39.530, -119.814), ("Los Angeles", "CA", 34.052, -118.244),
    ("Ontario", "CA", 34.063, -117.651), ("San Diego", "CA", 32.716, -117.161), ("Bakersfield", "CA", 35.373, -119.019),
    ("Fresno", "CA", 36.738, -119.787), ("Salinas", "CA", 36.678, -121.655), ("Stockton", "CA", 37.958, -121.291),
    ("Sacramento", "CA", 38.582, -121.494), ("Oakland", "CA", 37.804, -122.271), ("Portland", "OR", 45.515, -122.679),
    ("Seattle", "WA", 47.606, -122.332), ("Tacoma", "WA", 47.253, -122.444), ("Yakima", "WA", 46.602, -120.505),
    ("Spokane", "WA", 47.659, -117.426), ("Boise", "ID", 43.615, -116.202), ("Billings", "MT", 45.783, -108.501),
    ("Fargo", "ND", 46.877, -96.790), ("Sioux Falls", "SD", 43.546, -96.731), ("Charlotte", "NC", 35.227, -80.843),
    ("Greensboro", "NC", 36.073, -79.792), ("Raleigh", "NC", 35.780, -78.639), ("Charleston", "SC", 32.777, -79.931),
    ("Columbia", "SC", 34.001, -81.035), ("Greenville", "SC", 34.852, -82.394), ("Jacksonville", "FL", 30.332, -81.656),
    ("Orlando", "FL", 28.538, -81.379), ("Tampa", "FL", 27.951, -82.457), ("Lakeland", "FL", 28.039, -81.950),
    ("Miami", "FL", 25.762, -80.192), ("Tallahassee", "FL", 30.438, -84.281), ("Richmond", "VA", 37.541, -77.436),
    ("Norfolk", "VA", 36.851, -76.286), ("Roanoke", "VA", 37.271, -79.941), ("Baltimore", "MD", 39.290, -76.612),
    ("Harrisburg", "PA", 40.273, -76.887), ("Allentown", "PA", 40.608, -75.490), ("Philadelphia", "PA", 39.953, -75.165),
    ("Pittsburgh", "PA", 40.441, -79.996), ("Scranton", "PA", 41.409, -75.662), ("Newark", "NJ", 40.736, -74.172),
    ("Edison", "NJ", 40.519, -74.412), ("Albany", "NY", 42.652, -73.756), ("Buffalo", "NY", 42.886, -78.878),
    ("Syracuse", "NY", 43.049, -76.147), ("Hartford", "CT", 41.764, -72.685), ("Boston", "MA", 42.360, -71.058),
    ("Worcester", "MA", 42.262, -71.802), ("Portland", "ME", 43.659, -70.257),
]
HUBS = {"Atlanta", "Dallas", "Houston", "Chicago", "Los Angeles", "Ontario", "Memphis", "Columbus", "Indianapolis",
        "Kansas City", "Harrisburg", "Charlotte", "Nashville", "Jacksonville", "Laredo", "Phoenix", "Newark", "St. Louis"}

REGIONS = {
    "Northeast": "ME NH VT MA RI CT NY NJ", "Mid-Atlantic": "PA DE MD VA WV", "Southeast": "NC SC GA",
    "Florida": "FL", "Mid-South": "TN KY AL MS AR", "Gulf": "LA", "South Central": "TX OK",
    "Great Lakes": "OH MI IN", "Upper Midwest": "IL WI MN IA", "Plains": "KS NE MO ND SD",
    "Mountain": "CO UT ID MT NM NV", "Southwest": "AZ", "California": "CA", "Pacific NW": "OR WA",
}
STATE_REGION = {s: r for r, states in REGIONS.items() for s in states.split()}
# Outbound capacity: markets that ship more than they receive pay more to leave.
OUT_F = {"California": 1.12, "Pacific NW": 1.00, "Southwest": 0.90, "Mountain": 0.86, "South Central": 0.98,
         "Plains": 1.03, "Upper Midwest": 1.06, "Great Lakes": 1.03, "Northeast": 0.88, "Mid-Atlantic": 1.00,
         "Southeast": 1.04, "Florida": 0.78, "Mid-South": 1.05, "Gulf": 1.00}
IN_F = {"Florida": 1.12, "Northeast": 1.08, "Southwest": 1.03, "Mountain": 1.05, "California": 0.95,
        "Pacific NW": 1.02, "South Central": 0.99, "Plains": 1.00, "Upper Midwest": 0.98, "Great Lakes": 0.99,
        "Mid-Atlantic": 1.02, "Southeast": 1.00, "Mid-South": 0.98, "Gulf": 1.01}
EQUIP_BASE = {"dry_van": 2.30, "reefer": 2.70, "flatbed": 2.95}
COMMODITIES = {
    "dry_van": ["paper products", "canned goods", "auto parts", "beverages", "household goods", "retail freight",
                "packaging material", "building supplies", "electronics", "pet food", "plastic resin"],
    "reefer": ["produce", "frozen chicken", "dairy", "frozen vegetables", "beef", "ice cream", "fresh berries",
               "pharmaceuticals", "juice", "frozen pizza"],
    "flatbed": ["steel coils", "lumber", "rebar", "machinery", "pipe", "roofing shingles", "precast concrete",
                "bricks", "aluminum extrusions"],
}
REEFER_TEMP = {"produce": 34, "frozen chicken": 0, "dairy": 36, "frozen vegetables": -10, "beef": 28,
               "ice cream": -20, "fresh berries": 33, "pharmaceuticals": 40, "juice": 34, "frozen pizza": -5}

FIRST = ("James Maria Luis Tasha Kevin Rosa Marcus Denise Andre Carlos Linda Terrence Brandon Keisha Tyrone "
         "Maribel Dwayne Crystal Hector Shawn Tamika Rodney Yolanda Travis Ricardo Latoya Gary Veronica Omar "
         "Tonya Derek Jasmine Eddie Monica Reggie Angela Bobby Carmen Wesley Nadia Curtis Brenda Jamal Irene "
         "Victor Sonia Floyd Renee Samir Dana Igor Aziz Dmitri Olga Sergei Tomas Ana Pedro Kofi Amara").split()
LAST = ("Ortega Brooks Park Jimenez Hill Carter Reyes Coleman Nguyen Patel Washington Grant Foster Morales "
        "Bennett Simmons Ramirez Hayes Price Sanders Jenkins Perry Russell Diaz Butler Barnes Fisher Henderson "
        "Ward Torres Peterson Gray Watson Kim Rivera Cooper Howard Cruz Long Flores Wood Myers Ross Hughes "
        "Kozlov Petrov Novak Haddad Mensah Okafor Silva Costa Yilmaz Rahimov Karimov Tashkentov").split()
BROKER_A = ("Crescent Summit Pioneer Keystone Harbor Prairie Granite Liberty Northwind Cardinal Evergreen Copper "
            "Bluegrass Lone Star Magnolia Frontier Heartland Anchor Beacon Cedar Ridgeline Atlas Meridian Sterling "
            "Canyon Ironwood Redline Trident Horizon Legacy Cornerstone Timberline Coastal Midland Patriot").split(" ")
BROKER_B = ["Freight Partners", "Logistics", "Freight Solutions", "Transport Services", "Brokerage", "Supply Chain",
            "Freight Group", "Shipping", "Logistics Group", "Cargo", "Freight Brokers", "Distribution Services"]
FAC_A = ("Midway Riverside Parkside Gateway Southgate Northpoint Eastfield Westport Lakeside Hillcrest Pinewood "
         "Oakmont Sunbelt Prairie Harbor Valley Summit Central Metro Crossroads").split()
FAC_KIND = [("Distribution Center", "dc"), ("Cold Storage", "cold"), ("Packaging Co.", "plant"), ("Foods", "plant"),
            ("Steel Works", "plant"), ("Retail DC", "dc"), ("Warehouse", "dc"), ("Produce Terminal", "cold"),
            ("Building Supply", "plant"), ("Beverage Plant", "plant"), ("Grocery DC", "dc"), ("Lumber Yard", "plant")]
FAC_NOTES = ["", "", "", "Check in at gate 3; strict on appointments.", "No overnight parking on site.",
             "Lumper required, bring cash or comchek.", "Driver must stay with truck.", "Seal must be verified at door.",
             "First come first served before 10 am.", "Tight dock, back in from the east side.",
             "Late arrivals get worked in only if a door opens.", "Scale on site, bring a scale ticket.",
             "Reefer must be pre-cooled before check-in.", "Hard hats and vests required on the yard."]
DRIVER_NOTES = ["", "", "", "Prefers no-touch freight.", "Won't run New York City.", "Home every weekend in {home}.",
                "Team-capable with notice.", "Good with tight docks.", "Needs 34-hour reset at home.",
                "Speaks Spanish with shippers.", "Chains up in winter, does mountain routes."]


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 3958.8 * 2 * math.asin(math.sqrt(a))


def road_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> int:
    return int(round(haversine(lat1, lon1, lat2, lon2) * ROAD_FACTOR))


def ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M")


def week_of(dt: datetime) -> str:
    return (dt - timedelta(days=dt.weekday())).strftime("%Y-%m-%d")


SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE cities (id INTEGER PRIMARY KEY, name TEXT, state TEXT, lat REAL, lon REAL, region TEXT);
CREATE TABLE facilities (id INTEGER PRIMARY KEY, name TEXT, kind TEXT, city_id INT, address TEXT, opens TEXT,
  closes TEXT, appointment_required INT, avg_dwell_min INT, lumper_fee INT, notes TEXT);
CREATE TABLE brokers (id INTEGER PRIMARY KEY, name TEXT, mc TEXT, credit TEXT, days_to_pay INT, quick_pay_pct REAL,
  style REAL, status TEXT, status_reason TEXT, phone TEXT, email TEXT);
CREATE TABLE broker_contacts (id INTEGER PRIMARY KEY, broker_id INT, name TEXT, desk TEXT, phone TEXT);
CREATE TABLE drivers (id INTEGER PRIMARY KEY, name TEXT, phone TEXT, home_city_id INT, status TEXT, hazmat INT,
  twic INT, rating REAL, drive_left_h REAL, duty_left_h REAL, cycle_left_h REAL, available_at TEXT, notes TEXT);
CREATE TABLE trucks (unit TEXT PRIMARY KEY, year INT, make TEXT, status TEXT, driver_id INT, trailer TEXT,
  city_id INT, lat REAL, lon REAL, next_pm_miles INT, notes TEXT);
CREATE TABLE trailers (number TEXT PRIMARY KEY, type TEXT, length_ft INT, status TEXT, city_id INT,
  last_washout TEXT, reefer_hours INT);
CREATE TABLE postings (id TEXT PRIMARY KEY, broker_id INT, contact_id INT, origin_id INT, dest_id INT,
  pickup_start TEXT, pickup_end TEXT, delivery_start TEXT, delivery_end TEXT, equipment TEXT, weight INT,
  commodity TEXT, temp_f INT, miles INT, posted_rate INT, hazmat INT, team INT, tarps INT, stops INT, notes TEXT,
  status TEXT, posted_at TEXT);
CREATE TABLE loads (number TEXT PRIMARY KEY, broker_id INT, broker_ref TEXT, posting_id TEXT, origin_id INT,
  dest_id INT, pickup_appt TEXT, delivery_appt TEXT, picked_up_at TEXT, delivered_at TEXT, driver_id INT,
  truck TEXT, trailer TEXT, equipment TEXT, commodity TEXT, weight INT, temp_f INT, miles INT, deadhead INT,
  rate INT, detention INT, lumper INT, status TEXT, invoiced_at TEXT, paid_at TEXT, notes TEXT, booked_at TEXT);
CREATE TABLE pings (id INTEGER PRIMARY KEY, load_number TEXT, truck TEXT, at TEXT, lat REAL, lon REAL,
  city_id INT, miles_to_go INT, speed INT, note TEXT);
CREATE TABLE incidents (id INTEGER PRIMARY KEY, truck TEXT, load_number TEXT, at TEXT, kind TEXT, city_id INT,
  detail TEXT, resolved INT);
CREATE TABLE lane_rates (origin_region TEXT, dest_region TEXT, equipment TEXT, week TEXT, rpm REAL,
  loads_per_truck REAL);
CREATE TABLE fuel (region TEXT, week TEXT, diesel REAL);
CREATE TABLE policy (key TEXT PRIMARY KEY, value TEXT, note TEXT);
CREATE TABLE messages (id INTEGER PRIMARY KEY, at TEXT, direction TEXT, party TEXT, load_number TEXT, text TEXT);
CREATE TABLE updates (id INTEGER PRIMARY KEY, at TEXT, load_number TEXT, kind TEXT, text TEXT);
CREATE INDEX postings_origin ON postings(origin_id, status, equipment);
CREATE INDEX postings_broker ON postings(broker_id);
CREATE INDEX loads_ref ON loads(broker_ref);
CREATE INDEX loads_driver ON loads(driver_id, status);
CREATE INDEX loads_lane ON loads(origin_id, dest_id);
CREATE INDEX loads_broker ON loads(broker_id);
CREATE INDEX pings_load ON pings(load_number, at);
CREATE INDEX lane_idx ON lane_rates(origin_region, dest_region, equipment, week);
"""

# The company's rate rules: the floor is hard, the target is where a counter starts.
POLICY = [
    ("cost_per_mile", "1.95", "all-in operating cost per loaded or empty mile: driver 0.65, fuel 0.60, truck and trailer 0.35, maintenance 0.18, insurance 0.12, overhead 0.05"),
    ("min_margin", "250", "dollars over cost on every load, deadhead included"),
    ("min_rpm_dry_van", "2.10", "never below this per loaded mile"),
    ("min_rpm_reefer", "2.45", "never below this per loaded mile"),
    ("min_rpm_flatbed", "2.65", "never below this per loaded mile"),
    ("target_over_market", "0.08", "open a counter about 8 percent over the lane's 30-day market rate"),
    ("max_deadhead", "150", "empty miles to a pickup, more only with the owner's okay"),
    ("detention", "60", "dollars per hour after 2 free hours at a shipper or receiver"),
    ("tonu", "250", "truck ordered not used"),
    ("layover", "300", "per day when a load can't deliver as scheduled through no fault of ours"),
    ("extra_stop", "75", "per stop after the first pickup and first drop"),
    ("lumper", "reimbursed", "with a receipt, added to the invoice"),
    ("max_weight", "44000", "pounds legal on our equipment"),
    ("do_not_use", "never", "never book with a broker flagged do-not-use, whatever the rate"),
    ("hazmat", "endorsed", "hazmat loads only with a hazmat-endorsed driver, placards on the truck"),
    ("food_grade", "7", "reefer loads of food need a trailer washed out within 7 days"),
    ("quick_pay", "optional", "broker quick pay is fine when the fee is 3 percent or less"),
]

# What the scenarios rely on (bench/dispatch_eval.py). Hours are relative to ``now``.
PLANTED = {
    "lowball_posting": "LB-500101", "lowball_broker": "Crescent Freight Partners", "lowball_truck": "214",
    "late_load": "RO-48213", "late_ref": "7781234", "dnu_broker": "Rapid Eagle Brokerage", "dnu_posting": "LB-500201",
    "hazmat_posting": "LB-500301", "hazmat_truck": "188", "hazmat_decoy_truck": "142",
    "memphis_truck": "330", "memphis_best": "LB-500401", "broken_load": "RO-48307", "broken_truck": "118",
    "rescue_truck": "402",
}


def build(path: Path = DB_PATH, *, seed: int = SEED, now: datetime | None = None) -> dict[str, int]:
    """Write a fresh world to ``path``; returns row counts."""
    rng = random.Random(seed)
    now = (now or datetime.now()).replace(second=0, microsecond=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    db = sqlite3.connect(tmp)
    db.executescript(SCHEMA)

    # ------------------------------------------------------------------ geography
    cities = []
    for i, (name, st, lat, lon) in enumerate(CITIES, start=1):
        cities.append((i, name, st, lat, lon, STATE_REGION[st]))
    db.executemany("INSERT INTO cities VALUES (?,?,?,?,?,?)", cities)
    city = {c[0]: c for c in cities}
    by_name = {(c[1], c[2]): c[0] for c in cities}
    weights = [5 if c[1] in HUBS else 1 for c in cities]

    def cid(name: str, st: str) -> int:
        return by_name[(name, st)]

    def dist(a: int, b: int) -> int:
        return road_miles(city[a][3], city[a][4], city[b][3], city[b][4])

    # ------------------------------------------------------------------ facilities
    facilities = []
    fac_by_city: dict[int, list[int]] = {}
    fid = 0
    for c in cities:
        for _ in range(rng.randint(6, 12) if c[1] in HUBS else rng.randint(4, 8)):
            fid += 1
            kind_name, kind = rng.choice(FAC_KIND)
            name = f"{rng.choice(FAC_A)} {kind_name}" if rng.random() < 0.6 else f"{c[1]} {kind_name}"
            opens = rng.choice(["00:00", "05:00", "06:00", "07:00", "08:00"])
            closes = "23:59" if opens == "00:00" else rng.choice(["14:00", "15:00", "16:00", "18:00", "22:00"])
            facilities.append((fid, name, kind, c[0], f"{rng.randint(100, 9900)} {rng.choice(FAC_A)} {rng.choice(['Rd', 'Blvd', 'Pkwy', 'Dr', 'Ave'])}",
                               opens, closes, int(rng.random() < 0.7), rng.choice([45, 60, 90, 120, 150, 180]),
                               rng.choice([0, 0, 0, 150, 200, 275]) if kind != "plant" else 0, rng.choice(FAC_NOTES)))
            fac_by_city.setdefault(c[0], []).append(fid)

    def add_facility(name: str, kind: str, city_id: int, opens: str, closes: str, dwell: int, lumper: int, notes: str) -> int:
        nonlocal fid
        fid += 1
        facilities.append((fid, name, kind, city_id, f"{rng.randint(100, 9900)} Industrial Blvd", opens, closes, 1, dwell, lumper, notes))
        fac_by_city.setdefault(city_id, []).append(fid)
        return fid

    # ------------------------------------------------------------------ brokers
    brokers = []
    contacts = []
    names_used: set[str] = set()
    for bid in range(1, 221):
        while True:
            name = f"{rng.choice(BROKER_A)} {rng.choice(BROKER_B)}"
            if name not in names_used:
                names_used.add(name)
                break
        credit = rng.choices(["A", "B", "C", "D"], [45, 35, 15, 5])[0]
        dtp = {"A": rng.choice([15, 21, 30]), "B": rng.choice([30, 30, 35]), "C": rng.choice([35, 45]), "D": rng.choice([45, 60])}[credit]
        style = round(rng.uniform(0.84, 1.03), 3)  # how their offers sit against the market
        status, reason = "active", ""
        if rng.random() < 0.05:
            status, reason = "do_not_use", rng.choice(["Paid 70+ days late twice.", "Held a load hostage over a rate dispute.",
                                                       "Double-brokered one of our loads.", "Cancelled after dispatch without TONU."])
        brokers.append([bid, name, f"MC{rng.randint(100000, 999999)}", credit, dtp, rng.choice([0, 1.5, 2, 2.5, 3, 4, 5]),
                        style, status, reason, f"({rng.randint(201, 989)}) 555-{rng.randint(1000, 9999)}",
                        f"loads@{name.lower().replace(' ', '').replace('.', '')}.example"])
        for _ in range(rng.randint(1, 3)):
            contacts.append((len(contacts) + 1, bid, f"{rng.choice(FIRST)} {rng.choice(LAST)}",
                             rng.choice(["dry van desk", "reefer desk", "flatbed desk", "Southeast desk", "Midwest desk",
                                         "West Coast desk", "Texas desk", "after hours"]), f"ext {rng.randint(100, 499)}"))

    def plant_broker(bid: int, name: str, credit: str, dtp: int, style: float, status: str, reason: str, contact: str, desk: str) -> int:
        b = brokers[bid - 1]
        b[1], b[3], b[4], b[6], b[7], b[8] = name, credit, dtp, style, status, reason
        b[10] = f"loads@{name.lower().replace(' ', '')}.example"
        contacts.append((len(contacts) + 1, bid, contact, desk, "ext 214"))
        return len(contacts)

    c_crescent = plant_broker(11, "Crescent Freight Partners", "A", 21, 0.88, "active", "", "Mike Delgado", "reefer desk")
    c_rapid = plant_broker(12, "Rapid Eagle Brokerage", "D", 60, 1.18, "do_not_use",
                           "Double-brokered our load RO-43120 in June and paid 74 days late.", "Tony Marsh", "dry van desk")
    c_greatlakes = plant_broker(13, "Great Lakes Freight Brokerage", "A", 30, 0.97, "active", "", "Jenna Walsh", "Midwest desk")
    c_desert = plant_broker(14, "Desert Sun Logistics", "B", 30, 0.95, "active", "", "Carlos Vega", "reefer desk")
    c_gulf = plant_broker(15, "Gulf Coast Chemical Logistics", "A", 21, 1.00, "active", "", "Dana Price", "Texas desk")
    c_river = plant_broker(16, "Riverbend Freight Group", "B", 30, 0.98, "active", "", "Paul Hughes", "Midwest desk")
    brokers_rows = [tuple(b) for b in brokers]

    # ------------------------------------------------------------------ market rates, fuel
    weeks = [week_of(now - timedelta(weeks=w)) for w in range(25, -1, -1)]
    lane_rows = []
    walk: dict[tuple[str, str, str], float] = {}
    for w in weeks:
        for o in REGIONS:
            for d in REGIONS:
                for eq, base in EQUIP_BASE.items():
                    k = (o, d, eq)
                    walk[k] = min(1.12, max(0.9, walk.get(k, rng.uniform(0.96, 1.04)) + rng.uniform(-0.025, 0.025)))
                    rpm = base * OUT_F[o] * IN_F[d] * walk[k]
                    lane_rows.append((o, d, eq, w, round(rpm, 2), round(max(0.8, rng.gauss(3.2, 1.1) * OUT_F[o]), 1)))
    db.executemany("INSERT INTO lane_rates VALUES (?,?,?,?,?,?)", lane_rows)
    market_now = {(o, d, eq): rpm for (o, d, eq, w, rpm, _) in lane_rows if w == weeks[-1]}

    def market_rpm(o: int, d: int, eq: str, miles: int) -> float:
        rpm = market_now[(city[o][5], city[d][5], eq)]
        if miles < 400:
            rpm *= 1 + (400 - miles) / 650  # short hauls pay more per mile
        return rpm

    fuel_rows = []
    for r in REGIONS:
        price = rng.uniform(3.55, 4.35) + (0.55 if r in ("California", "Pacific NW") else 0)
        for w in weeks[-12:]:
            price += rng.uniform(-0.06, 0.06)
            fuel_rows.append((r, w, round(price, 3)))
    db.executemany("INSERT INTO fuel VALUES (?,?,?)", fuel_rows)

    # ------------------------------------------------------------------ fleet
    drivers, trucks, trailers = [], [], []
    for did in range(1, 141):
        home = rng.choices([c[0] for c in cities], weights)[0]
        note = rng.choice(DRIVER_NOTES).replace("{home}", city[home][1])
        drivers.append([did, f"{rng.choice(FIRST)} {rng.choice(LAST)}", f"({rng.randint(201, 989)}) 555-{rng.randint(1000, 9999)}",
                        home, "available", int(rng.random() < 0.3), int(rng.random() < 0.25), round(rng.uniform(3.6, 5.0), 1),
                        round(rng.uniform(4, 11), 1), round(rng.uniform(6, 14), 1), round(rng.uniform(12, 70), 1), ts(now), note])
    types = ["dry_van"] * 90 + ["reefer"] * 60 + ["flatbed"] * 30
    rng.shuffle(types)
    for n, t in enumerate(types):
        prefix = {"dry_van": "V", "reefer": "R", "flatbed": "F"}[t]
        c = rng.choices([c[0] for c in cities], weights)[0]
        trailers.append([f"{prefix}{5000 + n}", t, 48 if t == "flatbed" else 53, "active", c,
                         ts(now - timedelta(days=rng.randint(0, 20))) if t == "reefer" else None,
                         rng.randint(2000, 14000) if t == "reefer" else None])
    unit_numbers = sorted(rng.sample(range(101, 499), 131) + [])
    for special in ("214", "227", "118", "142", "188", "330", "402", "256"):
        if int(special) not in unit_numbers:
            unit_numbers[rng.randrange(len(unit_numbers))] = int(special)
    unit_numbers = sorted(set(unit_numbers))
    for n, unit in enumerate(unit_numbers):
        trl = trailers[n % len(trailers)]
        c = trl[4]
        trucks.append([str(unit), rng.randint(2019, 2026), rng.choice(["Freightliner Cascadia", "Kenworth T680", "Peterbilt 579", "Volvo VNL", "International LT"]),
                       "active", n + 1 if n + 1 <= len(drivers) else None, trl[0], c, city[c][3], city[c][4],
                       rng.randint(500, 25000), ""])
    truck_by_unit = {t[0]: t for t in trucks}
    driver_by_id = {d[0]: d for d in drivers}
    trailer_by_no = {t[0]: t for t in trailers}

    def set_truck(unit: str, city_id: int, trailer_type: str, driver_name: str, *, hazmat: int | None = None,
                  twic: int | None = None, drive: float | None = None, duty: float | None = None, cycle: float | None = None,
                  status: str = "available", available_at: datetime | None = None, note: str | None = None) -> None:
        t = truck_by_unit[unit]
        t[6], t[7], t[8] = city_id, city[city_id][3], city[city_id][4]
        trl = trailer_by_no[t[5]]
        if trl[1] != trailer_type:  # swap in a trailer of the right type
            other = next(x for x in trailers if x[1] == trailer_type and x[0] not in {tt[5] for tt in trucks})
            t[5] = other[0]
            trl = other
        trl[4] = city_id
        if trailer_type == "reefer":
            trl[5] = ts(now - timedelta(days=2))
        d = driver_by_id[t[4]]
        d[1], d[4] = driver_name, status
        d[11] = ts(available_at or now)
        if hazmat is not None:
            d[5] = hazmat
        if twic is not None:
            d[6] = twic
        if drive is not None:
            d[8] = drive
        if duty is not None:
            d[9] = duty
        if cycle is not None:
            d[10] = cycle
        if note is not None:
            d[12] = note

    # ------------------------------------------------------------------ history: 24k delivered loads
    loads, pings = [], []
    load_no = 20000
    eq_of_truck = {t[0]: trailer_by_no[t[5]][1] for t in trucks}
    for _ in range(24000):
        load_no += 1
        t = rng.choice(trucks)
        eq = eq_of_truck[t[0]]
        o = rng.choices([c[0] for c in cities], weights)[0]
        d = rng.choices([c[0] for c in cities], weights)[0]
        while d == o or not 90 <= dist(o, d) <= 2600:
            d = rng.choices([c[0] for c in cities], weights)[0]
        miles = dist(o, d)
        b = brokers[rng.randrange(len(brokers))]
        picked = now - timedelta(days=rng.uniform(3, 365), hours=rng.uniform(0, 12))
        delivered = picked + timedelta(hours=miles / AVG_MPH + rng.uniform(0, 10))
        rate = int(round(market_rpm(o, d, eq, miles) * miles * rng.uniform(0.92, 1.12) / 25) * 25)
        paid = delivered + timedelta(days=b[4] + rng.randint(-3, 12))
        status = "paid" if paid < now else "invoiced"
        commodity = rng.choice(COMMODITIES[eq])
        loads.append((f"RO-{load_no}", b[0], str(rng.randint(1000000, 9999999)), None, rng.choice(fac_by_city[o]),
                      rng.choice(fac_by_city[d]), ts(picked), ts(delivered), ts(picked), ts(delivered), t[4], t[0],
                      t[5], eq, commodity, rng.randint(18000, 44000), REEFER_TEMP.get(commodity) if eq == "reefer" else None,
                      miles, rng.randint(0, 140), rate, rng.choice([0, 0, 0, 0, 60, 120, 180]), rng.choice([0, 0, 0, 150, 200]),
                      status, ts(delivered + timedelta(days=1)), ts(paid) if status == "paid" else None, "", ts(picked - timedelta(days=1))))
        for f in (0.33, 0.66, 1.0):  # three pings a trip in history
            lat = city[o][3] + (city[d][3] - city[o][3]) * f
            lon = city[o][4] + (city[d][4] - city[o][4]) * f
            pings.append((None, f"RO-{load_no}", t[0], ts(picked + (delivered - picked) * f), round(lat, 4), round(lon, 4),
                          None, int(miles * (1 - f)), rng.randint(0, 65), "delivered" if f == 1.0 else ""))

    # ------------------------------------------------------------------ active loads on the road now
    busy: set[str] = set()
    reserved = {"214", "227", "118", "142", "188", "330", "402", "256"}

    def add_active(number: str, unit: str, broker_id: int, ref: str, o_fac: int, d_fac: int, pickup: datetime,
                   delivery: datetime, commodity: str, weight: int, rate: int, progress: float, status: str,
                   ping_back_min: int = 25, notes: str = "") -> None:
        t = truck_by_unit[unit]
        eq = trailer_by_no[t[5]][1]
        o = facilities[o_fac - 1][3]
        d = facilities[d_fac - 1][3]
        miles = dist(o, d)
        picked_at = pickup + timedelta(minutes=rng.randint(10, 90)) if progress > 0 else None
        loads.append((number, broker_id, ref, None, o_fac, d_fac, ts(pickup), ts(delivery), ts(picked_at) if picked_at else None,
                      None, t[4], unit, t[5], eq, commodity, weight, REEFER_TEMP.get(commodity) if eq == "reefer" else None,
                      miles, rng.randint(0, 80), rate, 0, 0, status, None, None, notes, ts(pickup - timedelta(days=1))))
        lat = city[o][3] + (city[d][3] - city[o][3]) * progress
        lon = city[o][4] + (city[d][4] - city[o][4]) * progress
        t[7], t[8] = round(lat, 4), round(lon, 4)
        nearest = min(cities, key=lambda c: haversine(lat, lon, c[3], c[4]))
        t[6] = nearest[0]
        if picked_at:
            n = 8
            for k in range(n + 1):
                f = progress * k / n
                at = picked_at + (now - timedelta(minutes=ping_back_min) - picked_at) * (k / n)
                plat = city[o][3] + (city[d][3] - city[o][3]) * f
                plon = city[o][4] + (city[d][4] - city[o][4]) * f
                near = min(cities, key=lambda c: haversine(plat, plon, c[3], c[4]))
                pings.append((None, number, unit, ts(at), round(plat, 4), round(plon, 4), near[0], int(miles * (1 - f)),
                              rng.randint(52, 66) if k < n else rng.randint(55, 64), "picked up" if k == 0 else ""))
        dr = driver_by_id[t[4]]
        dr[4] = "on_load"
        busy.add(unit)

    load_no = 48000
    free_units = [t[0] for t in trucks if t[0] not in reserved and t[4] is not None]
    rng.shuffle(free_units)
    for unit in free_units[:70]:
        load_no += 1
        if f"RO-{load_no}" in ("RO-48213", "RO-48307"):
            load_no += 1
        eq = eq_of_truck[unit]
        o = rng.choices([c[0] for c in cities], weights)[0]
        d = rng.choices([c[0] for c in cities], weights)[0]
        while d == o or not 150 <= dist(o, d) <= 2200:
            d = rng.choices([c[0] for c in cities], weights)[0]
        miles = dist(o, d)
        progress = rng.choice([0.0, 0.0, rng.uniform(0.1, 0.95), rng.uniform(0.1, 0.95), rng.uniform(0.1, 0.95)])
        trip_h = miles / AVG_MPH
        pickup = now - timedelta(hours=trip_h * progress + rng.uniform(1, 3)) if progress else now + timedelta(hours=rng.uniform(1, 20))
        delivery = pickup + timedelta(hours=trip_h + rng.uniform(2, 14))
        status = "in_transit" if progress else "dispatched"
        commodity = rng.choice(COMMODITIES[eq])
        rate = int(round(market_rpm(o, d, eq, miles) * miles * rng.uniform(0.95, 1.1) / 25) * 25)
        add_active(f"RO-{load_no}", unit, rng.randrange(1, 221), str(rng.randint(1000000, 9999999)),
                   rng.choice(fac_by_city[o]), rng.choice(fac_by_city[d]), pickup, delivery, commodity,
                   rng.randint(18000, 44000), rate, progress, status)
    for unit in free_units[70:]:  # the rest: empty somewhere, some off duty or resetting
        d = driver_by_id[truck_by_unit[unit][4]]
        r = rng.random()
        if r < 0.18:
            d[4], d[11] = "off_duty", ts(now + timedelta(hours=rng.uniform(4, 30)))
        elif r < 0.28:
            d[4], d[11], d[8], d[9] = "reset", ts(now + timedelta(hours=rng.uniform(6, 34))), 0.0, 0.0
        elif r < 0.33:
            d[4], d[11] = "home_time", ts(now + timedelta(days=rng.randint(1, 4)))
    for t in trucks:
        if t[4] is None:
            t[3], t[10] = "shop", "No driver assigned; PM service due."

    # ------------------------------------------------------------------ planted scenarios
    dallas, ftworth, atlanta = cid("Dallas", "TX"), cid("Fort Worth", "TX"), cid("Atlanta", "GA")
    lone_star = add_facility("Lone Star Produce Terminal", "cold", dallas, "05:00", "15:00", 90, 0,
                             "Reefer must be pre-cooled to 34 before check-in. Pulp temps checked at the door.")
    peachtree = add_facility("Peachtree Fresh DC", "cold", atlanta, "00:00", "23:59", 120, 225,
                             "Lumper required, 225 dollars, reimbursed with receipt. Appointments only.")
    set_truck("214", ftworth, "reefer", "Luis Ortega", drive=10.5, duty=13.0, cycle=52.0,
              note="Clean record, likes produce runs to the Southeast.")
    set_truck("256", dallas, "reefer", "Gary Fisher", drive=1.5, duty=2.0, cycle=6.0, status="reset",
              available_at=now + timedelta(hours=30), note="Out of hours: 34-hour reset.")

    joliet, columbus = cid("Joliet", "IL"), cid("Columbus", "OH")
    midway = add_facility("Midway Packaging Co.", "plant", joliet, "06:00", "18:00", 60, 0, "")
    buckeye = add_facility("Buckeye Retail DC", "dc", columbus, "04:00", "20:00", 150, 175,
                           "Strict appointments: more than 30 minutes late gets rescheduled to the next open slot.")
    set_truck("227", joliet, "dry_van", "Tasha Brooks", drive=6.5, duty=7.5, cycle=33.0, status="on_load")
    late_delivery = now + timedelta(hours=3)
    late_delivery = late_delivery.replace(minute=late_delivery.minute // 15 * 15)  # a quarter-hour slot 2:45-3:00 from now
    # the last ping, 20 minutes ago, had ~61 % of the ~340 miles to go: at 50 mph she arrives ~45-50 min
    # after the 3-hour appointment, and the receiver reschedules anything over 30 minutes late
    add_active("RO-48213", "227", 13, "7781234", midway, buckeye, now - timedelta(hours=5),
               late_delivery, "retail freight", 38500, 1150, 0.39, "in_transit", ping_back_min=20,
               notes="Broker ref is their PO 7781234.")

    memphis, chicago = cid("Memphis", "TN"), cid("Chicago", "IL")
    rapid_origin = add_facility("Delta Paper Mill", "plant", memphis, "06:00", "16:00", 60, 0, "")
    rapid_dest = add_facility("Lakefront Printing Supply", "dc", chicago, "07:00", "15:00", 90, 0, "")

    houston, beaumont, memphis_id = cid("Houston", "TX"), cid("Beaumont", "TX"), memphis
    bayou = add_facility("Bayou Solvents Plant", "plant", houston, "06:00", "16:00", 90, 0,
                         "Hazmat class 3. Placards required, TWIC card at the gate.")
    river_chem = add_facility("River City Coatings", "plant", memphis_id, "07:00", "15:00", 60, 0, "")
    set_truck("142", houston, "dry_van", "Kevin Park", hazmat=0, drive=11.0, duty=14.0, cycle=60.0,
              note="No hazmat endorsement.")
    set_truck("188", beaumont, "dry_van", "Rosa Jimenez", hazmat=1, twic=1, drive=10.0, duty=12.5, cycle=48.0,
              note="Hazmat and TWIC. Prefers chemical freight.")

    southaven, indy, stl = cid("Southaven", "MS"), cid("Indianapolis", "IN"), cid("St. Louis", "MO")
    set_truck("330", memphis, "dry_van", "Marcus Hill", drive=11.0, duty=14.0, cycle=58.0, status="available",
              available_at=(now + timedelta(days=1)).replace(hour=6, minute=0),
              note="Empty in Memphis from 6 am tomorrow, wants to head north toward home in Indianapolis.")
    driver_by_id[truck_by_unit["330"][4]][3] = indy

    okc, phoenix, tulsa = cid("Oklahoma City", "OK"), cid("Phoenix", "AZ"), cid("Tulsa", "OK")
    ozark = add_facility("Ozark Poultry Plant", "cold", cid("Fort Smith", "AR"), "00:00", "23:59", 90, 0, "")
    sonoran = add_facility("Sonoran Cold Storage", "cold", phoenix, "03:00", "17:00", 120, 200,
                           "Appointments only, 2 hour window. Reefer set to 0.")
    set_truck("118", okc, "reefer", "Andre Coleman", drive=6.0, duty=8.0, cycle=40.0, status="on_load")
    # Fort Smith to Phoenix on I-40: Oklahoma City is about 18 % of the way
    add_active("RO-48307", "118", 14, "PHX-55120", ozark, sonoran, now - timedelta(hours=7),
               now + timedelta(hours=20), "frozen chicken", 38000, 3350, 0.18, "delayed", ping_back_min=190,
               notes="Truck broke down on I-40 near Oklahoma City.")
    set_truck("402", tulsa, "reefer", "Denise Carter", drive=11.0, duty=14.0, cycle=55.0, status="available",
              note="Empty in Tulsa, can take a recovery load.")
    db.executemany("INSERT INTO incidents VALUES (?,?,?,?,?,?,?,?)", [
        (1, "118", "RO-48307", ts(now - timedelta(hours=3, minutes=10)), "breakdown", okc,
         "Turbo failure on I-40 eastbound near mile 140. Towed to Rush Truck Center, Oklahoma City. Shop says parts "
         "in 2 days. Reefer is running on its own fuel, 38,000 lbs frozen chicken at 0 F, product fine.", 0),
    ])

    # ------------------------------------------------------------------ the load board: ~6k open postings
    postings = []
    pno = 100000
    for _ in range(6000):
        pno += 1
        o = rng.choices([c[0] for c in cities], weights)[0]
        d = rng.choices([c[0] for c in cities], weights)[0]
        while d == o or not 120 <= dist(o, d) <= 2500:
            d = rng.choices([c[0] for c in cities], weights)[0]
        miles = dist(o, d)
        eq = rng.choices(list(EQUIP_BASE), [60, 28, 12])[0]
        b = brokers[rng.randrange(len(brokers))]
        pickup = (now + timedelta(hours=rng.uniform(2, 100))).replace(minute=0)
        window = rng.choice([2, 3, 4, 6])
        delivery = pickup + timedelta(hours=miles / AVG_MPH + rng.uniform(3, 18))
        commodity = rng.choice(COMMODITIES[eq])
        rate = None if rng.random() < 0.15 else int(round(market_rpm(o, d, eq, miles) * miles * b[6] * rng.uniform(0.95, 1.04) / 25) * 25)
        hazmat = int(eq == "dry_van" and rng.random() < 0.04)
        postings.append((f"LB-{pno}", b[0], None, rng.choice(fac_by_city[o]), rng.choice(fac_by_city[d]),
                         ts(pickup), ts(pickup + timedelta(hours=window)), ts(delivery), ts(delivery + timedelta(hours=window)),
                         eq, rng.randint(15000, 44000), commodity, REEFER_TEMP.get(commodity) if eq == "reefer" else None,
                         miles, rate, hazmat, int(miles > 1800 and rng.random() < 0.3), int(eq == "flatbed" and rng.random() < 0.5),
                         rng.choice([1, 1, 1, 2, 3]), "", "open", ts(now - timedelta(hours=rng.uniform(0, 30)))))
    tomorrow7 = (now + timedelta(days=1)).replace(hour=7, minute=0)
    dal_atl = dist(dallas, atlanta)
    postings += [
        ("LB-500101", 11, c_crescent, lone_star, peachtree, ts(tomorrow7), ts(tomorrow7 + timedelta(hours=3)),
         ts(tomorrow7 + timedelta(hours=26)), ts(tomorrow7 + timedelta(hours=30)), "reefer", 42000, "produce", 34,
         dal_atl, 1900, 0, 0, 0, 1, "Mixed produce, pulp temps at pickup. Lumper at receiver.", "open", ts(now - timedelta(hours=1))),
        ("LB-500201", 12, c_rapid, rapid_origin, rapid_dest, ts(tomorrow7 + timedelta(hours=1)), ts(tomorrow7 + timedelta(hours=4)),
         ts(tomorrow7 + timedelta(hours=16)), ts(tomorrow7 + timedelta(hours=20)), "dry_van", 40000, "paper products", None,
         dist(memphis, chicago), int(dist(memphis, chicago) * 3.40), 0, 0, 0, 1, "Needs a truck now, paying top dollar.", "open",
         ts(now - timedelta(minutes=40))),
        ("LB-500301", 15, c_gulf, bayou, river_chem, ts(tomorrow7 + timedelta(hours=1)), ts(tomorrow7 + timedelta(hours=5)),
         ts(tomorrow7 + timedelta(hours=18)), ts(tomorrow7 + timedelta(hours=24)), "dry_van", 41000, "paint solvent (UN1263, class 3)",
         None, dist(houston, memphis_id), int(dist(houston, memphis_id) * 2.95), 1, 0, 0, 1,
         "Hazmat class 3, placards, TWIC required at the plant gate.", "open", ts(now - timedelta(hours=2))),
    ]
    tmw = now + timedelta(days=1)
    for pid, dest, rate, org, note in (
        ("LB-500401", indy, 1275, memphis, "Drop and hook at both ends."),
        ("LB-500402", chicago, 1350, memphis, "Live load, 2 hours free."),
        ("LB-500403", stl, 900, southaven, "Short run, live unload."),
        ("LB-500404", columbus, 1500, memphis, "Two stops in Columbus, 75 per extra stop."),
    ):
        o_fac = add_facility(f"{rng.choice(FAC_A)} Distribution Center", "dc", org, "06:00", "18:00", 60, 0, "")
        d_fac = rng.choice(fac_by_city[dest])
        m = dist(org, dest)
        postings.append((pid, 16, c_river, o_fac, d_fac, ts(tmw.replace(hour=8, minute=0)), ts(tmw.replace(hour=11, minute=0)),
                         ts(tmw.replace(hour=8, minute=0) + timedelta(hours=m / AVG_MPH + 4)),
                         ts(tmw.replace(hour=8, minute=0) + timedelta(hours=m / AVG_MPH + 8)), "dry_van", 38000,
                         "retail freight", None, m, rate, 0, 0, 0, 2 if pid == "LB-500404" else 1, note, "open", ts(now - timedelta(hours=3))))

    # ------------------------------------------------------------------ write
    db.executemany("INSERT INTO facilities VALUES (?,?,?,?,?,?,?,?,?,?,?)", facilities)
    db.executemany("INSERT INTO brokers VALUES (?,?,?,?,?,?,?,?,?,?,?)", brokers_rows)
    db.executemany("INSERT INTO broker_contacts VALUES (?,?,?,?,?)", contacts)
    db.executemany("INSERT INTO drivers VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", [tuple(d) for d in drivers])
    db.executemany("INSERT INTO trucks VALUES (?,?,?,?,?,?,?,?,?,?,?)", [tuple(t) for t in trucks])
    db.executemany("INSERT INTO trailers VALUES (?,?,?,?,?,?,?)", [tuple(t) for t in trailers])
    db.executemany("INSERT INTO postings VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", postings)
    db.executemany("INSERT INTO loads VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", loads)
    db.executemany("INSERT INTO pings VALUES (?,?,?,?,?,?,?,?,?,?)", pings)
    db.executemany("INSERT INTO policy VALUES (?,?,?)", POLICY)
    db.executemany("INSERT INTO messages VALUES (?,?,?,?,?,?)", [
        (1, ts(now - timedelta(minutes=35)), "in", "Tasha Brooks (truck 227)", "RO-48213",
         "Construction on US-30 past Fort Wayne, down to one lane. Moving but slow."),
        (2, ts(now - timedelta(hours=3)), "in", "Andre Coleman (truck 118)", "RO-48307",
         "Lost power on I-40, turbo is shot. Waiting on the tow. Reefer is running fine at zero."),
        (3, ts(now - timedelta(hours=1, minutes=5)), "in", "Andre Coleman (truck 118)", "RO-48307",
         "At Rush Truck Center OKC. They say two days for parts."),
    ])
    db.executemany("INSERT INTO meta VALUES (?,?)", [("built_at", ts(now)), ("seed", str(seed)), ("company", COMPANY)])
    db.commit()
    counts = {t: db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in
              ("cities", "facilities", "brokers", "broker_contacts", "drivers", "trucks", "trailers", "postings",
               "loads", "pings", "lane_rates", "fuel", "incidents", "messages")}
    db.close()
    for attempt in range(20):  # Windows: a scanner can hold a fresh file for a moment
        try:
            tmp.replace(path)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.25)
    return counts


if __name__ == "__main__":
    t0 = time.perf_counter()
    counts = build()
    print(f"{DB_PATH} in {time.perf_counter() - t0:.1f} s ({DB_PATH.stat().st_size / 1e6:.1f} MB)")
    print(", ".join(f"{k} {v:,}" for k, v in counts.items()))
    sys.exit(0)
