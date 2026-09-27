"""The dispatcher's desk: the tools the agent uses on Red Oak Transport's database.

Every backend gets the same tools (Eva's own loop through ``dispatch_tools()``, the speech-to-
speech clients through ``eva.s2s.tools``): plain functions over SQLite that return short JSON
text. They are built so that answering a caller takes several hops the agent has to choose
itself: a broker's load -> a truck that can reach it -> its driver's hours -> the lane's market
rate and our cost floor -> the booking; a reference number -> the load -> its last ping ->
the receiver's rules. Nothing here is spoken as is; the model turns it into speech.

The rules the tools enforce, whatever the model says: never below the cost floor, never a
do-not-use broker, hazmat only with an endorsed driver, the right trailer, a truck that can make
the pickup window.
"""
from __future__ import annotations

import difflib
import json
import math
import re
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ...interfaces import Tool
from .world import AVG_MPH, COMPANY, DB_PATH, build, haversine, road_miles

STALE_AFTER = timedelta(hours=12)  # the world is anchored at its build time: rebuild when it drifts
EQUIPMENT_WORDS = {
    "dry_van": ("dry van", "van", "dry", "dryvan", "53 van", "box"),
    "reefer": ("reefer", "refrigerated", "temp controlled", "temperature controlled", "cold", "frozen"),
    "flatbed": ("flatbed", "flat", "step deck", "stepdeck", "open deck"),
}
ALIASES = {
    "dfw": "Dallas", "big d": "Dallas", "atl": "Atlanta", "okc": "Oklahoma City", "kc": "Kansas City",
    "nola": "New Orleans", "la": "Los Angeles", "l a": "Los Angeles", "slc": "Salt Lake City", "chi": "Chicago",
    "chicagoland": "Chicago", "twin cities": "Minneapolis", "philly": "Philadelphia", "vegas": "Las Vegas",
    "indy": "Indianapolis", "st louis": "St. Louis", "saint louis": "St. Louis", "stl": "St. Louis",
    "jax": "Jacksonville", "inland empire": "Ontario", "the bay area": "Oakland", "bay area": "Oakland",
    "motor city": "Detroit", "nyc": "Newark", "new york city": "Newark", "new york": "Newark", "nashville tn": "Nashville",
}
STATES = {
    "alabama": "AL", "arizona": "AZ", "arkansas": "AR", "california": "CA", "colorado": "CO", "connecticut": "CT",
    "florida": "FL", "georgia": "GA", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD", "massachusetts": "MA", "michigan": "MI",
    "minnesota": "MN", "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new jersey": "NJ", "new mexico": "NM", "north carolina": "NC", "north dakota": "ND",
    "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA", "south carolina": "SC", "south dakota": "SD",
    "tennessee": "TN", "texas": "TX", "utah": "UT", "virginia": "VA", "washington": "WA", "wisconsin": "WI",
}
AREAS = {  # words people use for a direction, as regions of the rate table
    "midwest": ["Upper Midwest", "Great Lakes", "Plains"], "the midwest": ["Upper Midwest", "Great Lakes", "Plains"],
    "southeast": ["Southeast", "Florida", "Mid-South"], "northeast": ["Northeast", "Mid-Atlantic"],
    "east coast": ["Northeast", "Mid-Atlantic", "Southeast", "Florida"], "west coast": ["California", "Pacific NW"],
    "pacific northwest": ["Pacific NW"], "southwest": ["Southwest", "Mountain"], "the south": ["Mid-South", "Southeast", "Gulf"],
    "great lakes": ["Great Lakes"], "gulf": ["Gulf", "South Central"], "mountain": ["Mountain"], "plains": ["Plains"],
    "north": ["Upper Midwest", "Great Lakes", "Plains", "Northeast"],
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def _j(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


class Desk:
    """One connection to the world, shared by the tool functions (they run in worker threads)."""

    def __init__(self, path: Path = DB_PATH, *, now: datetime | None = None, rebuild: bool | None = None) -> None:
        self.path = path
        fresh = rebuild
        if fresh is None:
            fresh = not path.exists() or self._age(path) > STALE_AFTER
        if fresh:
            build(path, now=now)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        self.built_at = datetime.strptime(self.q1("SELECT value FROM meta WHERE key='built_at'")["value"], "%Y-%m-%d %H:%M")
        self.cities = {r["id"]: r for r in self.q("SELECT * FROM cities")}
        self.policy = {r["key"]: r["value"] for r in self.q("SELECT key, value FROM policy")}

    @staticmethod
    def _age(path: Path) -> timedelta:
        try:
            con = sqlite3.connect(path)
            v = con.execute("SELECT value FROM meta WHERE key='built_at'").fetchone()[0]
            con.close()
            return datetime.now() - datetime.strptime(v, "%Y-%m-%d %H:%M")
        except Exception:  # noqa: BLE001 - anything wrong with the file means rebuild
            return STALE_AFTER * 2

    # ------------------------------------------------------------------ plumbing
    def close(self) -> None:
        with self.lock:
            self.db.close()

    def q(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self.lock:
            return self.db.execute(sql, args).fetchall()

    def q1(self, sql: str, args: tuple = ()) -> sqlite3.Row | None:
        with self.lock:
            return self.db.execute(sql, args).fetchone()

    def write(self, sql: str, args: tuple = ()) -> None:
        with self.lock:
            self.db.execute(sql, args)
            self.db.commit()

    def now(self) -> datetime:
        return datetime.now().replace(second=0, microsecond=0)

    def when(self, s: str | datetime | None) -> str | None:
        if s is None:
            return None
        dt = s if isinstance(s, datetime) else datetime.strptime(s, "%Y-%m-%d %H:%M")
        days = (dt.date() - self.now().date()).days
        day = {0: "today", 1: "tomorrow", -1: "yesterday"}.get(days, dt.strftime("%a %d %b"))
        return f"{day} {dt:%H:%M}"

    def place(self, city_id: int | None) -> str:
        if city_id is None:
            return "?"
        c = self.cities[city_id]
        return f"{c['name']}, {c['state']}"

    def nearest_city(self, lat: float, lon: float) -> int:
        return min(self.cities.values(), key=lambda c: haversine(lat, lon, c["lat"], c["lon"]))["id"]

    def miles_between(self, a: int, b: int) -> int:
        ca, cb = self.cities[a], self.cities[b]
        return road_miles(ca["lat"], ca["lon"], cb["lat"], cb["lon"])

    # ------------------------------------------------------------------ resolving words
    def resolve(self, text: str) -> list[int]:
        """City ids for a place as a caller says it: a city (with or without state), an alias, a state
        or a direction ("the Midwest"). Empty when nothing matches."""
        t = _norm(text)
        if not t or t in ("any", "anywhere", "open"):
            return []
        t = ALIASES.get(t, t)
        tl = _norm(t)
        m = re.match(r"^(.*?)[ ,]+([a-z]{2})$", tl)
        state = None
        if m and m.group(2).upper() in set(STATES.values()):
            tl, state = m.group(1).strip(), m.group(2).upper()
        for name, ab in STATES.items():
            if tl.endswith(" " + name):
                tl, state = tl[: -len(name)].strip(), ab
        names = {_norm(c["name"]): [] for c in self.cities.values()}
        for c in self.cities.values():
            names[_norm(c["name"])].append(c["id"])
        if tl in names:
            ids = names[tl]
            return [i for i in ids if self.cities[i]["state"] == state] or ids if state else ids
        if tl in STATES:
            return [c["id"] for c in self.cities.values() if c["state"] == STATES[tl]]
        if tl.upper() in set(STATES.values()) and len(tl) == 2:
            return [c["id"] for c in self.cities.values() if c["state"] == tl.upper()]
        if tl in AREAS:
            return [c["id"] for c in self.cities.values() if c["region"] in AREAS[tl]]
        regions = {_norm(r): r for r in {c["region"] for c in self.cities.values()}}
        if tl in regions:
            return [c["id"] for c in self.cities.values() if c["region"] == regions[tl]]
        close = difflib.get_close_matches(tl, list(names), n=1, cutoff=0.8)
        return names[close[0]] if close else []

    def near(self, ids: list[int], radius: float) -> set[int]:
        out = set(ids)
        for c in self.cities.values():
            if any(self.miles_between(c["id"], i) <= radius for i in ids):
                out.add(c["id"])
        return out

    @staticmethod
    def equipment(text: str) -> str | None:
        t = _norm(text or "")
        if not t or t == "any":
            return None
        for eq, words in EQUIPMENT_WORDS.items():
            if t == eq.replace("_", " ") or t in words or any(w in t for w in words):
                return eq
        return None

    def day(self, text: str) -> datetime | None:
        t = _norm(text or "")
        if not t or t in ("any", "anytime", "whenever"):
            return None
        today = self.now().replace(hour=0, minute=0)
        if t in ("today", "now", "tonight", "asap"):
            return today
        if t == "tomorrow":
            return today + timedelta(days=1)
        if t in ("day after tomorrow",):
            return today + timedelta(days=2)
        for i in range(7):
            d = today + timedelta(days=i)
            if t.startswith(d.strftime("%A").lower()) or t.startswith(d.strftime("%a").lower()):
                return d
        for fmt in ("%Y %m %d", "%m %d", "%d %m %Y"):
            try:
                d = datetime.strptime(t, fmt)
                return d.replace(year=today.year) if fmt == "%m %d" else d
            except ValueError:
                pass
        return None

    # ------------------------------------------------------------------ money
    def market(self, o: int, d: int, eq: str, miles: int) -> dict[str, Any]:
        ro, rd = self.cities[o]["region"], self.cities[d]["region"]
        rows = self.q("SELECT week, rpm, loads_per_truck FROM lane_rates WHERE origin_region=? AND dest_region=? "
                      "AND equipment=? ORDER BY week DESC LIMIT 5", (ro, rd, eq))
        short = 1 + (400 - miles) / 650 if miles < 400 else 1.0
        rpm_now = rows[0]["rpm"] * short
        rpm_30 = sum(r["rpm"] for r in rows[:4]) / len(rows[:4]) * short
        trend = (rows[0]["rpm"] - rows[-1]["rpm"]) / rows[-1]["rpm"] if len(rows) > 1 else 0.0
        return {"rpm": round(rpm_now, 2), "rpm_30d": round(rpm_30, 2), "trend_4wk_pct": round(trend * 100, 1),
                "loads_per_truck": rows[0]["loads_per_truck"], "regions": f"{ro} to {rd}"}

    def floor(self, eq: str, miles: int, deadhead: int) -> int:
        cost = float(self.policy["cost_per_mile"]) * (miles + deadhead) + float(self.policy["min_margin"])
        by_rpm = float(self.policy[f"min_rpm_{eq}"]) * miles
        return int(math.ceil(max(cost, by_rpm) / 25) * 25)

    # ------------------------------------------------------------------ trucks
    def truck_positions(self) -> list[dict[str, Any]]:
        """Every truck with a driver: where and when it is free next, and why not if it can't take a load."""
        out = []
        rows = self.q("SELECT t.unit, t.status AS tstatus, t.trailer, t.city_id, t.lat, t.lon, t.notes AS tnotes, "
                      "tr.type AS ttype, tr.last_washout, d.id AS did, d.name, d.phone, d.status, d.hazmat, d.twic, "
                      "d.rating, d.drive_left_h, d.duty_left_h, d.cycle_left_h, d.available_at, d.notes, d.home_city_id "
                      "FROM trucks t JOIN drivers d ON d.id = t.driver_id JOIN trailers tr ON tr.number = t.trailer")
        now = self.now()
        broken = {r["truck"] for r in self.q("SELECT truck FROM incidents WHERE resolved = 0")}
        for r in rows:
            item = dict(r)
            item["free_at"] = max(now, datetime.strptime(r["available_at"], "%Y-%m-%d %H:%M"))
            item["free_city"] = r["city_id"]
            item["lat_now"], item["lon_now"] = r["lat"], r["lon"]
            item["blocked"] = None
            if r["tstatus"] != "active":
                item["blocked"] = "truck in the shop"
            elif r["unit"] in broken:
                item["blocked"] = "broken down"
            elif r["status"] == "on_load":
                load = self.q1("SELECT * FROM loads WHERE truck = ? AND status IN ('booked','dispatched','in_transit','delayed') "
                               "ORDER BY pickup_appt DESC LIMIT 1", (r["unit"],))
                if load is not None:
                    eta = self.eta(load)
                    item["free_at"] = (eta["eta"] if eta["eta"] else now) + timedelta(hours=1)
                    item["free_city"] = self.q1("SELECT city_id FROM facilities WHERE id=?", (load["dest_id"],))["city_id"]
                    item["on_load"] = load["number"]
                    c = self.cities[item["free_city"]]
                    item["lat_now"], item["lon_now"] = c["lat"], c["lon"]
            elif r["status"] in ("off_duty", "reset", "home_time") and item["free_at"] <= now:
                item["status"] = "available"
            out.append(item)
        return out

    # ------------------------------------------------------------------ eta
    def eta(self, load: sqlite3.Row) -> dict[str, Any]:
        """When the load reaches its receiver: from the last ping, 50 mph, and a 10-hour break whenever
        the driver's 11 hours of driving run out."""
        ping = self.q1("SELECT * FROM pings WHERE load_number = ? ORDER BY at DESC LIMIT 1", (load["number"],))
        drv = self.q1("SELECT drive_left_h FROM drivers WHERE id = ?", (load["driver_id"],))
        now = self.now()
        dest = self.q1("SELECT city_id FROM facilities WHERE id=?", (load["dest_id"],))["city_id"]
        if ping is None:  # not picked up yet: from the truck's position through the pickup
            truck = self.q1("SELECT lat, lon FROM trucks WHERE unit = ?", (load["truck"],))
            origin = self.q1("SELECT city_id FROM facilities WHERE id=?", (load["origin_id"],))["city_id"]
            oc = self.cities[origin]
            to_pickup = road_miles(truck["lat"], truck["lon"], oc["lat"], oc["lon"]) if truck else 0
            start = max(now, datetime.strptime(load["pickup_appt"], "%Y-%m-%d %H:%M") - timedelta(hours=to_pickup / AVG_MPH))
            depart = max(start + timedelta(hours=to_pickup / AVG_MPH), datetime.strptime(load["pickup_appt"], "%Y-%m-%d %H:%M")) + timedelta(hours=1.5)
            miles = load["miles"]
            since = None
        else:
            depart = datetime.strptime(ping["at"], "%Y-%m-%d %H:%M")
            miles = ping["miles_to_go"]
            since = int((now - depart).total_seconds() // 60)
        drive = (drv["drive_left_h"] if drv else 11.0) or 0.0
        hours = miles / AVG_MPH
        breaks = 0
        left = hours - drive
        while left > 0:
            breaks += 1
            left -= 11.0
        eta = depart + timedelta(hours=hours + 10 * breaks)
        return {"eta": eta, "miles_to_go": miles, "ping_min_ago": since, "breaks": breaks,
                "ping_city": self.nearest_city(ping["lat"], ping["lon"]) if ping else None, "dest_city": dest}

    # ------------------------------------------------------------------ the tools
    def search_load_board(self, origin: str, destination: str = "", equipment: str = "", pickup_date: str = "",
                          radius_miles: int = 75, limit: int = 5) -> str:
        o_ids = self.resolve(origin)
        if not o_ids:
            return _j({"error": f"no market called {origin!r}"})
        area = len(o_ids) > 1 and _norm(origin) not in {_norm(c["name"]) for c in self.cities.values()}
        origins = set(o_ids) if area else self.near(o_ids, radius_miles)
        dests = set(self.resolve(destination)) if destination else None
        eq = self.equipment(equipment)
        day = self.day(pickup_date)
        rows = self.q("SELECT p.*, fo.city_id AS o_city, fo.name AS o_fac, fd.city_id AS d_city, fd.name AS d_fac, "
                      "b.name AS broker, b.status AS bstatus FROM postings p JOIN facilities fo ON fo.id = p.origin_id "
                      "JOIN facilities fd ON fd.id = p.dest_id JOIN brokers b ON b.id = p.broker_id WHERE p.status='open'")
        hits = []
        for r in rows:
            if r["o_city"] not in origins or (dests is not None and r["d_city"] not in dests):
                continue
            if eq and r["equipment"] != eq:
                continue
            ps = datetime.strptime(r["pickup_start"], "%Y-%m-%d %H:%M")
            if day and ps.date() != day.date():
                continue
            hits.append(r)
        hits.sort(key=lambda r: -(r["posted_rate"] / r["miles"]) if r["posted_rate"] else 0)
        out = []
        for r in hits[: max(1, min(int(limit), 10))]:
            flags = [f for f, on in (("hazmat", r["hazmat"]), ("team", r["team"]), ("tarps", r["tarps"]),
                                     (f"{r['stops']} stops", r["stops"] > 1)) if on]
            out.append({"id": r["id"], "broker": r["broker"], "from": f"{self.place(r['o_city'])} ({r['o_fac']})",
                        "to": f"{self.place(r['d_city'])} ({r['d_fac']})", "pickup": f"{self.when(r['pickup_start'])}-{r['pickup_end'][-5:]}",
                        "delivery": self.when(r["delivery_start"]), "equipment": r["equipment"], "miles": r["miles"],
                        "weight": r["weight"], "commodity": r["commodity"] + (f" at {r['temp_f']}F" if r["temp_f"] is not None else ""),
                        "rate": r["posted_rate"] or "call", "rpm": round(r["posted_rate"] / r["miles"], 2) if r["posted_rate"] else None,
                        **({"flags": flags} if flags else {}), **({"notes": r["notes"]} if r["notes"] else {})})
        return _j({"matches": len(hits), "showing": len(out), "loads": out})

    def find_available_trucks(self, near: str, equipment: str = "", needed_by: str = "", radius_miles: int = 150,
                              hazmat: bool = False, limit: int = 5) -> str:
        ids = self.resolve(near)
        if not ids:
            return _j({"error": f"no place called {near!r}"})
        target = self.cities[ids[0]]
        eq = self.equipment(equipment)
        by = self.day(needed_by)
        if by is not None and _norm(needed_by) in ("today", "now", "asap"):
            by = self.now() + timedelta(hours=6)
        elif by is not None:
            by = by + timedelta(hours=14)  # during that day
        ok, excluded = [], []
        for t in self.truck_positions():
            if eq and t["ttype"] != eq:
                continue
            dh = road_miles(t["lat_now"], t["lon_now"], target["lat"], target["lon"])
            if dh > radius_miles:
                continue
            reason = t["blocked"]
            if not reason and hazmat and not t["hazmat"]:
                reason = "driver has no hazmat endorsement"
            limit_at = by if by is not None else self.now() + timedelta(hours=12)
            if not reason and t["free_at"] > limit_at:
                resting = t["status"] in ("reset", "off_duty", "home_time")
                reason = f"{t['status'].replace('_', ' ') if resting else 'busy'} until {self.when(t['free_at'])}"
            if not reason and t["drive_left_h"] < 3 and not t.get("on_load"):
                reason = f"only {t['drive_left_h']} h of driving left"
            entry = {"truck": t["unit"], "driver": t["name"], "trailer": f"{t['ttype']} {t['trailer']}", "deadhead_miles": dh,
                     "where": self.place(t["free_city"]) if t.get("on_load") else self.place(self.nearest_city(t["lat_now"], t["lon_now"])),
                     "free": self.when(t["free_at"])}
            if reason:
                excluded.append({**entry, "why_not": reason})
                continue
            if t["status"] in ("reset", "off_duty", "home_time") and t["free_at"] > self.now():
                entry["status"] = f"{t['status'].replace('_', ' ')} until {self.when(t['free_at'])}"
            entry.update({"finishing_load": t.get("on_load"), "drive_h_left": t["drive_left_h"], "duty_h_left": t["duty_left_h"],
                          "cycle_h_left": t["cycle_left_h"], "hazmat": bool(t["hazmat"]), "twic": bool(t["twic"]),
                          "rating": t["rating"], "notes": t["notes"] or None})
            ok.append({k: v for k, v in entry.items() if v not in (None, "")})
        ok.sort(key=lambda e: (e["deadhead_miles"], e["free"]))
        excluded.sort(key=lambda e: e["deadhead_miles"])
        return _j({"near": self.place(target["id"]), "available": ok[: max(1, min(int(limit), 8))],
                   "nearby_but_not_available": excluded[:3]})

    def price_load(self, posting_id: str = "", origin: str = "", destination: str = "", equipment: str = "",
                   truck: str = "") -> str:
        posting = None
        if posting_id:
            posting = self.q1("SELECT p.*, fo.city_id AS o_city, fd.city_id AS d_city FROM postings p JOIN facilities fo ON "
                              "fo.id=p.origin_id JOIN facilities fd ON fd.id=p.dest_id WHERE p.id = ?", (posting_id.upper().strip(),))
            if posting is None:
                return _j({"error": f"no posting {posting_id}"})
            o, d, eq, miles = posting["o_city"], posting["d_city"], posting["equipment"], posting["miles"]
        else:
            oi, di = self.resolve(origin), self.resolve(destination)
            eq = self.equipment(equipment) or "dry_van"
            if not oi or not di:
                return _j({"error": "need a posting id, or an origin and a destination"})
            o, d = oi[0], di[0]
            miles = self.miles_between(o, d)
        deadhead = 0
        truck_note = None
        if truck:
            t = next((x for x in self.truck_positions() if x["unit"] == str(truck).strip().lstrip("#")), None)
            if t is None:
                return _j({"error": f"no truck {truck}"})
            oc = self.cities[o]
            deadhead = road_miles(t["lat_now"], t["lon_now"], oc["lat"], oc["lon"])
            truck_note = f"truck {t['unit']} is {deadhead} empty miles away"
        mk = self.market(o, d, eq, miles)
        ours = self.q("SELECT rate, miles FROM loads l JOIN facilities fo ON fo.id=l.origin_id JOIN facilities fd ON "
                      "fd.id=l.dest_id WHERE fo.city_id=? AND fd.city_id=? AND l.equipment=? ORDER BY l.pickup_appt DESC LIMIT 8",
                      (o, d, eq))
        floor = self.floor(eq, miles, deadhead)
        target = int(math.ceil(max(floor * 1.08, mk["rpm_30d"] * miles * (1 + float(self.policy["target_over_market"]))) / 25) * 25)
        fuel = self.q1("SELECT diesel FROM fuel WHERE region=? ORDER BY week DESC LIMIT 1", (self.cities[o]["region"],))
        out: dict[str, Any] = {"lane": f"{self.place(o)} to {self.place(d)}", "equipment": eq, "loaded_miles": miles,
                               "deadhead_miles": deadhead, "market_rpm_now": mk["rpm"], "market_rpm_30d": mk["rpm_30d"],
                               "market_rate_30d": int(round(mk["rpm_30d"] * miles / 25) * 25), "market_trend_4wk_pct": mk["trend_4wk_pct"],
                               "loads_per_truck": mk["loads_per_truck"], "floor_rate": floor, "floor_rpm": round(floor / miles, 2),
                               "target_ask": target, "diesel_origin": fuel["diesel"] if fuel else None,
                               "accessorials": f"detention {self.policy['detention']}/h after 2 free hours, lumper reimbursed, "
                                               f"TONU {self.policy['tonu']}, extra stop {self.policy['extra_stop']}"}
        if ours:
            out["our_last_loads_on_lane"] = {"count": len(ours), "avg_rpm": round(sum(r["rate"] / r["miles"] for r in ours) / len(ours), 2)}
        if posting is not None:
            b = self.q1("SELECT name, style FROM brokers WHERE id=?", (posting["broker_id"],))
            out.update({"posting": posting["id"], "broker": b["name"], "posted_rate": posting["posted_rate"] or "call",
                        "posted_rpm": round(posting["posted_rate"] / miles, 2) if posting["posted_rate"] else None})
            if posting["posted_rate"]:
                out["posted_vs_floor"] = posting["posted_rate"] - floor
        if truck_note:
            out["note"] = truck_note
        return _j(out)

    def broker_profile(self, name: str) -> str:
        b = self._broker(name)
        if b is None:
            return _j({"error": f"no broker matching {name!r}"})
        hist = self.q1("SELECT COUNT(*) AS n, AVG(rate * 1.0 / miles) AS rpm, MAX(pickup_appt) AS last FROM loads "
                       "WHERE broker_id=? AND pickup_appt >= ?", (b["id"], (self.now() - timedelta(days=365)).strftime("%Y-%m-%d")))
        pay = self.q1("SELECT AVG(julianday(paid_at) - julianday(invoiced_at)) AS days FROM loads WHERE broker_id=? AND paid_at IS NOT NULL",
                      (b["id"],))
        active = self.q("SELECT number, status FROM loads WHERE broker_id=? AND status IN ('booked','dispatched','in_transit','delayed')", (b["id"],))
        contacts = [f"{c['name']} ({c['desk']}, {c['phone']})" for c in self.q("SELECT * FROM broker_contacts WHERE broker_id=?", (b["id"],))]
        out = {"broker": b["name"], "mc": b["mc"], "status": b["status"].replace("_", " "), "credit": b["credit"],
               "terms_days": b["days_to_pay"], "quick_pay_fee_pct": b["quick_pay_pct"], "contacts": contacts,
               "loads_last_12_months": hist["n"], "avg_rpm_paid": round(hist["rpm"], 2) if hist["rpm"] else None,
               "last_load": self.when(hist["last"]) if hist["last"] else None,
               "avg_days_to_actually_pay": round(pay["days"], 1) if pay["days"] else None,
               "open_loads": [a["number"] for a in active]}
        if b["status_reason"]:
            out["why"] = b["status_reason"]
        return _j(out)

    def _broker(self, name: str) -> sqlite3.Row | None:
        t = name.strip()
        if re.fullmatch(r"(?i)mc ?\d{5,7}", t.replace("#", "")):
            return self.q1("SELECT * FROM brokers WHERE mc = ?", ("MC" + re.sub(r"\D", "", t),))
        rows = self.q("SELECT * FROM brokers")
        key = _norm(t)
        exact = [r for r in rows if _norm(r["name"]) == key]
        if exact:
            return exact[0]
        starts = [r for r in rows if _norm(r["name"]).startswith(key) or key in _norm(r["name"])]
        if starts:
            return starts[0]
        close = difflib.get_close_matches(key, [_norm(r["name"]) for r in rows], n=1, cutoff=0.6)
        return next((r for r in rows if _norm(r["name"]) == close[0]), None) if close else None

    def book_load(self, posting_id: str, truck: str, rate: int, contact: str = "", notes: str = "") -> str:
        p = self.q1("SELECT p.*, fo.city_id AS o_city, fd.city_id AS d_city FROM postings p JOIN facilities fo ON "
                    "fo.id=p.origin_id JOIN facilities fd ON fd.id=p.dest_id WHERE p.id = ?", (posting_id.upper().strip(),))
        if p is None:
            return _j({"booked": False, "error": f"no posting {posting_id}: search the load board for the broker's lane "
                                                 "and date to find the load's id (it looks like LB-123456), then book that"})
        if p["status"] != "open":
            return _j({"booked": False, "error": f"{p['id']} is already {p['status']}"})
        b = self.q1("SELECT * FROM brokers WHERE id=?", (p["broker_id"],))
        if b["status"] == "do_not_use":
            return _j({"booked": False, "error": f"{b['name']} is flagged do-not-use: {b['status_reason']}"})
        t = next((x for x in self.truck_positions() if x["unit"] == str(truck).strip().lstrip("#")), None)
        if t is None:
            return _j({"booked": False, "error": f"no truck {truck} with a driver"})
        if t["blocked"]:
            return _j({"booked": False, "error": f"truck {t['unit']}: {t['blocked']}"})
        if t["ttype"] != p["equipment"]:
            return _j({"booked": False, "error": f"truck {t['unit']} pulls a {t['ttype']}, the load needs a {p['equipment']}"})
        if p["hazmat"] and not t["hazmat"]:
            return _j({"booked": False, "error": f"{t['name']} has no hazmat endorsement"})
        if p["weight"] > int(self.policy["max_weight"]):
            return _j({"booked": False, "error": f"{p['weight']} lbs is over our {self.policy['max_weight']} limit"})
        oc = self.cities[p["o_city"]]
        deadhead = road_miles(t["lat_now"], t["lon_now"], oc["lat"], oc["lon"])
        if deadhead > int(self.policy["max_deadhead"]):
            return _j({"booked": False, "error": f"{deadhead} empty miles is over the {self.policy['max_deadhead']} limit; needs the owner's okay"})
        arrive = t["free_at"] + timedelta(hours=deadhead / AVG_MPH)
        pickup_end = datetime.strptime(p["pickup_end"], "%Y-%m-%d %H:%M")
        if arrive > pickup_end:
            return _j({"booked": False, "error": f"truck {t['unit']} can't reach the pickup before {self.when(pickup_end)} (earliest {self.when(arrive)})"})
        floor = self.floor(p["equipment"], p["miles"], deadhead)
        rate = int(round(float(rate)))
        if rate < floor:
            return _j({"booked": False, "error": f"{rate} is under our floor of {floor} for this load with {deadhead} empty miles"})
        n = self.q1("SELECT MAX(CAST(SUBSTR(number, 4) AS INT)) AS m FROM loads")["m"] + 1
        number = f"RO-{n}"
        now = self.now()
        with self.lock:
            self.db.execute("INSERT INTO loads VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                number, p["broker_id"], None, p["id"], p["origin_id"], p["dest_id"], p["pickup_start"], p["delivery_start"],
                None, None, t["did"], t["unit"], t["trailer"], p["equipment"], p["commodity"], p["weight"], p["temp_f"],
                p["miles"], deadhead, rate, 0, 0, "booked", None, None, notes or "", now.strftime("%Y-%m-%d %H:%M")))
            self.db.execute("UPDATE postings SET status='covered' WHERE id=?", (p["id"],))
            self.db.execute("UPDATE drivers SET status='on_load' WHERE id=?", (t["did"],))
            self.db.execute("INSERT INTO updates (at, load_number, kind, text) VALUES (?,?,?,?)",
                            (now.strftime("%Y-%m-%d %H:%M"), number, "booked", f"{b['name']} {contact} at {rate}".strip()))
            self.db.commit()
        return _j({"booked": True, "load_number": number, "broker": b["name"], "rate": rate, "rpm": round(rate / p["miles"], 2),
                   "truck": t["unit"], "driver": t["name"], "deadhead_miles": deadhead,
                   "pickup": f"{self.when(p['pickup_start'])}-{p['pickup_end'][-5:]}", "delivery": self.when(p["delivery_start"]),
                   "next": "rate confirmation requested from the broker by email; driver gets the load details by text"})

    def _find_load(self, ref: str) -> sqlite3.Row | None:
        r = ref.strip()
        digits = re.sub(r"\D", "", r)
        if digits:
            row = self.q1("SELECT * FROM loads WHERE number = ?", (f"RO-{digits}",))
            if row is None:
                row = self.q1("SELECT * FROM loads WHERE broker_ref = ? OR broker_ref = ? ORDER BY pickup_appt DESC LIMIT 1", (r, digits))
            if row is None and len(digits) >= 4:
                # a reference as a caller says it and a model writes it: "PHX 55120", "PHX55120", "pax 55,120"
                # for PHX-55120 (spoken calls, 2026-09-27: 6 of 12 models missed the load on the exact match)
                key = re.sub(r"[^a-z0-9]", "", r.lower())
                for cand in self.q("SELECT * FROM loads WHERE broker_ref GLOB ? ORDER BY pickup_appt DESC", (f"*{digits[-4:]}*",)):
                    ref_key = re.sub(r"[^a-z0-9]", "", (cand["broker_ref"] or "").lower())
                    if ref_key == key or re.sub(r"\D", "", ref_key) == digits:
                        row = cand
                        break
            if row is None and len(digits) <= 3:  # a truck unit
                row = self.q1("SELECT * FROM loads WHERE truck = ? ORDER BY pickup_appt DESC LIMIT 1", (digits,))
            if row is not None:
                return row
        d = self._driver(r)
        if d is not None:
            return self.q1("SELECT * FROM loads WHERE driver_id = ? ORDER BY pickup_appt DESC LIMIT 1", (d["id"],))
        return self.q1("SELECT * FROM loads WHERE broker_ref = ? ORDER BY pickup_appt DESC LIMIT 1", (r,))

    def load_status(self, reference: str) -> str:
        ld = self._find_load(reference)
        if ld is None:
            return _j({"error": f"no load found for {reference!r} (try the load number, the broker's reference, the truck or the driver)"})
        fo = self.q1("SELECT * FROM facilities WHERE id=?", (ld["origin_id"],))
        fd = self.q1("SELECT * FROM facilities WHERE id=?", (ld["dest_id"],))
        b = self.q1("SELECT name FROM brokers WHERE id=?", (ld["broker_id"],))
        d = self.q1("SELECT name, phone FROM drivers WHERE id=?", (ld["driver_id"],))
        out: dict[str, Any] = {"load": ld["number"], "broker": b["name"], "broker_ref": ld["broker_ref"], "status": ld["status"],
                               "from": f"{fo['name']}, {self.place(fo['city_id'])}", "to": f"{fd['name']}, {self.place(fd['city_id'])}",
                               "pickup_appt": self.when(ld["pickup_appt"]), "picked_up": self.when(ld["picked_up_at"]),
                               "delivery_appt": self.when(ld["delivery_appt"]), "driver": f"{d['name']} {d['phone']}" if d else None,
                               "truck": ld["truck"], "trailer": ld["trailer"],
                               "freight": f"{ld['weight']} lbs {ld['commodity']}" + (f" at {ld['temp_f']}F" if ld["temp_f"] is not None else ""),
                               "rate": ld["rate"]}
        down = self.q1("SELECT 1 FROM incidents WHERE load_number=? AND kind='breakdown' AND resolved=0", (ld["number"],))
        if ld["status"] in ("delivered", "invoiced", "paid"):
            out["delivered"] = self.when(ld["delivered_at"])
        elif down:
            e = self.eta(ld)
            out.update({"last_ping": f"near {self.place(e['ping_city'])}, {e['ping_min_ago']} min ago", "miles_to_go": e["miles_to_go"],
                        "eta": "none: the truck is broken down (see incidents); a recovery truck or a repair decides it",
                        "vs_appointment": "will miss it unless recovered"})
            if fd["notes"]:
                out["receiver_rules"] = fd["notes"]
        else:
            e = self.eta(ld)
            appt = datetime.strptime(ld["delivery_appt"], "%Y-%m-%d %H:%M")
            late = int((e["eta"] - appt).total_seconds() // 60)
            out.update({"last_ping": f"near {self.place(e['ping_city'])}, {e['ping_min_ago']} min ago" if e["ping_city"] else "not picked up yet",
                        "miles_to_go": e["miles_to_go"], "eta": self.when(e["eta"]),
                        "vs_appointment": f"{late} min late" if late > 10 else "on time" if late >= -30 else f"{-late} min early"})
            if e["breaks"]:
                out["hos_breaks_needed"] = e["breaks"]
            if fd["notes"]:
                out["receiver_rules"] = fd["notes"]
        inc = self.q("SELECT at, kind, detail FROM incidents WHERE load_number=? AND resolved=0", (ld["number"],))
        if inc:
            out["incidents"] = [f"{self.when(i['at'])} {i['kind']}: {i['detail']}" for i in inc]
        msgs = self.q("SELECT at, party, text FROM messages WHERE load_number=? ORDER BY at DESC LIMIT 3", (ld["number"],))
        if msgs:
            out["driver_messages"] = [f"{self.when(m['at'])} {m['party']}: {m['text']}" for m in msgs]
        ups = self.q("SELECT at, kind, text FROM updates WHERE load_number=? ORDER BY id DESC LIMIT 3", (ld["number"],))
        if ups:
            out["our_updates"] = [f"{self.when(u['at'])} {u['kind']}: {u['text']}" for u in ups]
        if ld["notes"]:
            out["notes"] = ld["notes"]
        return _j({k: v for k, v in out.items() if v is not None})

    def _driver(self, name: str) -> sqlite3.Row | None:
        t = name.strip()
        digits = re.sub(r"\D", "", t)
        if digits and len(digits) <= 3:
            return self.q1("SELECT d.* FROM drivers d JOIN trucks t ON t.driver_id = d.id WHERE t.unit = ?", (digits,))
        rows = self.q("SELECT d.* FROM drivers d JOIN trucks t ON t.driver_id = d.id")
        key = _norm(t)
        if not key:
            return None
        full = [r for r in rows if _norm(r["name"]) == key]
        if full:
            return full[0]
        part = [r for r in rows if key in _norm(r["name"]).split() or _norm(r["name"]).startswith(key)]
        return part[0] if len(part) == 1 else None

    def driver_info(self, name_or_truck: str) -> str:
        d = self._driver(name_or_truck)
        if d is None:
            return _j({"error": f"no single driver matches {name_or_truck!r}; give the full name or the truck number"})
        t = next((x for x in self.truck_positions() if x["did"] == d["id"]), None)
        cur = self.q1("SELECT number, status FROM loads WHERE driver_id=? AND status IN ('booked','dispatched','in_transit','delayed') "
                      "ORDER BY pickup_appt DESC LIMIT 1", (d["id"],))
        out = {"driver": d["name"], "phone": d["phone"], "status": d["status"].replace("_", " "), "home": self.place(d["home_city_id"]),
               "truck": t["unit"] if t else None, "trailer": f"{t['ttype']} {t['trailer']}" if t else None,
               "near": self.place(self.nearest_city(t["lat"], t["lon"])) if t else None,
               "current_load": f"{cur['number']} ({cur['status']})" if cur else None, "free": self.when(t["free_at"]) if t else None,
               "drive_h_left": d["drive_left_h"], "duty_h_left": d["duty_left_h"], "cycle_h_left": d["cycle_left_h"],
               "hazmat": bool(d["hazmat"]), "twic": bool(d["twic"]), "rating": d["rating"], "notes": d["notes"] or None}
        return _j({k: v for k, v in out.items() if v is not None})

    def facility_info(self, name: str, city: str = "") -> str:
        ids = set(self.resolve(city)) if city else None
        rows = self.q("SELECT * FROM facilities")
        key = _norm(name)
        cand = [r for r in rows if key in _norm(r["name"]) and (ids is None or r["city_id"] in ids)]
        if not cand:
            names = difflib.get_close_matches(key, [_norm(r["name"]) for r in rows], n=3, cutoff=0.6)
            cand = [r for r in rows if _norm(r["name"]) in names and (ids is None or r["city_id"] in ids)]
        if not cand:
            return _j({"error": f"no facility matching {name!r}"})
        out = [{"facility": r["name"], "city": self.place(r["city_id"]), "address": r["address"], "hours": f"{r['opens']}-{r['closes']}",
                "appointments": "required" if r["appointment_required"] else "first come first served",
                "avg_wait_min": r["avg_dwell_min"], "lumper": r["lumper_fee"] or None, "notes": r["notes"] or None} for r in cand[:4]]
        return _j({"matches": len(cand), "facilities": [{k: v for k, v in f.items() if v is not None} for f in out]})

    def estimate_trip(self, origin: str, destination: str, start: str = "now", drive_hours_left: float = 11.0) -> str:
        oi, di = self.resolve(origin), self.resolve(destination)
        if not oi or not di:
            return _j({"error": "need two places I know"})
        miles = self.miles_between(oi[0], di[0])
        t0 = self.now()
        if start and _norm(start) not in ("now", ""):
            m = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", start.lower())
            d = self.day(start.split()[0]) if start.split() else None
            if m:
                h = int(m.group(1)) % 12 + (12 if m.group(3) == "pm" else 0) if m.group(3) else int(m.group(1))
                t0 = (d or t0).replace(hour=h, minute=int(m.group(2) or 0))
            elif d:
                t0 = d.replace(hour=6)
        hours = miles / AVG_MPH
        breaks, left = 0, hours - float(drive_hours_left)
        while left > 0:
            breaks += 1
            left -= 11.0
        arrive = t0 + timedelta(hours=hours + 10 * breaks)
        return _j({"from": self.place(oi[0]), "to": self.place(di[0]), "miles": miles, "driving_hours": round(hours, 1),
                   "ten_hour_breaks": breaks, "leave": self.when(t0), "arrive": self.when(arrive)})

    def log_update(self, load_number: str, kind: str, text: str) -> str:
        ld = self._find_load(load_number)
        if ld is None:
            return _j({"logged": False, "error": f"no load {load_number}"})
        now = self.now()
        self.write("INSERT INTO updates (at, load_number, kind, text) VALUES (?,?,?,?)", (now.strftime("%Y-%m-%d %H:%M"), ld["number"], kind, text))
        return _j({"logged": True, "load": ld["number"], "at": self.when(now), "kind": kind})

    def notify_facility(self, load_number: str, which: str, request: str) -> str:
        """Email the load's shipper or receiver through the system; their answer comes later, never now."""
        ld = self._find_load(load_number)
        if ld is None:
            return _j({"sent": False, "error": f"no load {load_number}"})
        fid = ld["origin_id"] if _norm(which).startswith(("ship", "pick", "orig")) else ld["dest_id"]
        f = self.q1("SELECT name, city_id FROM facilities WHERE id=?", (fid,))
        now = self.now()
        self.write("INSERT INTO messages (at, direction, party, load_number, text) VALUES (?,?,?,?,?)",
                   (now.strftime("%Y-%m-%d %H:%M"), "out", f"{f['name']} (email)", ld["number"], request))
        self.write("INSERT INTO updates (at, load_number, kind, text) VALUES (?,?,?,?)",
                   (now.strftime("%Y-%m-%d %H:%M"), ld["number"], "facility request", f"to {f['name']}: {request}"))
        return _j({"sent": True, "to": f"{f['name']}, {self.place(f['city_id'])}", "load": ld["number"], "at": self.when(now),
                   "answer": "pending: they reply by email, usually within the hour"})

    def message_driver(self, driver_or_truck: str, text: str) -> str:
        d = self._driver(driver_or_truck)
        if d is None:
            return _j({"sent": False, "error": f"no single driver matches {driver_or_truck!r}"})
        t = self.q1("SELECT unit FROM trucks WHERE driver_id=?", (d["id"],))
        cur = self.q1("SELECT number FROM loads WHERE driver_id=? AND status IN ('booked','dispatched','in_transit','delayed') "
                      "ORDER BY pickup_appt DESC LIMIT 1", (d["id"],))
        now = self.now()
        self.write("INSERT INTO messages (at, direction, party, load_number, text) VALUES (?,?,?,?,?)",
                   (now.strftime("%Y-%m-%d %H:%M"), "out", f"{d['name']} (truck {t['unit'] if t else '?'})", cur["number"] if cur else None, text))
        return _j({"sent": True, "to": d["name"], "truck": t["unit"] if t else None, "at": self.when(now)})


# ---------------------------------------------------------------------------- tool list
_desk: Desk | None = None


def desk() -> Desk:
    global _desk
    if _desk is None:
        _desk = Desk()
    return _desk


def _s(desc: str, **props: Any) -> dict[str, Any]:
    required = [k for k, v in props.items() if v.pop("required", False)]
    return {"type": "object", "properties": props, "required": required}


def _str(desc: str, required: bool = False) -> dict[str, Any]:
    return {"type": "string", "description": desc, "required": required}


def _int(desc: str, required: bool = False) -> dict[str, Any]:
    return {"type": "integer", "description": desc, "required": required}


def dispatch_tools() -> list[Tool]:
    """The dispatcher's tools over the shared desk, plus hanging up."""
    from ...tools import end_conversation

    def call(method: str):
        def fn(**kw: Any) -> str:
            return getattr(desk(), method)(**kw)
        fn.__name__ = method
        return fn

    return [
        Tool("search_load_board", "Open loads posted by brokers, best rate per mile first, with each load's posting id. A broker "
             "calling about a load: search their lane and date to find it. Origin can be a city, a state or a direction; "
             "a city includes everything within radius_miles.",
             _s("", origin=_str("pickup city, state or area, e.g. 'Memphis' or 'Texas'", True),
                destination=_str("drop city, state or area like 'the Midwest'; empty for anywhere"),
                equipment=_str("dry van, reefer or flatbed; empty for any"), pickup_date=_str("today, tomorrow, a weekday or a date; empty for any"),
                radius_miles=_int("around a city, default 75"), limit=_int("how many to return, default 5")), call("search_load_board"),
             spoken_hint="Let me pull up the board."),
        Tool("find_available_trucks", "Our trucks that could take a load near a place: empty ones, ones finishing a load there, with "
             "empty miles to get there, driver hours left and endorsements; also the nearby ones that can't and why.",
             _s("", near=_str("the pickup city", True), equipment=_str("dry van, reefer or flatbed"),
                needed_by=_str("today, tomorrow, a weekday or a date"), radius_miles=_int("default 150"),
                hazmat={"type": "boolean", "description": "true when the load is hazmat"}, limit=_int("default 5")),
             call("find_available_trucks"), spoken_hint="Checking who's close."),
        Tool("price_load", "What a load should pay: loaded and empty miles, the lane's market rate, our history on the lane, our "
             "floor (never book below it) and the target to ask. Give the posting id, or origin and destination; add the truck for its empty miles.",
             _s("", posting_id=_str("load board id like LB-500101"), origin=_str("pickup city"), destination=_str("drop city"),
                equipment=_str("dry van, reefer or flatbed"), truck=_str("our truck number, for the empty miles")),
             call("price_load"), spoken_hint="Running the numbers."),
        Tool("broker_profile", "A broker's standing with us: do-not-use flag and why, credit, pay terms and how fast they really pay, our "
             "history with them, contacts.", _s("", name=_str("broker name or MC number", True)), call("broker_profile"),
             spoken_hint="Let me look them up."),
        Tool("book_load", "Book a load from the board on one of our trucks at an agreed rate; the posting id comes from "
             "search_load_board. It refuses below our floor, with a do-not-use "
             "broker, the wrong trailer, a driver without hazmat for hazmat, or a truck that can't make the pickup.",
             _s("", posting_id=_str("load board id", True), truck=_str("our truck number", True), rate=_int("agreed all-in rate in dollars", True),
                contact=_str("the broker's rep"), notes=_str("anything agreed, like detention or a lumper")),
             call("book_load"), spoken_hint="Booking it now."),
        Tool("load_status", "Where one of our loads is: status, last GPS ping, miles to go, ETA against the appointment, driver, "
             "receiver rules, incidents and the driver's messages. Find it by our load number, the broker's reference or PO, the truck or the driver.",
             _s("", reference=_str("load number, broker reference or PO, truck number or driver name", True)), call("load_status"),
             spoken_hint="Pulling it up."),
        Tool("driver_info", "One driver: status, where, hours left, endorsements, current load, phone, notes.",
             _s("", name_or_truck=_str("driver name or truck number", True)), call("driver_info"), spoken_hint="One sec."),
        Tool("facility_info", "A shipper or receiver: hours, appointment rules, usual wait, lumper, notes.",
             _s("", name=_str("facility name", True), city=_str("its city, if known")), call("facility_info"), spoken_hint="Checking."),
        Tool("estimate_trip", "Drive time between two places at 50 mph with the 10-hour breaks an 11-hour driving day needs.",
             _s("", origin=_str("from", True), destination=_str("to", True), start=_str("when they leave: now, or like 'tomorrow 6am'"),
                drive_hours_left={"type": "number", "description": "driving hours the driver has left today, default 11"}),
             call("estimate_trip"), spoken_hint="Let me work that out."),
        Tool("log_update", "Record something on a load: a new ETA, a delay, an appointment change, a note from a call.",
             _s("", load_number=_str("our load number", True), kind=_str("eta, delay, appointment, note", True), text=_str("what to record", True)),
             call("log_update")),
        Tool("notify_facility", "Email a load's shipper or receiver through the system, like an appointment change request. "
             "Their answer comes later by email, not during this call.",
             _s("", load_number=_str("our load number", True), which=_str("receiver or shipper", True),
                request=_str("what we're asking or telling them", True)), call("notify_facility")),
        Tool("message_driver", "Text one of our drivers.", _s("", driver_or_truck=_str("driver name or truck number", True),
                                                              text=_str("the message", True)), call("message_driver")),
        Tool("end_conversation", "Hang up after saying goodbye, when the caller is done.",
             {"type": "object", "properties": {"reason": {"type": "string", "description": "why"}}, "required": []},
             end_conversation, final=True),
    ]


__all__ = ["Desk", "desk", "dispatch_tools", "COMPANY"]
