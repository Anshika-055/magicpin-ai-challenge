"""magicpin challenge bot ("Vera"): context-grounded composition + multi-turn API.

Composition is trigger-routed: every trigger kind has a handler that anchors the
message on facts that exist in the four contexts (category, merchant, trigger,
customer), never invents offers/data, and ends on a single call-to-action.
An optional LLM pass (env-configured) can polish drafts; its output is only used
if it passes the same grounding checks, otherwise the deterministic draft ships.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.request as urlrequest
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="magicpin Vera bot", version="2.0.0")
START = time.time()
contexts: dict[tuple[str, str], dict] = {}
conversations: dict[str, dict] = {}
sent_suppressions: set[str] = set()
auto_reply_counts: dict[str, int] = {}   # merchant_id -> auto-replies seen across conversations

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# =============================================================================
# Small helpers
# =============================================================================
def get(scope: str, cid: Any) -> Optional[dict]:
    return contexts.get((scope, str(cid)), {}).get("payload")


def first(d: Any, *paths: str, default: Any = None) -> Any:
    for path in paths:
        cur = d
        try:
            for k in path.split("."):
                cur = cur[k]
            if cur is not None and cur != "" and cur != []:
                return cur
        except (KeyError, TypeError, IndexError):
            pass
    return default


def safe(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def human(s: Any) -> str:
    """snake_case field values -> readable words ("6_month_cleaning" -> "6-month cleaning")."""
    t = safe(s)
    t = re.sub(r"(\d+)_(month|day|week|year)", r"\1-\2", t)
    t = t.replace("_", " ")
    for dname in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"):
        t = re.sub(r"\b" + dname + r"\b", dname.capitalize(), t)
    return t


def num(n: Any) -> str:
    try:
        f = float(n)
    except (TypeError, ValueError):
        return safe(n)
    return f"{int(f):,}" if f == int(f) else f"{f:,.1f}"


def pct(v: Any, signed: bool = False) -> str:
    try:
        f = float(v) * 100
    except (TypeError, ValueError):
        return ""
    s = f"{abs(f):.0f}%"
    if signed:
        s = ("+" if f >= 0 else "-") + s
    return s


def inr(v: Any) -> str:
    try:
        return "₹" + num(v)
    except Exception:
        return safe(v)


def parse_dt(s: Any) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None


def fmt_date(s: Any, with_day: bool = False, with_year: bool = False) -> str:
    d = parse_dt(s)
    if not d:
        return safe(s)
    out = f"{d.day} {MONTHS[d.month - 1]}"
    if with_year:
        out += f" {d.year}"
    if with_day:
        out = f"{DAYS[d.weekday()]} {out}"
    return out


def fmt_time(s: Any) -> str:
    d = parse_dt(s)
    if not d:
        return ""
    h = d.hour % 12 or 12
    ap = "am" if d.hour < 12 else "pm"
    return f"{h}:{d.minute:02d}{ap}" if d.minute else f"{h}{ap}"


_ABBR = re.compile(r"\b(Dr|Mr|Mrs|Ms|vs|No|approx|[A-Z])\.$")


def sentences(text: str, n: int = 2) -> str:
    parts: list[str] = []
    for chunk in re.split(r"(?<=[.!?])\s+", safe(text)):
        if parts and _ABBR.search(parts[-1]):
            parts[-1] += " " + chunk          # "Dr. R. Mehta" must not be split
        else:
            parts.append(chunk)
    return " ".join(parts[:n]).strip()


def pretty_dates(text: str) -> str:
    """Inline ISO dates (2026-12-15) -> '15 Dec 2026' so no machine formats reach the merchant."""
    return re.sub(r"\b(20\d\d)-(\d\d)-(\d\d)\b",
                  lambda mm: f"{int(mm.group(3))} {MONTHS[int(mm.group(2)) - 1]} {mm.group(1)}", text)


def stale_days(sig: str) -> str:
    return re.sub(r"^(\d+)d$", r"\1 days", str(sig).split(":")[-1])


def lc_first(s: str) -> str:
    return s[:1].lower() + s[1:] if s and not s[:2].isupper() else s


def make_key(trg: dict) -> str:
    return str(trg.get("suppression_key") or trg.get("id")
               or hashlib.sha1(json.dumps(trg, sort_keys=True).encode()).hexdigest()[:12])


# =============================================================================
# Merchant / category accessors (all grounded — nothing falls back to fiction)
# =============================================================================
def is_placeholder(payload: dict) -> bool:
    return bool(payload.get("placeholder"))


def merchant_offers(merchant: dict) -> list[str]:
    """ONLY the merchant's own active offers. The category catalog is never presented as theirs."""
    return [safe(o.get("title")) for o in merchant.get("offers") or []
            if str(o.get("status", "active")).lower() == "active" and o.get("title")]


def catalog_offer(category: dict, *words: str) -> str:
    """A catalog offer to *suggest* (framed as a suggestion, never as the merchant's live offer)."""
    cat = [safe(o.get("title")) for o in category.get("offer_catalog") or [] if o.get("title")]
    for w in words:
        for t in cat:
            if w.lower() in t.lower():
                return t
    return ""


def uses_hindi(merchant: dict, category: dict) -> bool:
    langs = [str(x).lower() for x in first(merchant, "identity.languages", default=[]) or []]
    mix = str(first(category, "voice.code_mix", default="")).lower()
    return "hi" in langs and "hindi_english" in mix


def salutation(merchant: dict, category: dict, hi: bool) -> str:
    ident = merchant.get("identity") or {}
    owner = safe(ident.get("owner_first_name") or "")
    slug = merchant.get("category_slug") or category.get("slug", "")
    if slug == "dentists":
        if owner:
            return owner if owner.lower().startswith("dr") else f"Dr. {owner}"
        return safe(ident.get("name")) or "Doctor"
    if owner:
        return f"{owner} ji" if hi and slug in ("pharmacies", "restaurants") else owner
    return f"{safe(ident.get('name')) or 'there'} team"


def cta(hi: bool, en: str, hi_text: str = "", reply: str = "Reply YES") -> str:
    q = hi_text if (hi and hi_text) else en
    return f"{q} {reply}." if reply else q


def peer(category: dict, key: str) -> Any:
    return first(category, f"peer_stats.{key}")


def locality(merchant: dict) -> str:
    return safe(first(merchant, "identity.locality", "identity.city", default=""))


def digest_item(category: dict, trigger: dict) -> Optional[dict]:
    """Resolve the digest item a trigger points at. No 'latest item' guessing."""
    payload = trigger.get("payload") or {}
    digest = category.get("digest") or []
    item_id = first(payload, "top_item_id", "digest_item_id", "item_id", "digest_id", "alert_id")
    item = next((x for x in digest if x.get("id") == item_id), None) if item_id else None
    inline = payload.get("top_item") if isinstance(payload.get("top_item"), dict) else None
    if inline:
        item = {**(item or {}), **inline}
    if item:
        return item
    # Fall back only to an item whose *kind* matches the trigger kind (e.g. a compliance trigger -> compliance item)
    kind = str(trigger.get("kind", "")).lower()
    want = {"research": "research", "regulation": "compliance", "compliance": "compliance", "cde": "cde",
            "supply": "alert", "alert": "alert", "trend": "trend", "tech": "tech"}
    for k, v in want.items():
        if k in kind:
            matches = [x for x in digest if x.get("kind") == v]
            if len(matches) == 1:
                return matches[0]
    if ("research" in kind or "digest" in kind) and digest:
        for pref in ("research", "trend", "tech", "seasonal"):
            hit = next((x for x in digest if x.get("kind") == pref), None)
            if hit:
                return hit
    return None


def matching_trend(category: dict, *words: str) -> Optional[dict]:
    for t in category.get("trend_signals") or []:
        q = str(t.get("query", "")).lower()
        if any(w and w.lower() in q for w in words):
            return t
    return None


def top_review(merchant: dict, sentiment: str = "pos") -> Optional[dict]:
    themes = [t for t in merchant.get("review_themes") or [] if t.get("sentiment") == sentiment]
    return max(themes, key=lambda t: t.get("occurrences_30d", 0)) if themes else None


def last_vera_body(merchant: dict) -> str:
    for turn in reversed(merchant.get("conversation_history") or []):
        if turn.get("from") == "vera":
            return safe(turn.get("body"))
    return ""


def merchant_said_yes(merchant: dict) -> str:
    """Last merchant message that expressed intent, e.g. 'Yes send me the list please'."""
    for turn in reversed(merchant.get("conversation_history") or []):
        if turn.get("from") == "merchant" and str(turn.get("engagement", "")).startswith("intent"):
            return safe(turn.get("body"))
    return ""


def strengths(merchant: dict, category: dict) -> list[str]:
    out = []
    perf = merchant.get("performance") or {}
    for key, label in (("views", "views"), ("calls", "calls")):
        mine, avg = perf.get(key), peer(category, f"avg_{key}_30d")
        if mine is not None and avg and float(mine) > 1.2 * float(avg):
            out.append(f"{num(mine)} {label} in 30 days vs ~{num(avg)} peer avg")
    return out


def relevant_trend(merchant: dict, category: dict) -> Optional[dict]:
    """Trend that matches this merchant's own offers/review themes; else the strongest category trend."""
    words = []
    for o in merchant_offers(merchant):
        words += [w for w in re.findall(r"[a-zA-Z]+", o) if len(w) > 3]
    for t in merchant.get("review_themes") or []:
        words += [w for w in human(t.get("theme", "")).split() if len(w) > 3]
    tr = matching_trend(category, *[w for w in words if w.lower() not in ("free", "with", "offer", "first", "month")])
    if tr:
        return tr
    ts = category.get("trend_signals") or []
    return max(ts, key=lambda x: x.get("delta_yoy", 0)) if ts else None


def extreme_delta(merchant: dict, sign: int) -> tuple[str, Any]:
    d = {k.replace("_pct", ""): v for k, v in (first(merchant, "performance.delta_7d", default={}) or {}).items()
         if isinstance(v, (int, float)) and k != "ctr_pct"}
    if not d:
        return "views", None
    k = (min if sign < 0 else max)(d, key=lambda x: d[x])
    return k, d[k]


def weak_spots(merchant: dict, category: dict) -> list[str]:
    """Concrete, verifiable gaps vs peers — used as the 'why it matters' line."""
    out = []
    perf = merchant.get("performance") or {}
    for key, label in (("views", "views"), ("calls", "calls")):
        mine, avg = perf.get(key), peer(category, f"avg_{key}_30d")
        if mine is not None and avg and float(mine) < 0.8 * float(avg):
            out.append(f"{num(mine)} {label} in 30 days vs ~{num(avg)} peer avg")
    ctr, avg_ctr = perf.get("ctr"), peer(category, "avg_ctr")
    if ctr is not None and avg_ctr and float(ctr) < float(avg_ctr):
        out.append(f"CTR {float(ctr) * 100:.1f}% vs {float(avg_ctr) * 100:.1f}% peer avg")
    return out


# =============================================================================
# Merchant-facing handlers — each returns (body, cta_type, rationale)
# =============================================================================
def h_digest(ctx: dict) -> tuple[str, str, str]:
    m, c, t, hi, sal = ctx["merchant"], ctx["category"], ctx["trigger"], ctx["hi"], ctx["sal"]
    payload = t.get("payload") or {}
    item = digest_item(c, t)
    kind = str(t.get("kind", "")).lower()
    if not item:
        return h_generic(ctx)
    ik = str(item.get("kind", "")).lower()
    title, source = safe(item.get("title")), safe(item.get("source"))
    summ = sentences(item.get("summary", ""), 2)
    agg = m.get("customer_aggregate") or {}
    parts = []
    if ik == "research" or (not ik and "research" in kind):
        n = item.get("trial_n")
        parts.append(f"{sal}, worth a look from {source}: {title}.")
        if summ:
            parts.append((f"{num(n)}-patient study — " + lc_first(summ)) if n else summ)
        seg = str(item.get("patient_segment", "")).lower()
        seg_root = seg.replace("_adults", "_adult").replace("adults", "adult")
        count_key = next((k for k in agg if seg_root and seg_root in k.lower()), None)
        if count_key:
            cnt = num(agg[count_key])
            parts.append(f"That's exactly your {cnt} {human(seg_root).replace('high risk', 'high-risk')} patients.")
            parts.append(cta(hi, f"Want me to pull the abstract and draft a recall WhatsApp for those {cnt} patients?",
                             f"Abstract + in {cnt} patients ke liye recall WhatsApp draft kar doon?"))
        else:
            parts.append(cta(hi, "Want me to pull the abstract and draft a patient WhatsApp on it?",
                             "Abstract + patient WhatsApp draft kar doon?"))
        return " ".join(parts), "binary_yes_no", "Research digest the trigger points at, with trial size/effect and the merchant's own matching patient cohort."
    if ik == "compliance" or (not ik and any(x in kind for x in ("regulation", "compliance"))):
        deadline = first(payload, "deadline_iso", "deadline") or first(item, "effective_date")
        parts.append(f"{sal}, compliance heads-up ({source}): {title}.")
        if summ:
            parts.append(summ)
        if item.get("actionable"):
            parts.append(f"Action: {safe(item['actionable']).rstrip('.')}.")
        d = fmt_date(deadline, with_year=True) if deadline else ""
        parts.append(cta(hi, f"Want a 5-point audit checklist so you're covered{' before ' + d if d else ''}?",
                         f"{d + ' se pehle ' if d else ''}5-point audit checklist bhej doon?"))
        return " ".join(parts), "binary_yes_no", "Regulatory change with the exact limit and deadline from the digest; offers to externalise the audit effort."
    if ik == "cde" or (not ik and "cde" in kind):
        when = item.get("date")
        credits = first(payload, "credits") or item.get("credits")
        fee = human(first(payload, "fee", default=""))
        head = f"{sal}, {title}"
        if when:
            head += f" — {fmt_date(when, with_day=True)}, {fmt_time(when)}"
        parts.append(head + ".")
        if summ:
            parts.append(summ)
        bits = []
        if credits:
            bits.append(f"{credits} CDE credits")
        if item.get("actionable"):
            bits.append(safe(item["actionable"]).rstrip("."))
        elif fee:
            bits.append(fee)
        if bits:
            parts.append("; ".join(bits) + ".")
        parts.append(cta(hi, "Want me to send you the registration details?", "Registration details bhej doon?"))
        return " ".join(parts), "binary_yes_no", "CDE event from the digest with date, speaker, credits and fee; low-effort registration CTA."
    # trend / tech / seasonal / generic digest item
    parts.append(f"{sal}, {title} ({source}).")
    if summ:
        parts.append(summ)
    if item.get("actionable"):
        parts.append(f"Suggested move: {lc_first(safe(item['actionable']).rstrip('.'))}.")
    parts.append(cta(hi, "Want me to draft how this would look on your listing?", "Aapki listing ke liye draft bana doon?"))
    return " ".join(parts), "binary_yes_no", "Category digest item referenced by the trigger, with source and a concrete suggested action."


def h_perf_dip(ctx: dict) -> tuple[str, str, str]:
    m, c, t, hi, sal = ctx["merchant"], ctx["category"], ctx["trigger"], ctx["hi"], ctx["sal"]
    p = t.get("payload") or {}
    perf = m.get("performance") or {}
    if p.get("metric"):
        metric, delta = safe(p["metric"]), p.get("delta_pct", first(perf, f"delta_7d.{p['metric']}_pct"))
    else:
        metric, delta = extreme_delta(m, -1)
    name = safe(first(m, "identity.name"))
    parts = [f"{sal}, {name}'s {metric} are down {pct(delta)} this week"]
    if p.get("vs_baseline") is not None:
        parts[0] += f" (vs baseline {num(p['vs_baseline'])})"
    parts[0] += "."
    gaps = weak_spots(m, c)
    if gaps:
        parts.append("For context: " + "; ".join(gaps[:2]) + ".")
    fixes = []
    if first(m, "identity.verified") is False:
        fixes.append("the Google profile is still unverified")
    if not merchant_offers(m):
        fixes.append("there's no active offer live")
    stale = next((s for s in m.get("signals") or [] if str(s).startswith("stale_posts")), None)
    if stale:
        fixes.append(f"last Google post was {stale_days(stale)} ago")
    if fixes:
        parts.append("Fixable gaps: " + (", ".join(fixes[:-1]) + " and " + fixes[-1] if len(fixes) > 1 else fixes[0]) + ".")
    sugg = catalog_offer(c, "Cleaning", "Haircut", "Trial", "Thali", "Delivery", "Consultation")
    if not merchant_offers(m) and sugg:
        parts.append(cta(hi, f"Want me to put a '{sugg}' offer live to recover {metric}?",
                         f"'{sugg}' offer live kar doon taaki {metric} wapas aayein?"))
    else:
        parts.append(cta(hi, "Want me to run a quick listing fix-up to recover this?", "Listing ka quick fix-up kar doon?"))
    return " ".join(parts), "binary_yes_no", "Performance dip with magnitude/window from the trigger, peer benchmarks and the specific fixable gaps in this merchant's profile."


def h_perf_spike(ctx: dict) -> tuple[str, str, str]:
    m, c, t, hi, sal = ctx["merchant"], ctx["category"], ctx["trigger"], ctx["hi"], ctx["sal"]
    p = t.get("payload") or {}
    perf = m.get("performance") or {}
    if p.get("metric"):
        metric, delta = safe(p["metric"]), p.get("delta_pct", first(perf, f"delta_7d.{p['metric']}_pct"))
    else:
        metric, delta = extreme_delta(m, +1)
    driver = human(p.get("likely_driver", ""))
    if driver:
        head = f"{sal}, your {driver} is working — {metric} are up {pct(delta)} this week"
    else:
        head = f"{sal}, {metric} are up {pct(delta)} this week"
    if p.get("vs_baseline") is not None:
        head += f" (baseline {num(p['vs_baseline'])})"
    if p.get("vs_baseline") is None and perf.get(metric) is not None:
        head += f" ({num(perf[metric])} in the last 30 days)"
    parts = [head + "."]
    good = strengths(m, c)
    if good and not driver:
        parts.append("You're ahead of peers: " + "; ".join(good[:2]) + ".")
    words = driver.split() if driver else []
    tr = matching_trend(c, *[w for w in words if len(w) > 3]) if words else relevant_trend(m, c)
    if tr:
        parts.append(f"Demand is there: '{tr['query']}' searches are {pct(tr.get('delta_yoy'), True)} YoY.")
    parts.append(cta(hi, "Want me to publish a follow-up post while interest is high?",
                     "Interest high hai — follow-up post daal doon?"))
    return " ".join(parts), "binary_yes_no", "Positive spike with the metric, magnitude and likely driver from the trigger; momentum-based follow-up."


def h_seasonal_dip(ctx: dict) -> tuple[str, str, str]:
    m, c, t, hi, sal = ctx["merchant"], ctx["category"], ctx["trigger"], ctx["hi"], ctx["sal"]
    p = t.get("payload") or {}
    perf, agg = m.get("performance") or {}, m.get("customer_aggregate") or {}
    metric = safe(p.get("metric")) or "views"
    parts = [f"{sal}, {metric} are down {pct(p.get('delta_pct'))} this week — that's the expected {human(p.get('season_note', 'seasonal')).replace('post resolution window', 'post-resolution').replace(' apr jun', ' Apr-Jun')} dip, not a listing problem."]
    beat = next((b for b in c.get("seasonal_beats") or [] if "retention" in str(b.get("note", "")).lower() or "lowest" in str(b.get("note", "")).lower()), None)
    ctr, avg = perf.get("ctr"), peer(c, "avg_ctr")
    if ctr is not None and avg and float(ctr) >= float(avg):
        parts.append(f"Your CTR is still {float(ctr) * 100:.1f}% vs {float(avg) * 100:.1f}% peer avg.")
    churn, pchurn, members = agg.get("monthly_churn_pct"), peer(c, "monthly_churn_pct"), agg.get("total_active_members")
    if churn is not None and members:
        line = f"The bigger lever now is retention: monthly churn is {pct(churn)}"
        if pchurn:
            line += f" vs {pct(pchurn)} peer"
        line += f" — about {round(float(churn) * float(members))} of your {num(members)} members a month."
        parts.append(line)
    elif beat:
        parts.append(f"Category pattern: {beat['note']}.")
    parts.append(cta(hi, "Want me to draft a check-in for current members instead of spending on ads?",
                     "Ads ke bajaye current members ke liye check-in draft kar doon?"))
    return " ".join(parts), "binary_yes_no", "Reframes an expected seasonal dip with peer CTR and churn numbers; redirects effort to retention per category seasonality."


def h_renewal(ctx: dict) -> tuple[str, str, str]:
    m, c, t, hi, sal = ctx["merchant"], ctx["category"], ctx["trigger"], ctx["hi"], ctx["sal"]
    p = t.get("payload") or {}
    sub = m.get("subscription") or {}
    days = p.get("days_remaining", sub.get("days_remaining"))
    plan = p.get("plan") or sub.get("plan")
    amt = p.get("renewal_amount")
    head = f"{sal}, your {plan + ' ' if plan else ''}plan renews in {days} days" if days is not None else f"{sal}, your plan renewal is coming up"
    if amt:
        head += f" ({inr(amt)})"
    parts = [head + "."]
    perf = m.get("performance") or {}
    dips = [f"{k.replace('_pct', '')} {pct(v)}" for k, v in (perf.get("delta_7d") or {}).items() if isinstance(v, (int, float)) and v < -0.1 and k != "ctr_pct"]
    if dips:
        parts.append("Timing matters: " + " and ".join(dips[:2]).replace(" ", " down ", 1) + " this week" +
                     (", and the Google profile is still unverified" if first(m, "identity.verified") is False else "") +
                     " — a lapse would pause profile upkeep right when you need the recovery.")
    else:
        parts.append(f"Last 30 days: {num(perf.get('views', 0))} views, {num(perf.get('calls', 0))} calls — renewing keeps that visibility running without a gap.")
    parts.append(cta(hi, "Want me to send the renewal link?", "Renewal link bhej doon?"))
    return " ".join(parts), "binary_yes_no", "Renewal with exact days, plan and amount; loss-aversion anchored on the merchant's current performance trend."


def h_winback(ctx: dict) -> tuple[str, str, str]:
    m, c, t, hi, sal = ctx["merchant"], ctx["category"], ctx["trigger"], ctx["hi"], ctx["sal"]
    p = t.get("payload") or {}
    sub, agg = m.get("subscription") or {}, m.get("customer_aggregate") or {}
    name = safe(first(m, "identity.name"))
    days = p.get("days_since_expiry", sub.get("days_since_expiry"))
    parts = [f"{sal}, it's been {days} days since {name}'s plan lapsed." if days is not None else f"{sal}, {name}'s plan has lapsed."]
    facts = []
    dip = p.get("perf_dip_pct", first(m, "performance.delta_7d.calls_pct"))
    if dip is not None and float(dip) < 0:
        facts.append(f"calls are down {pct(dip)}")
    if p.get("lapsed_customers_added_since_expiry"):
        facts.append(f"{p['lapsed_customers_added_since_expiry']} more customers have slipped into the lapsed list")
    lapsed_key = next((k for k in agg if k.startswith("lapsed")), None)
    if facts:
        line = "Since then " + " and ".join(facts)
        if lapsed_key:
            line += f" ({num(agg[lapsed_key])} now haven't visited in {lapsed_key.split('_')[1].replace('d', '')}+ days)"
        parts.append(line + ".")
    parts.append("Reactivating restarts profile upkeep, and I can send those lapsed customers a comeback message in week one.")
    parts.append(cta(hi, "Want the reactivation steps?", "Reactivation steps bhej doon?"))
    return " ".join(parts), "binary_yes_no", "Win-back using days since expiry, performance drop and lapsed-customer growth from the trigger + merchant aggregate."


def h_dormant(ctx: dict) -> tuple[str, str, str]:
    m, c, t, hi, sal = ctx["merchant"], ctx["category"], ctx["trigger"], ctx["hi"], ctx["sal"]
    p = t.get("payload") or {}
    agg = m.get("customer_aggregate") or {}
    days = p.get("days_since_last_merchant_message")
    parts = [f"{sal}, it's been {days} days since we last spoke." if days else f"{sal}, quick one."]
    lapsed_key = next((k for k in agg if k.startswith("lapsed")), None)
    total = agg.get("total_unique_ytd")
    if lapsed_key and total:
        share = float(agg[lapsed_key]) / float(total)
        parts.append(f"One number worth a look: {num(agg[lapsed_key])} of your {num(total)} customers this year haven't visited in {lapsed_key.split('_')[1].replace('d', '')}+ days ({share * 100:.0f}%).")
        parts.append(cta(hi, "Want me to draft a comeback message for them?", "Unke liye comeback message draft kar doon?"))
        return " ".join(parts), "binary_yes_no", "Re-engages a dormant merchant with a concrete loss-aversion number from their own customer aggregate."
    return h_curious(ctx)


def h_curious(ctx: dict) -> tuple[str, str, str]:
    m, c, t, hi, sal = ctx["merchant"], ctx["category"], ctx["trigger"], ctx["hi"], ctx["sal"]
    slug = m.get("category_slug") or c.get("slug", "")
    perf = m.get("performance") or {}
    ask = {"restaurants": "which dish are customers asking for most this week?",
           "dentists": "which treatment are patients asking about most this week?",
           "salons": "which service is getting the most asks this week?",
           "gyms": "which class or membership question is coming up most this week?",
           "pharmacies": "which medicine or health question is coming up most at the counter this week?"}.get(slug, "what are customers asking for most this week?")
    parts = [f"{sal}, quick one — {ask}"]
    calls, dc = perf.get("calls"), first(perf, "delta_7d.calls_pct")
    rev = top_review(m, "pos")
    ev = []
    if calls is not None and dc is not None and float(dc) > 0:
        ev.append(f"calls are up {pct(dc)} this week ({num(calls)} in 30 days)")
    if rev and rev.get("occurrences_30d"):
        q = f" ('{rev['common_quote']}')" if rev.get("common_quote") else ""
        ev.append(f"{rev['occurrences_30d']} reviews this month praise {human(rev['theme'])}{q}")
    if ev:
        parts.append("Asking because " + " and ".join(ev) + ".")
    tr = relevant_trend(m, c)
    if tr:
        parts.append(f"Whatever it is, I'll turn it into a Google post today — e.g. '{tr['query']}' searches are {pct(tr.get('delta_yoy'), True)} YoY.")
    parts.append("Just reply with the name.")
    return " ".join(parts), "open_ended", "Curiosity/asking-the-merchant lever, grounded in their call trend, review themes and a category search trend."


def h_milestone(ctx: dict) -> tuple[str, str, str]:
    m, c, t, hi, sal = ctx["merchant"], ctx["category"], ctx["trigger"], ctx["hi"], ctx["sal"]
    p = t.get("payload") or {}
    name = safe(first(m, "identity.name"))
    metric = human(p.get("metric", "")) or "reviews"
    now, target = p.get("value_now"), p.get("milestone_value")
    if now is not None and target is not None and float(target) > float(now):
        head = f"{sal}, {name} is {num(float(target) - float(now))} {metric.replace(' count', 's')} away from {num(target)} ({num(now)} now)"
    elif now is not None:
        head = f"{sal}, {name} just crossed {num(now)} {metric.replace(' count', 's')}"
    else:
        perf = m.get("performance") or {}
        head = f"{sal}, {name} has hit a new milestone — {num(perf.get('views', 0))} views and {num(perf.get('calls', 0))} calls in the last 30 days" if perf.get("views") else f"{sal}, {name} has hit a new milestone"
    avg = peer(c, "avg_review_count") if "review" in metric else None
    if avg and now is not None and float(now) > float(avg):
        head += f" — already above the {num(avg)} average for similar businesses"
    parts = [head + "."]
    agg = m.get("customer_aggregate") or {}
    rev = top_review(m, "pos")
    ev = []
    if agg.get("repeat_customer_pct"):
        ev.append(f"{pct(agg['repeat_customer_pct'])} of your customers are repeat")
    if rev and rev.get("occurrences_30d"):
        ev.append(f"{rev['occurrences_30d']} reviews this month praise {human(rev['theme'])}")
    if ev:
        parts.append("Your regulars will get you there: " + " and ".join(ev) + ".")
    parts.append(cta(hi, "Want me to draft a short 'review us' note for your regulars?", "Regulars ke liye short 'review us' note draft kar doon?"))
    return " ".join(parts), "binary_yes_no", "Milestone with exact current vs target values, peer benchmark and the merchant's repeat-customer base as the path to it."


def h_review_theme(ctx: dict) -> tuple[str, str, str]:
    m, c, t, hi, sal = ctx["merchant"], ctx["category"], ctx["trigger"], ctx["hi"], ctx["sal"]
    p = t.get("payload") or {}
    theme = p.get("theme")
    rt = next((x for x in m.get("review_themes") or [] if x.get("theme") == theme), None) if theme else None
    if not theme:
        rt = next((x for x in m.get("review_themes") or [] if x.get("sentiment") == "neg"), None)
        theme = rt.get("theme") if rt else None
    if not theme:
        avg = peer(c, "avg_review_count")
        parts = [f"{sal}, a pattern is starting to show up in {safe(first(m, 'identity.name'))}'s recent reviews."]
        parts.append("Quick check from your side — what's the one thing customers mention most, good or bad?")
        if avg:
            parts.append(f"Businesses like yours average {num(avg)} reviews, and replying to each one visibly lifts trust.")
        parts.append("Tell me the theme and I'll draft replies you can post today.")
        return " ".join(parts), "open_ended", "Review-theme trigger without theme details: asks the merchant instead of inventing a theme; peer review benchmark as context."
    n = p.get("occurrences_30d") or (rt or {}).get("occurrences_30d")
    quote = p.get("common_quote") or (rt or {}).get("common_quote")
    head = f"{sal}, {n} reviews in the last 30 days mention {human(theme).replace('delivery late', 'late delivery')}" if n else f"{sal}, recent reviews keep mentioning {human(theme)}"
    if p.get("trend") == "rising":
        head += ", and it's rising"
    parts = [head + (f" — one says '{quote}'." if quote else ".")]
    pos = top_review(m, "pos")
    if pos and pos.get("theme") != theme and pos.get("occurrences_30d"):
        parts.append(f"{human(pos['theme']).capitalize()} is still your strongest theme ({pos['occurrences_30d']} positive reviews), so this is fixable without touching what works.")
    parts.append(cta(hi, "Want me to draft a polite public reply to these reviews plus a fix note for your team?",
                     "In reviews ka polite public reply + team ke liye fix note draft kar doon?"))
    return " ".join(parts), "binary_yes_no", "Emerging review theme with count, trend and verbatim quote from the trigger, balanced against the merchant's strongest positive theme."


def h_competitor(ctx: dict) -> tuple[str, str, str]:
    m, c, t, hi, sal = ctx["merchant"], ctx["category"], ctx["trigger"], ctx["hi"], ctx["sal"]
    p = t.get("payload") or {}
    comp = safe(p.get("competitor_name"))
    if not comp:
        gaps = weak_spots(m, c)
        parts = [f"{sal}, a new {c.get('display_name', 'business') if c.get('display_name') else 'competitor'} listing has appeared near {locality(m)}."]
        good = strengths(m, c)
        if gaps:
            parts.append("Where you stand today: " + "; ".join(gaps[:2]) + ".")
        elif good:
            parts.append("You're ahead today — " + "; ".join(good[:2]) + " — and a fresh post keeps it that way.")
        parts.append(cta(hi, "Want me to tighten your listing this week so you stay ahead?", "Is hafte listing tight kar doon taaki aap aage rahein?"))
        return " ".join(parts), "binary_yes_no", "Competitor-opened trigger without named details; avoids inventing a name and anchors on the merchant's own benchmarks."
    head = f"{sal}, {comp} opened {p.get('distance_km')} km from you" if p.get("distance_km") is not None else f"{sal}, {comp} has opened nearby"
    if p.get("opened_date"):
        head += f" on {fmt_date(p['opened_date'])}"
    theirs = safe(p.get("their_offer"))
    mine = merchant_offers(m)
    if theirs:
        head += f", listing {theirs}"
        same = next((o for o in mine if o.split("@")[0].strip().lower() == theirs.split("@")[0].strip().lower()), None)
        if same:
            head += f" vs your {same.split('@')[-1].strip()}"
    parts = [head + "."]
    pos = top_review(m, "pos")
    if pos and pos.get("common_quote"):
        parts.append(f"Don't race them on price — lean on what they can't copy: {pos['occurrences_30d']} reviews this month say '{pos['common_quote']}'.")
    stale = next((s for s in m.get("signals") or [] if str(s).startswith("stale_posts")), None)
    if stale:
        parts.append(f"Your last Google post was {stale_days(stale)} ago, so theirs will look fresher right now.")
    parts.append(cta(hi, "Want me to draft a post highlighting that this week?", "Is hafte uspe ek post draft kar doon?"))
    return " ".join(parts), "binary_yes_no", "Named competitor with distance, date and price from the trigger; differentiates on the merchant's real review strengths, not price."


def h_ipl(ctx: dict) -> tuple[str, str, str]:
    m, c, t, hi, sal = ctx["merchant"], ctx["category"], ctx["trigger"], ctx["hi"], ctx["sal"]
    p = t.get("payload") or {}
    agg = m.get("customer_aggregate") or {}
    when = p.get("match_time_iso")
    head = f"{sal}, {p.get('match', 'the match')} tonight"
    if when:
        head += f" at {fmt_time(when)}"
    if p.get("venue"):
        head += f", {p['venue']}"
    parts = [head + "."]
    item = next((x for x in c.get("digest") or [] if "ipl" in str(x.get("id", "")).lower() or "ipl" in str(x.get("title", "")).lower()), None)
    weeknight = p.get("is_weeknight")
    if item and weeknight is False:
        parts.append(f"It's a weekend match — {safe(item.get('source')).replace(', ', ' (', 1) + ')' if ', ' in safe(item.get('source')) else safe(item.get('source'))} shows weekend IPL games shift orders to home-watch parties (dine-in covers -12%), while weeknight matches drive +18%." if "12%" in str(item.get("summary")) else f"Heads-up from {safe(item.get('source'))}: {sentences(item.get('summary', ''), 1)}")
    offers = merchant_offers(m)
    restricted = next((o for o in offers if re.search(r"\((mon|tue|wed|thu|fri|sat|sun)", o.lower())), None)
    if restricted and weeknight is False:
        parts.append(f"Your '{restricted}' won't cover tonight.")
    dl, di = agg.get("delivery_orders_30d"), agg.get("dine_in_orders_30d")
    if dl and di:
        parts.append(f"And {num(dl)} of your last {num(dl + di)} orders were delivery.")
    combo = catalog_offer(c, "Match-night", "Match")
    if combo:
        parts.append(cta(hi, f"Want me to put up a delivery-first '{combo}' post before 6pm?", f"6pm se pehle delivery-first '{combo}' post daal doon?"))
    else:
        parts.append(cta(hi, "Want me to put up a match-night post before 6pm?", "6pm se pehle match-night post daal doon?"))
    return " ".join(parts), "binary_yes_no", "Match-day trigger (teams, time, venue) + category data on weekend vs weeknight matches + the merchant's delivery share and offer validity."


def h_festival(ctx: dict) -> tuple[str, str, str]:
    m, c, t, hi, sal = ctx["merchant"], ctx["category"], ctx["trigger"], ctx["hi"], ctx["sal"]
    p = t.get("payload") or {}
    fest = safe(p.get("festival") or p.get("event_name"))
    beat = next((b for b in c.get("seasonal_beats") or [] if "festival" in str(b.get("note", "")).lower() or "diwali" in str(b.get("note", "")).lower()), None)
    offers = merchant_offers(m)
    if fest:
        head = f"{sal}, {fest} is on {fmt_date(p['date'])}" if p.get("date") else f"{sal}, {fest} is coming up"
        if p.get("days_until"):
            head += f" — {p['days_until']} days out"
        parts = [head + "."]
    else:
        parts = [f"{sal}, festive season is the next big window for {c.get('display_name') or 'your category'}."]
    if beat:
        parts.append(f"For {c.get('slug', 'your category')}, {beat['month_range']} is the {beat['note'].split(' — ')[0]}" + (f" — {beat['note'].split(' — ')[1]}." if " — " in beat["note"] else "."))
    if offers:
        parts.append(cta(hi, f"Want me to draft a {fest or 'festive'} pre-booking package around your {offers[-1]} so it's ready early?",
                         f"Aapke {offers[-1]} ke around {fest or 'festive'} pre-booking package draft kar doon?"))
    else:
        parts.append(cta(hi, f"Want me to draft a {fest or 'festive'} pre-booking offer so you're ready early?", f"{fest or 'Festive'} pre-booking offer draft kar doon?"))
    return " ".join(parts), "binary_yes_no", "Festival date/lead time from the trigger, category seasonal volume pattern, and the merchant's real active offer as the anchor."


def h_planning(ctx: dict) -> tuple[str, str, str]:
    m, c, t, hi, sal = ctx["merchant"], ctx["category"], ctx["trigger"], ctx["hi"], ctx["sal"]
    p = t.get("payload") or {}
    topic = human(p.get("intent_topic", "")) or "your plan"
    words = [w for w in topic.split() if len(w) > 3]
    offers = merchant_offers(m)
    base = next((o for o in offers if any(w.lower() in o.lower() for w in words)), None)
    prev = last_vera_body(m)
    tr = matching_trend(c, *words)
    parts = [f"{sal}, here's a first cut of the {topic}:"]
    bullets = []
    nums_in_prev = re.findall(r"(\d+-week[^,.]*|\d+ classes/week|age \d+-\d+|₹[\d,]+|\d+ orders/day[^.,?]*)", prev)
    if base:
        per_day = re.search(r"(\d+) orders/day", prev)
        bullets.append(f"built on your {base}" + (f" (already ~{per_day.group(1)} orders/day)" if per_day else ""))
    if nums_in_prev and not base:
        bullets.append("format as we discussed: " + ", ".join(dict.fromkeys(nums_in_prev)))
    if "corporate" in topic or "bulk" in topic:
        bullets.append(f"bulk weekday orders for offices around {locality(m)}, pre-order by 11am, one delivery slot")
        bullets.append("price per plate at or below your current thali so office admins say yes fast" if base and "thali" in base.lower() else "a per-plate bulk price you set")
    elif "kids" in topic or "camp" in topic:
        bullets.append("small batches, parent WhatsApp updates after each week")
    if tr:
        bullets.append(f"demand check: '{tr['query']}' searches are {pct(tr.get('delta_yoy'), True)} YoY" + (f" ({human(tr.get('segment_age'))})" if tr.get("segment_age") and tr.get("segment_age") != "all" else ""))
    parts.append(" • " + " • ".join(bullets) + ".") if bullets else None
    parts.append(cta(hi, "Shall I turn this into a Google post + WhatsApp flyer for your approval?", "Isko Google post + WhatsApp flyer bana doon approval ke liye?"))
    return " ".join(x for x in parts if x), "binary_yes_no", "Merchant already asked 'what would it look like' — responds with a concrete draft (their offer, prior agreed format, demand data) instead of re-qualifying."


def h_supply(ctx: dict) -> tuple[str, str, str]:
    m, c, t, hi, sal = ctx["merchant"], ctx["category"], ctx["trigger"], ctx["hi"], ctx["sal"]
    p = t.get("payload") or {}
    item = digest_item(c, t) or {}
    mol = safe(p.get("molecule"))
    batches = p.get("affected_batches") or []
    parts = [f"{sal}, recall alert{' (' + safe(item.get('source')) + ')' if item.get('source') else ''} on {mol or 'a stocked medicine'}"]
    if batches:
        parts[0] += f": batches {' and '.join(batches)}" + (f" ({p['manufacturer']})" if p.get("manufacturer") else "") + " are flagged"
    parts[0] += "."
    if item.get("summary"):
        s = safe(item["summary"])
        risk = re.search(r"(No safety risk[^.]*\.)", s)
        repl = re.search(r"(replacement available[^.]*)", s, re.I)
        if "sub-potency" in s:
            parts.append("Reason: sub-potency." + (f" {risk.group(1)}" if risk else ""))
        if repl:
            parts.append(repl.group(1)[0].upper() + repl.group(1)[1:] + ".")
    agg = m.get("customer_aggregate") or {}
    said = merchant_said_yes(m)
    if agg.get("chronic_rx_count"):
        pre = f"You'd asked for the list ('{said}') — " if said else ""
        parts.append(f"{pre}I can pull the {mol} customers from your {num(agg['chronic_rx_count'])} chronic-Rx patients with a ready replacement message.")
    parts.append(cta(hi, "Send it now?", "Abhi bhej doon?"))
    return " ".join(parts), "binary_yes_no", "Urgent supply recall with molecule, batch numbers and manufacturer from the trigger; follows through on the merchant's earlier request."


def h_season_shift(ctx: dict) -> tuple[str, str, str]:
    m, c, t, hi, sal = ctx["merchant"], ctx["category"], ctx["trigger"], ctx["hi"], ctx["sal"]
    p = t.get("payload") or {}
    trends = p.get("trends") or []
    pretty = []
    for tr in trends:
        mm = re.match(r"(.+?)_demand_([+-]\d+)", str(tr))
        if mm:
            pretty.append(f"{human(mm.group(1)).replace('antifungal', 'anti-fungal').replace('cold cough', 'cold/cough')} {mm.group(2)}%")
        else:
            pretty.append(human(tr))
    ups = [x for x in pretty if "+" in x]
    downs = [x for x in pretty if "-" in x and "+" not in x]
    season = human(p.get("season", "")).replace(" 2026", "") or "seasonal"
    head = f"{sal}, the {season} demand shift has started: " + ", ".join(ups)
    if downs:
        head += f", while {', '.join(downs)}"
    parts = [head + "."]
    item = next((x for x in c.get("digest") or [] if x.get("kind") == "seasonal"), None)
    if p.get("shelf_action_recommended") and item and item.get("actionable"):
        parts.append(f"Quick win: {lc_first(safe(item['actionable']).rstrip('.'))}.")
    agg = m.get("customer_aggregate") or {}
    if agg.get("repeat_customer_pct"):
        parts.append(f"With {pct(agg['repeat_customer_pct'])} repeat customers, a 'summer essentials' post brings them in before the peak.")
    parts.append(cta(hi, "Want me to draft it?", "Draft kar doon?"))
    return " ".join(parts), "binary_yes_no", "Seasonal demand shift with exact category deltas, the digest's shelf action, and the merchant's repeat base."


def h_unverified(ctx: dict) -> tuple[str, str, str]:
    m, c, t, hi, sal = ctx["merchant"], ctx["category"], ctx["trigger"], ctx["hi"], ctx["sal"]
    p = t.get("payload") or {}
    name = safe(first(m, "identity.name"))
    parts = [f"{sal}, {name}'s Google profile is still unverified."]
    gaps = weak_spots(m, c)
    if gaps:
        parts.append("Right now you're at " + "; ".join(gaps[:2]) + ".")
    if p.get("estimated_uplift_pct") is not None:
        parts.append(f"Verified profiles see an estimated ~{pct(p['estimated_uplift_pct'])} uplift.")
    path = human(p.get("verification_path", "")).replace("postcard or phone call", "a Google postcard or phone call")
    parts.append(f"It only takes {path}; I'll walk you through it in 5 minutes." if path else "I'll walk you through it in 5 minutes.")
    parts.append(cta(hi, "Start now?", "Abhi shuru karein?"))
    return " ".join(parts), "binary_yes_no", "Unverified GBP with the trigger's uplift estimate and verification path, framed against the merchant's gap to peers."


def h_weather_event(ctx: dict) -> tuple[str, str, str]:
    m, c, t, hi, sal = ctx["merchant"], ctx["category"], ctx["trigger"], ctx["hi"], ctx["sal"]
    p = t.get("payload") or {}
    desc = safe(first(p, "headline", "title", "event_name", "description", "event", default=""))
    temp = p.get("temperature_c") or p.get("temp_c")
    parts = [f"{sal}, heads-up for {locality(m)}: " + (desc or (f"{temp}°C today" if temp else human(t.get('kind'))))+ "."]
    offers = merchant_offers(m)
    if offers:
        parts.append(f"Your live offer '{offers[0]}' is a good fit to push today.")
    parts.append(cta(hi, "Want me to post a same-day update for customers?", "Customers ke liye same-day update post kar doon?"))
    return " ".join(parts), "binary_yes_no", "External event/weather from the trigger tied to the merchant's locality and live offer."


def h_generic(ctx: dict) -> tuple[str, str, str]:
    m, c, t, hi, sal = ctx["merchant"], ctx["category"], ctx["trigger"], ctx["hi"], ctx["sal"]
    p = {k: v for k, v in (t.get("payload") or {}).items() if k not in ("placeholder", "metric_or_topic", "category")}
    name = safe(first(m, "identity.name"))
    facts = []
    for k, v in list(p.items())[:3]:
        if isinstance(v, (str, int, float)) and not isinstance(v, bool):
            val = pct(v) if isinstance(v, float) and abs(v) < 1 else (fmt_date(v) if parse_dt(v) and "-" in str(v) else human(v))
            facts.append(f"{human(k)}: {val}")
    parts = [f"{sal}, update on {name} — {human(t.get('kind', 'account'))}" + (f" ({'; '.join(facts)})" if facts else "") + "."]
    gaps = weak_spots(m, c)
    if gaps:
        parts.append("Where you stand: " + "; ".join(gaps[:2]) + ".")
    offers = merchant_offers(m)
    if offers:
        parts.append(f"Your live offer '{offers[0]}' can carry this.")
    parts.append(cta(hi, "Want me to draft the next step for you?", "Next step draft kar doon?"))
    return " ".join(parts), "binary_yes_no", "Generic grounded fallback: trigger fields, peer gaps and live offer only."


def route_merchant(kind: str):
    k = kind.lower()
    table = [
        ("active_planning", h_planning), ("planning", h_planning),
        ("competitor", h_competitor), ("gbp_unverified", h_unverified), ("unverified", h_unverified),
        ("renewal", h_renewal), ("winback", h_winback), ("dormant", h_dormant),
        ("curious", h_curious), ("scheduled_recurring", h_curious),
        ("milestone", h_milestone), ("review_theme", h_review_theme),
        ("ipl", h_ipl), ("match", h_ipl), ("festival", h_festival),
        ("seasonal_perf_dip", h_seasonal_dip), ("perf_dip", h_perf_dip), ("perf_spike", h_perf_spike),
        ("supply", h_supply), ("category_seasonal", h_season_shift),
        ("weather", h_weather_event), ("heatwave", h_weather_event), ("local_news", h_weather_event),
        ("research", h_digest), ("digest", h_digest), ("regulation", h_digest), ("compliance", h_digest),
        ("cde", h_digest), ("trend", h_digest),
    ]
    for key, fn in table:
        if key in k:
            return fn
    return h_generic


# =============================================================================
# Customer-facing (sent as merchant_on_behalf)
# =============================================================================
CONSENT_MAP = {
    "refill": {"refill_reminders", "recall_alerts", "transactional"},
    "recall": {"recall_reminders", "appointment_reminders", "recall_alerts", "treatment_followup"},
    "appointment": {"appointment_reminders", "recall_reminders", "transactional", "promotional_offers"},
    "trial": {"kids_program_updates", "program_updates", "appointment_reminders", "trial_followup", "promotional_offers"},
    "wedding": {"bridal_package_followup", "appointment_reminders"},
    "winback": {"winback_offers", "renewal_reminders", "promotional_offers"},
    "lapsed": {"winback_offers", "recall_reminders", "promotional_offers", "renewal_reminders"},
    "winback_offers": {"winback_offers"},
}


def consent_ok(customer: dict, kind: str) -> bool:
    consent = customer.get("consent") or {}
    if not consent.get("opted_in_at"):
        return False
    scopes = {str(s).lower() for s in consent.get("scope") or []}
    if "all" in scopes:
        return True
    k = kind.lower()
    for key, allowed in CONSENT_MAP.items():
        if key in k and scopes & allowed:
            return True
    # token overlap, e.g. kind "wedding_package_followup" vs scope "bridal_package_followup"
    ktoks = set(k.split("_")) - {"due", "followup", "soft", "hard"}
    return any(ktoks & set(s.split("_")) for s in scopes)


def cust_lang(customer: dict) -> str:
    lp = str(first(customer, "identity.language_pref", default="en")).lower()
    if lp.startswith("hi"):
        return "hi"
    return {"ta": "ta", "te": "te", "kn": "kn", "mr": "mr"}.get(lp[:2], "en")


def cust_name(customer: dict) -> tuple[str, str]:
    """Returns (addressee, child_name_or_empty) — minors are addressed via the parent."""
    raw = safe(first(customer, "identity.name", default="there"))
    mm = re.match(r"(.+?)\s*\(parent:\s*(.+?)\)", raw)
    if mm:
        return mm.group(2), mm.group(1)
    if raw.startswith("("):
        return "there", ""
    return raw, ""


def greet(lang: str, name: str) -> str:
    return {"hi": f"Namaste {name}", "ta": f"Vanakkam {name}", "te": f"Namaskaram {name}",
            "kn": f"Namaskara {name}", "mr": f"Namaskar {name}"}.get(lang, f"Hi {name}")


def slot_labels(slots: Any) -> list[str]:
    out = []
    for s in slots or []:
        if isinstance(s, dict):
            out.append(safe(s.get("label") or fmt_date(s.get("iso"), with_day=True) + " " + fmt_time(s.get("iso"))))
        else:
            out.append(safe(s))
    return [x for x in out if x]


def compose_customer(category: dict, merchant: dict, trigger: dict, customer: dict) -> dict:
    kind = str(trigger.get("kind", "")).lower()
    p = trigger.get("payload") or {}
    biz = safe(first(merchant, "identity.name"))
    lang = cust_lang(customer)
    hi = lang == "hi"
    name, child = cust_name(customer)
    rel = customer.get("relationship") or {}
    offers = merchant_offers(merchant)
    slug = merchant.get("category_slug", "")
    emoji = {"dentists": " 🦷", "salons": " 💇", "gyms": " 💪", "pharmacies": "", "restaurants": " 🍽️"}.get(slug, "")
    opener = f"{greet(lang, name)}, {biz} here{emoji}."
    body, rationale = "", ""
    if "refill" in kind:
        meds = ", ".join(p.get("molecule_list") or []) or "your regular medicines"
        due = p.get("stock_runs_out_iso")
        if hi:
            body = f"{greet(lang, name)}{'' if name.lower().startswith(('mr', 'mrs', 'ms', 'dr')) else ' ji'}, {biz} se. Aapki {meds} ki supply {fmt_date(due, with_day=True) if due else 'jaldi'} tak khatam ho jayegi."
            if p.get("delivery_address_saved"):
                body += " Hum saved address par home delivery kar sakte hain."
            if first(customer, "identity.senior_citizen") and any("senior" in o.lower() for o in offers):
                body += f" {next(o for o in offers if 'senior' in o.lower())} bhi lagega."
            body += " Refill bhej dein? Reply YES. Dose badla ho to pharmacist se baat karein."
        else:
            body = f"{opener} Your {meds} stock runs out by {fmt_date(due, with_day=True) if due else 'soon'}."
            if p.get("delivery_address_saved"):
                body += " We can deliver to your saved address."
            body += " Shall we pack the refill? Reply YES. If your dose has changed, please tell the pharmacist."
        rationale = "Chronic refill with exact molecules and run-out date, delivery option, senior discount if live; dose-change safety line."
    elif "appointment" in kind:
        when = first(p, "appointment_iso", "appointment_time", "slot")
        svc = human(first(p, "service", default=""))
        body = f"{opener} Reminder: your {svc + ' ' if svc else ''}appointment is " + (f"{fmt_date(when, with_day=True)}, {fmt_time(when)}." if parse_dt(when) else "tomorrow.")
        body += " Reply YES to confirm, or tell us a better time." if not hi else " Confirm karne ke liye YES reply karein, ya better time batayein."
        rationale = "Appointment reminder with the booked time; one confirm CTA."
    elif "recall" in kind or ("lapsed_soft" in kind and p.get("service_due")):
        svc = human(p.get("service_due", "")) or "check-up"
        last = p.get("last_service_date") or rel.get("last_visit")
        slots = slot_labels(p.get("available_slots"))
        who = f"{child}'s" if child else "your"
        if hi:
            body = f"{opener} {child + ' ki' if child else 'Aapki'} {svc} due hai" + (f" — last visit {fmt_date(last)} ko tha." if last else ".")
        else:
            body = f"{opener} {who.capitalize()} {svc} is due" + (f" — last visit was {fmt_date(last)}." if last else ".")
        price = next((o for o in offers if any(w in o.lower() for w in svc.lower().split() if len(w) > 3)), offers[0] if offers else "")
        if price:
            body += (f" Offer: {price}." if not hi else f" Offer chal raha hai: {price}.")
        if len(slots) >= 2:
            body += (f" Aapke liye 2 slots ready hain: {slots[0]} ya {slots[1]}." if hi else f" Two slots open: {slots[0]} or {slots[1]}.")
            body += " Reply 1 or 2, or tell us a time that works."
        else:
            pref = human(first(customer, "preferences.preferred_slots", default=""))
            body += (f" {pref} slot chahiye? Reply YES." if hi and pref else f" Want a {pref} slot? Reply YES." if pref else " Reply YES to book.")
        rationale = "Recall with service, last visit, real slots and live price; honours language and slot preference."
    elif "trial" in kind:
        opts = slot_labels(p.get("next_session_options"))
        who = child or "you"
        body = f"{opener} Hope {who} enjoyed the trial on {fmt_date(p.get('trial_date'))}!" if p.get("trial_date") else f"{opener} Hope {who} enjoyed the trial!"
        if opts:
            body += f" Next session: {opts[0]}. Shall we hold a spot? Reply YES."
        else:
            body += " Want us to hold a spot in the next session? Reply YES."
        rationale = "Trial follow-up with trial date and the actual next session option."
    elif "wedding" in kind or "bridal" in kind:
        wd = p.get("wedding_date") or first(customer, "preferences.wedding_date")
        body = f"{opener} Your bridal trial was on {fmt_date(p.get('trial_completed'))}" if p.get("trial_completed") else opener
        if wd:
            body += f" and the big day is {fmt_date(wd, with_year=True)}" + (f" — {p['days_to_wedding']} days to go" if p.get("days_to_wedding") else "") + "."
        nxt = human(p.get("next_step_window_open", "")).replace("30-day", "30-day").replace("program 30day", "program (30 days)").replace("30day", "30-day")
        if nxt:
            body += f" Next step is the {nxt}, which we'd time before the wedding."
        pref = human(first(customer, "preferences.preferred_slots", default=""))
        body += f" Shall we pencil in a {pref} start? Reply YES." if pref else " Shall we pencil in a start date? Reply YES."
        rationale = "Bridal follow-up with trial date, wedding date/countdown and the specific next program."
    elif "lapsed" in kind or "winback" in kind:
        days = p.get("days_since_last_visit")
        last = rel.get("last_visit")
        focus = human(p.get("previous_focus", "") or first(customer, "preferences.training_focus", default=""))
        if hi:
            body = f"{opener} " + (f"Aapki last visit ko {days} din ho gaye" if days else f"Aapki last visit {fmt_date(last)} ko thi" if last else "Aapko miss kar rahe hain")
            body += (f" — {focus} ka progress wapas pick karte hain." if focus else ".")
            if offers:
                body += f" Wapas aane ke liye: {offers[0]}."
            body += " Is hafte ek slot rakh dein? Reply YES."
        else:
            body = f"{opener} " + (f"It's been {days} days since your last visit" if days else f"We haven't seen you since {fmt_date(last)}" if last else "We've missed you")
            body += (f" — your {focus} progress is worth picking back up." if focus else ".")
            if offers:
                body += f" {offers[0]} to ease back in."
            body += " Want us to save you a slot this week? Reply YES."
        rationale = "Win-back with days since visit and the customer's own focus; no guilt, one low-friction CTA."
    else:
        body = f"{opener}" + (f" {offers[0]} is live this week." if offers else "") + " Want the details? Reply YES."
        rationale = "Customer message limited to the merchant's live offer and consented scope."
    return {"body": pretty_dates(safe(body)), "cta": "binary_yes_no", "send_as": "merchant_on_behalf",
            "suppression_key": make_key(trigger), "rationale": rationale}


def consent_blocked_to_merchant(category: dict, merchant: dict, trigger: dict, customer: dict) -> dict:
    """No consent for this message type: tell the merchant instead of messaging the customer."""
    hi = uses_hindi(merchant, category)
    sal = salutation(merchant, category, hi)
    name, child = cust_name(customer)
    who = child or name
    scopes = ", ".join(human(s) for s in first(customer, "consent.scope", default=[]) or []) or "nothing yet"
    what = re.sub(r"\s*due$", "", human(trigger.get("kind", "follow-up"))).replace("followup", "follow-up")
    body = (f"{sal}, {who}'s {what} is due, but they've only opted in to {scopes} — "
            f"so I haven't messaged them. {cta(hi, 'Want me to ask for reminder opt-in at their next visit?', 'Next visit par reminder opt-in le lein?')}")
    return {"body": safe(body), "cta": "binary_yes_no", "send_as": "vera", "suppression_key": make_key(trigger),
            "rationale": "Customer consent scope does not cover this message type, so outreach is routed to the merchant instead of the customer."}


# =============================================================================
# compose()
# =============================================================================
TABOO_DEFAULT = ["guaranteed", "miracle", "100% safe", "best in city"]


def compose(category: dict, merchant: dict, trigger: dict, customer: Optional[dict] = None) -> dict:
    category = category or {}
    merchant = merchant or {}
    trigger = trigger or {}
    if customer:
        if not consent_ok(customer, str(trigger.get("kind", ""))):
            return consent_blocked_to_merchant(category, merchant, trigger, customer)
        return compose_customer(category, merchant, trigger, customer)
    hi = uses_hindi(merchant, category)
    ctx = {"merchant": merchant, "category": category, "trigger": trigger, "hi": hi,
           "sal": salutation(merchant, category, hi)}
    try:
        body, cta_type, rationale = route_merchant(str(trigger.get("kind", "")))(ctx)
    except Exception as e:  # never let one odd payload kill a tick
        body, cta_type, rationale = h_generic(ctx)
        rationale += f" (handler fallback: {type(e).__name__})"
    body = safe(body)
    for w in (first(category, "voice.vocab_taboo", default=[]) or []) + TABOO_DEFAULT:
        w0 = str(w).split(" (")[0]
        if w0 and w0.lower() in body.lower():
            body = re.sub(re.escape(w0), "", body, flags=re.I)
    return {"body": pretty_dates(safe(body)) or "Quick update on your listing — want the details? Reply YES.",
            "cta": cta_type, "send_as": "vera", "suppression_key": make_key(trigger), "rationale": rationale}


# =============================================================================
# Optional LLM polish (off unless VERA_LLM_PROVIDER + VERA_LLM_API_KEY are set)
# =============================================================================
LLM_PROVIDER = os.getenv("VERA_LLM_PROVIDER", "").lower()
LLM_KEY = os.getenv("VERA_LLM_API_KEY", "")
LLM_MODEL = os.getenv("VERA_LLM_MODEL", "")
_llm_cache: dict[str, str] = {}
_POOL = ThreadPoolExecutor(max_workers=8)   # shared; a tick never blocks on a slow LLM call

LLM_SYSTEM = """You polish WhatsApp messages written by Vera, magicpin's merchant assistant.
Rewrite the DRAFT so it reads naturally and is more compelling, but:
- keep EVERY fact, number, name, date and price exactly; add NO new numbers, names, offers or claims
- keep the salutation and the final call-to-action (last sentence, same meaning, keep "Reply YES" if present)
- peer/colleague tone, no hype, no greetings like "I hope you're well", no self-introduction
- keep any Hindi-English mix the draft uses; stay under 420 characters if the draft is shorter than that
Return ONLY the rewritten message text."""


def _llm_call(prompt: str) -> str:
    if LLM_PROVIDER == "gemini":
        model = LLM_MODEL or "gemini-flash-latest"
        body = json.dumps({"contents": [{"parts": [{"text": LLM_SYSTEM + "\n\n" + prompt}]}],
                           "generationConfig": {"temperature": 0, "maxOutputTokens": 600}}).encode()
        req = urlrequest.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                                 data=body, headers={"Content-Type": "application/json", "x-goog-api-key": LLM_KEY})
        data = json.loads(urlrequest.urlopen(req, timeout=9).read())
        return data["candidates"][0]["content"]["parts"][0]["text"]
    url = {"groq": "https://api.groq.com/openai/v1/chat/completions",
           "openai": "https://api.openai.com/v1/chat/completions",
           "openrouter": "https://openrouter.ai/api/v1/chat/completions"}.get(LLM_PROVIDER)
    if not url:
        raise RuntimeError("unsupported provider")
    model = LLM_MODEL or {"groq": "openai/gpt-oss-120b", "openai": "gpt-4o-mini"}.get(LLM_PROVIDER, "")
    payload = {"model": model, "temperature": 0, "max_tokens": 800,
               "messages": [{"role": "system", "content": LLM_SYSTEM}, {"role": "user", "content": prompt}]}
    if "gpt-oss" in model:
        payload["reasoning_effort"] = "low"
    req = urlrequest.Request(url, data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json", "Authorization": f"Bearer {LLM_KEY}",
                                      "User-Agent": "vera-bot/2.0"})
    data = json.loads(urlrequest.urlopen(req, timeout=9).read())
    return data["choices"][0]["message"]["content"]


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "").lstrip("0") or "0" for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}


def llm_polish(draft: str, category: dict) -> str:
    """Returns the polished text only if it passes grounding checks; else the draft."""
    if not (LLM_PROVIDER and LLM_KEY) or not draft:
        return draft
    key = hashlib.sha1(draft.encode()).hexdigest()
    if key in _llm_cache:
        return _llm_cache[key]
    try:
        out = safe(_llm_call(f"DRAFT:\n{draft}")).strip('"')
    except Exception:
        return draft
    ok = (bool(out) and _numbers(out) <= _numbers(draft)
          and ("YES" in out) == ("YES" in draft)
          and len(out) <= max(len(draft) * 1.25, 200)
          and not any(str(w).split(" (")[0].lower() in out.lower() for w in (first(category, "voice.vocab_taboo", default=[]) or [])))
    result = out if ok else draft
    _llm_cache[key] = result
    return result


# =============================================================================
# HTTP API
# =============================================================================
class ContextIn(BaseModel):
    scope: str
    context_id: str
    version: int = Field(ge=0)
    payload: dict[str, Any]
    delivered_at: Optional[str] = None


class TickIn(BaseModel):
    now: str
    available_triggers: list[str] = Field(default_factory=list)


class ReplyIn(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: Optional[str] = None
    turn_number: int = 1


def _ack(b: ContextIn) -> str:
    return "ack_" + hashlib.md5((b.scope + b.context_id + str(b.version)).encode()).hexdigest()[:12]


@app.post("/v1/context")
async def push_context(b: ContextIn):
    if b.scope not in {"category", "merchant", "customer", "trigger"}:
        raise HTTPException(400, detail={"accepted": False, "reason": "invalid_scope",
                                         "details": "scope must be category, merchant, customer, or trigger"})
    key = (b.scope, b.context_id)
    cur = contexts.get(key)
    if cur and b.version < cur["version"]:
        raise HTTPException(status_code=409, detail={"accepted": False, "reason": "stale_version",
                                                     "current_version": cur["version"]})
    now = datetime.now(timezone.utc).isoformat()
    if cur and b.version == cur["version"]:
        return {"accepted": True, "ack_id": _ack(b), "stored_at": now, "duplicate": True}
    contexts[key] = {"version": b.version, "payload": b.payload}
    return {"accepted": True, "ack_id": _ack(b), "stored_at": now}


def _template_params(msg: dict, merchant: dict, customer: Optional[dict]) -> list[str]:
    who = cust_name(customer)[0] if customer else safe(first(merchant, "identity.owner_first_name", "identity.name"))
    return [who, safe(first(merchant, "identity.name")), msg["body"]]


@app.post("/v1/tick")
async def tick(b: TickIn):
    planned = []
    for tid in b.available_triggers:
        if len(planned) >= 20:
            break
        trg = get("trigger", tid)
        if not trg:
            continue
        mid = trg.get("merchant_id") or first(trg, "payload.merchant_id")
        m = get("merchant", mid) if mid else None
        cat_slug = (m or {}).get("category_slug") or first(trg, "payload.category")
        cat = get("category", cat_slug) if cat_slug else None
        if not (m and cat):
            continue
        cid = trg.get("customer_id") or first(trg, "payload.customer_id")
        cust = get("customer", cid) if cid else None
        if cid and not cust:
            continue
        sk = make_key(trg)
        if sk in sent_suppressions:
            continue
        msg = compose(cat, m, trg, cust)
        is_cust = msg["send_as"] == "merchant_on_behalf"
        planned.append({"trg": trg, "tid": tid, "mid": mid, "cid": cid if is_cust else None, "cust": cust if is_cust else None,
                        "cat": cat, "cat_slug": cat_slug, "m": m, "msg": msg, "sk": sk, "merchant_id": mid,
                        "customer_id": cid if is_cust else None})
    # optional LLM polish in parallel, bounded so the tick always returns in time
    if LLM_PROVIDER and LLM_KEY and planned:
        futs = {_POOL.submit(llm_polish, x["msg"]["body"], x["cat"]): x for x in planned}
        done, _ = wait(futs, timeout=11)   # stragglers keep their deterministic draft
        for f in done:
            try:
                futs[f]["msg"]["body"] = f.result()
            except Exception:
                pass
    actions = []
    for x in planned:
        msg, trg = x["msg"], x["trg"]
        conv = "conv_" + hashlib.sha1(f"{x['mid']}|{x['tid']}|{b.now}".encode()).hexdigest()[:16]
        kind = str(trg.get("kind", "update"))
        actions.append({"conversation_id": conv, "merchant_id": x["mid"], "customer_id": x["cid"],
                        "send_as": msg["send_as"], "trigger_id": x["tid"],
                        "template_name": ("merchant_" if x["cid"] else "vera_") + kind + "_v1",
                        "template_params": _template_params(msg, x["m"], x["cust"]),
                        "body": msg["body"], "cta": msg["cta"], "suppression_key": x["sk"], "rationale": msg["rationale"]})
        sent_suppressions.add(x["sk"])
        conversations[conv] = {"merchant_id": x["mid"], "customer_id": x["cid"], "trigger_id": x["tid"],
                               "category_id": x["cat_slug"], "scope": "customer" if x["cid"] else "merchant",
                               "turns": [{"role": "bot", "body": msg["body"]}], "ended": False, "auto_count": 0,
                               "offered": msg["body"]}
    return {"actions": actions}


# ---------------------------------------------------------------------------
# Multi-turn replies
# ---------------------------------------------------------------------------
NO_PATTERNS = ["stop", "unsubscribe", "not interested", "do not message", "dont message", "don't message",
               "don’t message", "no thanks", "leave me alone", "remove me", "not required", "not needed",
               "spam", "band karo", "mat bhejo", "nahi chahiye", "nahin chahiye"]
YES_PATTERNS = ["yes", "yeah", "yep", "sure", "go ahead", "do it", "let's do it", "let’s do it", "lets do it",
                "send it", "please do", "interested", "join", "sign me up", "ok", "okay", "haan", "ha ji",
                "theek hai", "thik hai", "kar do", "karo", "chalega", "done", "proceed", "whats next", "what's next"]
AUTO_PATTERNS = ["thank you for contacting", "our team will respond", "respond shortly", "automated assistant",
                 "auto-reply", "auto reply", "this is an automated", "we will get back", "will get back to you",
                 "out of office", "business hours", "aapki jaankari ke liye", "team tak pahuncha"]
HINDI_MARKERS = ["haan", "nahi", "kya", "karo", "kar do", "chahiye", "bhai", "ji", "theek", "acha", "accha",
                 "kaise", "kitna", "hai", "mujhe", "aap"]
OFF_TOPIC = ["gst", "income tax", "tax filing", "itr", "loan", "legal", "lawyer", "assignment", "visa", "passport"]


def _has(low: str, pats: list[str]) -> bool:
    return any(re.search(r"(?<![a-z])" + re.escape(p) + r"(?![a-z])", low) for p in pats)


def _next_step(conv: dict, ctx: dict) -> str:
    """Concrete next action tied to what we offered in the opening message."""
    offered = (conv.get("offered") or "").lower()
    trig = ctx.get("trigger") or {}
    kind = str(trig.get("kind", "")).lower()
    m = ctx.get("merchant") or {}
    name = safe(first(m, "identity.name", default="your listing"))
    if "renewal" in offered or "renewal" in kind:
        return f"Sending the renewal link for {name} now — once paid, profile upkeep continues without a gap."
    if "abstract" in offered:
        return "Sending the abstract now, and drafting the recall WhatsApp — you'll get it here in a few minutes to approve."
    if "checklist" in offered:
        return "Sending the 5-point checklist now. Tick each item and reply DONE when finished."
    if "registration" in offered:
        return "Sending the registration details now."
    if "verification" in kind or "walk you through" in offered:
        return "Step 1: open your Google Business Profile and tap 'Get verified'. Choose postcard or phone call — tell me which you picked and I'll guide the next step."
    if "reactivation" in offered:
        return f"Sending the reactivation steps for {name} now; the comeback message to lapsed customers goes out in week one."
    if "recall" in kind and "list" in offered or "send it now" in offered:
        return "Sending the replacement WhatsApp to the affected customers now; I'll share the delivered count here."
    return f"Done — drafting it for {name} now. You'll get the draft here to approve before anything goes live."


def respond(conv: dict, message: str, turn: int, context: Optional[dict] = None) -> dict:
    text = message.strip()
    low = text.lower()
    turns = conv.setdefault("turns", [])
    turns.append({"role": "merchant", "body": text})
    context = context or {}
    merchant = context.get("merchant") or {}
    category = context.get("category") or {}
    hi_turn = _has(low, HINDI_MARKERS) or bool(re.search(r"[ऀ-ॿ]", text))
    if conv.get("ended"):
        return {"action": "end", "rationale": "Conversation already closed."}

    # 1) auto-reply detection (per conversation AND per merchant across conversations)
    prev = [x.get("body", "").strip().lower() for x in turns if x.get("role") == "merchant"]
    auto = _has(low, AUTO_PATTERNS) or (len(prev) >= 2 and prev[-1] == prev[-2])
    if auto:
        mid = conv.get("merchant_id") or ""
        auto_reply_counts[mid] = auto_reply_counts.get(mid, 0) + 1
        conv["auto_count"] = conv.get("auto_count", 0) + 1
        if max(conv["auto_count"], auto_reply_counts[mid]) >= 2:
            conv["ended"] = True
            return {"action": "end", "rationale": "Same canned auto-reply seen again; exiting instead of burning turns on an unattended inbox."}
        hi = uses_hindi(merchant, category)
        sal = salutation(merchant, category, hi) if merchant else ""
        body = (f"Lagta hai yeh auto-reply hai. {sal + ', ' if sal else ''}jab aap dekhein, bas YES reply kar dijiye — main baaki sambhal lungi."
                if hi else
                f"Looks like an auto-reply — no problem. {sal + ', w' if sal else 'W'}hen you see this, just reply YES and I'll handle the rest.")
        return {"action": "send", "body": body, "cta": "binary_yes_no",
                "rationale": "Detected an auto-reply; one owner-directed nudge, will exit if it repeats."}

    # 2) opt-out / hostility
    if _has(low, NO_PATTERNS) or low in ("no", "nahi", "nope", "na"):
        conv["ended"] = True
        return {"action": "end", "rationale": "Merchant opted out or was hostile; stop messaging."}

    # 3) off-topic (stay on mission, politely)
    if _has(low, OFF_TOPIC):
        body = ("GST/tax filing mere scope se bahar hai — uske liye apne CA se baat karein. Main aapki listing aur customers mein help kar sakti hoon; jo draft offer kiya tha, woh bhej doon?"
                if hi_turn else
                "That's outside what I can help with — a CA is the right person for tax filing. I can keep helping with your listing and customers: shall I send the draft I mentioned?")
        return {"action": "send", "body": body, "cta": "binary_yes_no", "rationale": "Polite boundary on an off-topic ask, then back to the open offer."}

    # 4) explicit intent -> action immediately (no re-qualifying)
    if _has(low, YES_PATTERNS):
        if _has(low, ["join", "sign me", "onboard", "register"]):
            body = "Great — starting your onboarding now. Next: I'll send a 3-step setup link here; your listing goes live once step 3 is done."
        else:
            body = _next_step(conv, context)
        conv["turns"].append({"role": "bot", "body": body})
        return {"action": "send", "body": body, "cta": "none",
                "rationale": "Explicit acceptance — switched straight to action with a concrete next step."}

    # 5) back off
    if _has(low, ["later", "busy", "not now", "call me later", "tomorrow", "give me time", "baad mein", "abhi nahi"]):
        return {"action": "wait", "wait_seconds": 1800, "rationale": "Merchant asked for time; backing off 30 minutes."}

    # 6) questions
    if _has(low, ["price", "cost", "how much", "charges", "fee", "kitna", "kitne"]):
        plan = first(merchant, "subscription.plan")
        body = (f"Your current plan is {plan}. " if plan else "") + "I don't want to guess numbers — I'll get the exact pricing from the magicpin team and share it here. Shall I?"
        return {"action": "send", "body": body, "cta": "binary_yes_no", "rationale": "Honest about pricing; no fabricated numbers."}
    if _has(low, ["who are you", "what is this", "why message", "how got my number", "kaun"]):
        body = "I'm Vera, magicpin's assistant for your Google listing and customers. Reply STOP anytime and I won't message again."
        return {"action": "send", "body": body, "cta": "none", "rationale": "Transparent identity + easy opt-out."}
    if turn >= 5:
        conv["ended"] = True
        return {"action": "end", "rationale": "Turn budget reached without a clear next step; closing gracefully."}

    trigger = context.get("trigger") or {}
    item = digest_item(category, trigger) if trigger else None
    tp = (trigger or {}).get("payload") or {}
    title = safe((item or {}).get("title") or tp.get("title") or "")
    detail = sentences((item or {}).get("summary", "") or tp.get("summary", ""), 2)
    if title or detail:
        body = "Sure — " + (f"{title}. " if title else "") + (f"{detail} " if detail else "") + "I can draft the next step for you; shall I go ahead?"
    else:
        body = "Got it. I'll keep this short: I can draft the next step for you right now — shall I go ahead?"
    prior = {safe(x.get("body", "")) for x in turns if x.get("role") == "bot"}
    if safe(body) in prior:
        conv["ended"] = True
        return {"action": "end", "rationale": "Next message would repeat verbatim; closing instead."}
    conv["turns"].append({"role": "bot", "body": body})
    return {"action": "send", "body": body, "cta": "binary_yes_no", "rationale": "Answered from the trigger's context and offered one next step."}


@app.post("/v1/reply")
async def reply(b: ReplyIn):
    conv = conversations.get(b.conversation_id)
    if not conv:
        conv = {"merchant_id": b.merchant_id, "customer_id": b.customer_id, "turns": [], "ended": False,
                "scope": "customer" if b.customer_id else "merchant"}
        conversations[b.conversation_id] = conv
    if b.merchant_id:
        conv["merchant_id"] = b.merchant_id
    if b.customer_id:
        conv["customer_id"] = b.customer_id
    mid, cid, tid = conv.get("merchant_id"), conv.get("customer_id"), conv.get("trigger_id")
    merchant = get("merchant", mid) if mid else None
    category = get("category", merchant.get("category_slug")) if merchant and merchant.get("category_slug") else None
    ctx = {"merchant": merchant, "category": category,
           "customer": get("customer", cid) if cid else None,
           "trigger": get("trigger", tid) if tid else None}
    return respond(conv, b.message, b.turn_number, ctx)


@app.get("/v1/healthz")
async def healthz():
    counts = {k: 0 for k in ["category", "merchant", "customer", "trigger"]}
    for (scope, _), _v in contexts.items():
        counts[scope] = counts.get(scope, 0) + 1
    return {"status": "ok", "uptime_seconds": int(time.time() - START), "contexts_loaded": counts}


@app.get("/v1/metadata")
async def metadata():
    return {"team_name": os.getenv("TEAM_NAME", "Anshika AI"),
            "team_members": [os.getenv("TEAM_MEMBER", "Anshika Singh")],
            "model": ("trigger-routed composer + " + (LLM_MODEL or LLM_PROVIDER) + " polish") if LLM_PROVIDER and LLM_KEY else "trigger-routed deterministic composer",
            "approach": "Per-trigger-kind handlers grounded in the 4 contexts (no invented offers/data), peer benchmarks as proof, single YES CTA; consent-gated customer flows; optional LLM polish accepted only if it adds no new numbers.",
            "contact_email": os.getenv("CONTACT_EMAIL", ""), "version": "2.0.0",
            "submitted_at": datetime.now(timezone.utc).isoformat()}


@app.post("/v1/teardown")
async def teardown():
    contexts.clear()
    conversations.clear()
    sent_suppressions.clear()
    auto_reply_counts.clear()
    return {"ok": True, "cleared": True}
