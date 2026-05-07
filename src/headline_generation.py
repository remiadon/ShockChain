"""
ShockChain Event Generation Pipeline
=====================================
Generates ~5,000 historical market events using Claude API,
then verifies dates via a second pass.

Requirements:
    pip install anthropic python-dotenv

Project structure:
    project/
    ├── .env                 # ANTHROPIC_API_KEY=sk-ant-...
    └── src/
        └── generate_events.py

Usage:
    cd src
    python generate_events.py --phase generate
    python generate_events.py --phase verify
    python generate_events.py --phase merge

Options:
    --phase generate|verify|merge
    --no-resume       Start from scratch (ignore progress)
    --max-concurrent  Max parallel API calls (default: 10)
"""

import anthropic
import json
import csv
import asyncio
import argparse
import os
from pathlib import Path
from calendar import monthrange
from collections import Counter

# ---------------------------------------------------------------------------
# Load .env from parent directory
# ---------------------------------------------------------------------------

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_GENERATE = "claude-sonnet-4-6"
MODEL_VERIFY = "claude-sonnet-4-6"
TEMPERATURE_GENERATE = 0.7
TEMPERATURE_VERIFY = 0.0
MAX_CONCURRENT_DEFAULT = 10

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
RAW_CSV = OUTPUT_DIR / "events_raw.csv"
VERIFIED_CSV = OUTPUT_DIR / "events_verified.csv"
FINAL_CSV = OUTPUT_DIR / "events_final.csv"
LOG_FILE = OUTPUT_DIR / "generation_log.json"

FIELDNAMES = ["headline", "date", "market_fact", "impact", "category", "window"]

START_YEAR = 2016
END_YEAR = 2025

# ---------------------------------------------------------------------------
# Few-shot examples (verified)
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = """
| headline | date | market_fact | impact | category |
|----------|------|-------------|--------|----------|
| BOJ introduces negative interest rate policy for first time | 2016-01-29 | Nikkei up 2.8%, yen weakened | Moderate | monetary_policy |
| Oil prices crash below $27 per barrel as oversupply deepens | 2016-02-11 | WTI hit $26.21 intraday, SPX down 1.2% | High | energy_opec |
| UK votes to leave European Union in referendum | 2016-06-23 | GBP down 8%, SPX down 3.6% Friday | High | election_political |
| Samsung recalls all Galaxy Note 7 phones over exploding batteries | 2016-09-02 | Samsung KS down 7% | Low | mega_cap_corporate |
| Trump wins US presidential election | 2016-11-08 | Futures crashed 5% overnight, recovered to close up 1.1% | Moderate | election_political |
| US launches 59 Tomahawk missiles at Syrian airbase after chemical attack | 2017-04-06 | SPX flat, oil up 1.5%, gold up 0.5% | Low | geopolitical_military |
| North Korea fires Hwasong-12 missile over Hokkaido, Japan | 2017-08-29 | Nikkei down 0.5%, VIX up 1 point | Moderate | geopolitical_military |
| VIX spikes above 50, XIV inverse volatility ETN liquidated overnight | 2018-02-05 | SPX down 4.1%, VIX hit 50 | High | credit_event |
| US withdraws from Iran nuclear deal, reimposes sanctions | 2018-05-08 | Oil up 2.5%, SPX up 0.3% | Low | geopolitical_diplomatic |
| Turkey lira crashes 25% in one week as US doubles steel tariffs | 2018-08-10 | TRY down 16% on the day, EM currencies sold off | High | trade_policy |
| Apple warns Q1 revenue will miss guidance on weak China iPhone demand | 2019-01-02 | AAPL down 10%, SPX down 2.5% | High | mega_cap_earnings |
| Trump tweets US will raise tariffs on $200B Chinese goods to 25% | 2019-05-05 | SPX down 2.4% on Monday, VIX up 5 pts | High | trade_policy |
| Drone attack on Saudi Aramco knocks out 5% of global oil supply | 2019-09-14 | Oil up 15% at Monday open | High | energy_opec |
| WHO declares COVID-19 a global pandemic | 2020-03-11 | SPX down 4.9%, entered bear market | High | pandemic_health |
| Fed cuts rates to zero in emergency Sunday meeting, launches unlimited QE | 2020-03-15 | SPX futures hit limit down, opened down 8% Monday | High | monetary_policy |
| Pfizer and BioNTech announce COVID vaccine over 90% effective in trial | 2020-11-09 | SPX up 1.2%, massive value rotation | High | pandemic_health |
| Archegos Capital collapses, banks lose over $10B in forced liquidation | 2021-03-26 | CS down 14%, Nomura down 16% | Moderate | credit_event |
| Colonial Pipeline shut down after DarkSide ransomware attack | 2021-05-07 | Gasoline futures up 3%, SPX down 0.3% | Moderate | supply_disruption |
| China orders private tutoring companies to become non-profit | 2021-07-23 | Chinese tech stocks down 8-15%, US tech down 1% | High | regulatory |
| Russia launches full-scale invasion of Ukraine | 2022-02-24 | SPX down 2.1%, VIX hit 37, oil above $100 | High | geopolitical_military |
| UK Chancellor Kwarteng announces mini-budget with unfunded tax cuts | 2022-09-23 | GBP crashed to record low, gilt yields spiked 1% | High | fiscal_policy |
| Nord Stream pipelines damaged by underwater explosions in Baltic Sea | 2022-09-26 | European gas up 12%, DAX down 1.7% | Moderate | supply_disruption |
| Meta reports Q3 earnings, shares plunge 24% on metaverse spending | 2022-10-26 | META down 24% next day, Nasdaq down 2% | High | mega_cap_earnings |
| SVB collapses, FDIC seizes bank in second-largest US bank failure | 2023-03-10 | KRE regional bank ETF down 8%, SPX down 1.5% | High | credit_event |
| Ukraine grain export deal expires as Russia refuses to renew | 2023-07-17 | Wheat futures up 8% | Low | trade_policy |
| Hamas launches surprise attack on Israel from Gaza | 2023-10-07 | Oil up 4% Monday, SPX down 0.5% | Low | geopolitical_military |
| CrowdStrike software update causes global IT outage, airlines grounded | 2024-07-19 | CRWD down 11%, airlines down 1-3% | Moderate | cyber_infrastructure |
| Fed cuts rates by 50bp, first rate cut since 2020 | 2024-09-18 | SPX up 0.2%, move was largely priced in | Low | monetary_policy |
""".strip()

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_GENERATE = """You are a financial historian specializing in market-moving events.
You return ONLY valid JSON arrays. No preamble, no markdown backticks, no commentary.
Every response must be a JSON array of objects that can be parsed with json.loads().

EVENT CATEGORIES (use exactly these strings):
geopolitical_military, geopolitical_diplomatic, monetary_policy, fiscal_policy,
trade_policy, regulatory, election_political, natural_disaster, pandemic_health,
energy_opec, supply_disruption, credit_event, mega_cap_earnings,
mega_cap_corporate, cyber_infrastructure

IMPACT LEVELS:
- High: multi-sigma moves in major indices (VIX spike, SPX 2%+, currency 5%+)
- Moderate: noticeable but contained (SPX 0.5-2%, sector-level moves)
- Low: event seemed significant but markets barely reacted

DISTRIBUTION TARGETS for each batch:
- Include at least 25% Low impact events (events where markets shrugged it off)
- Mix at least 3 different categories
- Include both US and non-US events
- Include at least 1 stock-specific event (mega_cap_earnings or mega_cap_corporate)
  if any major earnings or corporate events happened in the window

DATE RULES:
- Use the date the event FIRST became known to market participants (YYYY-MM-DD)
- Policy announcements: use announcement date, not effective date
- Earnings: use the report date, not the stock reaction date
- Gradual events (hurricanes, pandemics): use when markets first moved on the news
- Rumors/leaks that moved markets: use the rumor date
- Weekend events: use the actual date; the market_fact references next trading day

HEADLINE RULES:
- Present tense, factual, as it would appear on a news wire on the day
- No retrospective framing ("would eventually"), no editorial ("markets brace")
- 8 to 15 words, specific enough to verify
- Include key numbers when relevant (rate changes in bp, tariff percentages, etc.)

MARKET FACT RULES:
- Must be a specific, verifiable number (e.g., "SPX down 2.4%", "oil up 15%")
- Reference the affected instrument: SPX, VIX, DXY, WTI, specific stock ticker, etc.
- For weekend events, reference the next trading day's reaction

REFERENCE EXAMPLES (use these as style and quality guide):
""" + FEW_SHOT_EXAMPLES + """

Return a JSON array of objects with exactly these keys:
headline, date, market_fact, impact, category

Example output format:
[
  {"headline": "Fed raises rates by 25bp", "date": "2017-06-14", "market_fact": "SPX up 0.1%", "impact": "Low", "category": "monetary_policy"},
  {"headline": "North Korea tests missile over Japan", "date": "2017-08-29", "market_fact": "VIX up 1 point", "impact": "Moderate", "category": "geopolitical_military"}
]"""

GENERATION_PROMPT_TEMPLATE = """Generate real historical events that were relevant to financial markets for the period {start_date} to {end_date}.

Generate between {events_min} and {events_max} events for this window."""

TOPUP_PROMPT_TEMPLATE = """Generate additional real historical events that were relevant to financial markets for the period {start_date} to {end_date}.

Generate between {events_min} and {events_max} NEW events for this window.

IMPORTANT: Do NOT regenerate any of the following events, which were already captured:
{existing_headlines}

Focus on events NOT in that list. Look for:
- Lesser-known but real market-moving events
- Regional market events (Asia, Europe, emerging markets)
- Individual stock events (earnings surprises, M&A, scandals, IPOs)
- Regulatory actions, antitrust, sanctions
- Commodity-specific events (agriculture, metals, natural gas)
- Corporate bankruptcies, credit downgrades, fund blowups"""

SYSTEM_PROMPT_VERIFY = """You are a fact-checker specializing in financial market events.
You return ONLY valid JSON arrays. No preamble, no markdown backticks, no commentary.

For each event, verify that the date is correct. Use your knowledge to confirm the
EXACT date each event first became known to market participants.

Rules:
- Policy announcements: use announcement date, not effective date
- Earnings: use report date, not reaction date
- Gradual events: use when markets first reacted, not the climax
- If the date is wrong, correct it
- If the event is fabricated (never happened), mark status as "fabricated"
- If you cannot verify the date with confidence, mark status as "unverified"

Return a JSON array of objects with these keys:
headline, date, original_date, market_fact, impact, category, status

Where:
- date = the verified/corrected date
- original_date = the date from the input
- status = one of: "confirmed", "date_corrected", "fabricated", "unverified"
"""

VERIFICATION_PROMPT_TEMPLATE = """Events to verify:
{events_json}"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_existing_headlines(csv_path: Path) -> dict[str, list[str]]:
    """Load headlines grouped by base window (without round suffix)."""
    headlines_by_window: dict[str, list[str]] = {}
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                # Strip round suffix: "2016-01-H1-r2" -> "2016-01-H1"
                base_window = row.get("window", "").split("-r")[0]
                headlines_by_window.setdefault(base_window, []).append(row["headline"])
    return headlines_by_window


def get_half_month_windows(start_year: int, end_year: int, round: int = 1) -> list[dict]:
    suffix = f"-r{round}" if round > 1 else ""
    windows = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            _, last_day = monthrange(year, month)
            windows.append({
                "start": f"{year}-{month:02d}-01",
                "end": f"{year}-{month:02d}-15",
                "label": f"{year}-{month:02d}-H1{suffix}",
            })
            windows.append({
                "start": f"{year}-{month:02d}-16",
                "end": f"{year}-{month:02d}-{last_day:02d}",
                "label": f"{year}-{month:02d}-H2{suffix}",
            })
    return windows


def estimate_event_count(start_date: str) -> tuple[int, int]:
    month = int(start_date.split("-")[1])
    year = int(start_date.split("-")[0])
    ym = f"{year}-{month:02d}"

    crisis_windows = [
        ("2020-02", "2020-05"),
        ("2022-02", "2022-04"),
        ("2022-09", "2022-11"),
        ("2023-03", "2023-04"),
    ]
    for cs, ce in crisis_windows:
        if cs <= ym <= ce:
            return 12, 20

    if month in [7, 8, 12]:
        return 6, 12

    return 8, 16


def load_progress(log_file: Path) -> dict:
    if log_file.exists():
        with open(log_file) as f:
            return json.load(f)
    return {"completed_windows": [], "failed_windows": []}


def save_progress(log_file: Path, progress: dict):
    with open(log_file, "w") as f:
        json.dump(progress, f, indent=2)


def parse_json_response(text: str) -> list[dict]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()
    if cleaned.startswith("json"):
        cleaned = cleaned[4:].strip()
    return json.loads(cleaned)


# ---------------------------------------------------------------------------
# Async generation
# ---------------------------------------------------------------------------

async def generate_single_window(
    client: anthropic.AsyncAnthropic,
    window: dict,
    semaphore: asyncio.Semaphore,
    progress: dict,
    progress_lock: asyncio.Lock,
    csv_lock: asyncio.Lock,
    index: int,
    total: int,
    counters: dict,
    round: int = 1,
    existing_headlines_map: dict[str, list[str]] | None = None,
):
    label = window["label"]

    if label in progress["completed_windows"]:
        return

    async with semaphore:
        if round > 1:
            events_min, events_max = 5, 8
            base_label = label.split("-r")[0]
            existing = existing_headlines_map.get(base_label, []) if existing_headlines_map else []
            headlines_list = "\n".join(f"- {h}" for h in existing) if existing else "(none)"
            prompt = TOPUP_PROMPT_TEMPLATE.format(
                start_date=window["start"],
                end_date=window["end"],
                events_min=events_min,
                events_max=events_max,
                existing_headlines=headlines_list,
            )
        else:
            events_min, events_max = estimate_event_count(window["start"])
            prompt = GENERATION_PROMPT_TEMPLATE.format(
                start_date=window["start"],
                end_date=window["end"],
                events_min=events_min,
                events_max=events_max,
            )

        try:
            response = await client.messages.create(
                model=MODEL_GENERATE,
                max_tokens=4096,
                temperature=TEMPERATURE_GENERATE,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT_GENERATE,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": prompt}],
            )

            events = parse_json_response(response.content[0].text)
            for event in events:
                event["window"] = label

            # Thread-safe CSV write
            async with csv_lock:
                with open(RAW_CSV, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                    writer.writerows(events)

            # Thread-safe progress update
            async with progress_lock:
                progress["completed_windows"].append(label)
                counters["total_events"] += len(events)
                counters["completed"] += 1
                save_progress(LOG_FILE, progress)

            print(f"  [{counters['completed']}/{total}] {label} -> {len(events)} events ({counters['total_events']} total)")

        except json.JSONDecodeError as e:
            async with progress_lock:
                progress["failed_windows"].append({"label": label, "error": f"JSON: {e}"})
                counters["failed"] += 1
            print(f"  [{label}] JSON error: {e}")

        except anthropic.RateLimitError:
            async with progress_lock:
                progress["failed_windows"].append({"label": label, "error": "rate_limit"})
                counters["failed"] += 1
            print(f"  [{label}] Rate limited (will retry on re-run)")

        except anthropic.APIError as e:
            async with progress_lock:
                progress["failed_windows"].append({"label": label, "error": str(e)})
                counters["failed"] += 1
            print(f"  [{label}] API error: {e}")


async def generate_events_async(max_concurrent: int, resume: bool = True, round: int = 1):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    progress = load_progress(LOG_FILE) if resume else {"completed_windows": [], "failed_windows": []}

    # On resume, clear old failed windows so they get retried
    if resume:
        progress["failed_windows"] = []

    windows = get_half_month_windows(START_YEAR, END_YEAR, round=round)
    pending = [w for w in windows if w["label"] not in progress["completed_windows"]]

    print(f"Total windows: {len(windows)}  (round {round})")
    print(f"Already done: {len(windows) - len(pending)}")
    print(f"Pending: {len(pending)}")
    print(f"Max concurrent: {max_concurrent}\n")

    if not pending:
        print("All windows already generated. Use --no-resume to start fresh.")
        return

    # Load existing headlines for dedup when doing top-up rounds
    existing_headlines_map = load_existing_headlines(RAW_CSV) if round > 1 else None

    # Initialize CSV header if needed
    if not RAW_CSV.exists() or not resume:
        with open(RAW_CSV, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    semaphore = asyncio.Semaphore(max_concurrent)
    progress_lock = asyncio.Lock()
    csv_lock = asyncio.Lock()
    counters = {"total_events": 0, "completed": 0, "failed": 0}

    tasks = [
        generate_single_window(
            client, w, semaphore, progress, progress_lock, csv_lock,
            i, len(pending), counters,
            round=round, existing_headlines_map=existing_headlines_map,
        )
        for i, w in enumerate(pending)
    ]

    await asyncio.gather(*tasks)
    save_progress(LOG_FILE, progress)

    print(f"\nDone. Generated {counters['total_events']} events in {counters['completed']} windows.")
    print(f"Failed: {counters['failed']} windows.")
    if counters["failed"] > 0:
        print("Re-run the same command to retry failed windows.")


# ---------------------------------------------------------------------------
# Async verification
# ---------------------------------------------------------------------------

async def verify_single_batch(
    client: anthropic.AsyncAnthropic,
    batch: list[dict],
    semaphore: asyncio.Semaphore,
    csv_lock: asyncio.Lock,
    batch_num: int,
    total_batches: int,
    counters: dict,
):
    async with semaphore:
        events_for_verify = [
            {k: v for k, v in e.items() if k not in ("window", "input_idx")}
            for e in batch
        ]

        prompt = VERIFICATION_PROMPT_TEMPLATE.format(
            events_json=json.dumps(events_for_verify, indent=2)
        )

        try:
            response = await client.messages.create(
                model=MODEL_VERIFY,
                max_tokens=4096,
                temperature=TEMPERATURE_VERIFY,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT_VERIFY,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": prompt}],
            )

            verified = parse_json_response(response.content[0].text)

            for j, event in enumerate(verified):
                if j < len(batch):
                    event["window"] = batch[j].get("window", "")
                    event["input_idx"] = batch[j].get("input_idx", "")

            verified_fieldnames = FIELDNAMES + ["original_date", "status", "input_idx"]
            async with csv_lock:
                with open(VERIFIED_CSV, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=verified_fieldnames)
                    for event in verified:
                        writer.writerow({k: event.get(k, "") for k in verified_fieldnames})

            confirmed = sum(1 for e in verified if e.get("status") == "confirmed")
            corrected = sum(1 for e in verified if e.get("status") == "date_corrected")
            fabricated = sum(1 for e in verified if e.get("status") == "fabricated")
            unverified = sum(1 for e in verified if e.get("status") == "unverified")
            counters["confirmed"] += confirmed
            counters["corrected"] += corrected
            counters["fabricated"] += fabricated
            counters["unverified"] += unverified

            print(f"  [Batch {batch_num}/{total_batches}] OK "
                  f"(confirmed: {confirmed}, corrected: {corrected}, fabricated: {fabricated}, unverified: {unverified})")

        except (json.JSONDecodeError, anthropic.APIError, anthropic.RateLimitError) as e:
            counters["errors"] += 1
            print(f"  [Batch {batch_num}/{total_batches}] Error: {e}")


async def verify_events_async(max_concurrent: int, batch_size: int = 20, resume: bool = True):
    if not RAW_CSV.exists():
        print("Error: raw events file not found. Run generate phase first.")
        return

    with open(RAW_CSV) as f:
        all_events = list(csv.DictReader(f))

    # Tag each raw event with a stable index so we can resume partial runs
    for i, e in enumerate(all_events):
        e["input_idx"] = str(i)

    verified_fieldnames = FIELDNAMES + ["original_date", "status", "input_idx"]

    # Figure out which input indices have already been verified
    already_done: set[str] = set()
    if resume and VERIFIED_CSV.exists():
        with open(VERIFIED_CSV) as f:
            reader = csv.DictReader(f)
            # Backwards-compat: if old file lacks input_idx, rewrite from scratch
            if "input_idx" not in (reader.fieldnames or []):
                already_done = set()
            else:
                already_done = {row["input_idx"] for row in reader if row.get("input_idx")}

    # Initialize verified CSV header only if fresh start
    if not resume or not VERIFIED_CSV.exists() or not already_done:
        with open(VERIFIED_CSV, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=verified_fieldnames).writeheader()

    pending = [e for e in all_events if e["input_idx"] not in already_done]

    print(f"Loaded {len(all_events)} events for verification")
    print(f"Already verified: {len(already_done)}")
    print(f"Pending: {len(pending)}")
    print(f"Max concurrent: {max_concurrent}")
    print(f"Batch size: {batch_size}\n")

    if not pending:
        print("All events already verified. Use --no-resume to start fresh.")
        return

    client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    semaphore = asyncio.Semaphore(max_concurrent)
    csv_lock = asyncio.Lock()
    counters = {"confirmed": 0, "corrected": 0, "fabricated": 0, "unverified": 0, "errors": 0}

    batches = [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]

    tasks = [
        verify_single_batch(
            client, batch, semaphore, csv_lock,
            i + 1, len(batches), counters,
        )
        for i, batch in enumerate(batches)
    ]

    await asyncio.gather(*tasks)

    print(f"\nVerification complete.")
    print(f"  Confirmed: {counters['confirmed']}")
    print(f"  Date corrected: {counters['corrected']}")
    print(f"  Fabricated: {counters['fabricated']}")
    print(f"  Unverified: {counters['unverified']}")
    print(f"  Errors: {counters['errors']}")
    if counters["errors"] > 0:
        print("Re-run the same command to retry failed batches.")


# ---------------------------------------------------------------------------
# Merge (sync, no API calls)
# ---------------------------------------------------------------------------

def merge_results():
    if not VERIFIED_CSV.exists():
        print("Error: verified events file not found. Run verify phase first.")
        return

    with open(VERIFIED_CSV) as f:
        all_events = list(csv.DictReader(f))

    clean_events = [e for e in all_events if e.get("status") != "fabricated"]
    fabricated_count = len(all_events) - len(clean_events)

    # Deduplicate by headline (case-insensitive)
    seen: set[str] = set()
    deduped: list[dict] = []
    for e in clean_events:
        key = e.get("headline", "").strip().lower()
        if key not in seen:
            seen.add(key)
            deduped.append(e)
    dup_count = len(clean_events) - len(deduped)
    clean_events = deduped

    clean_events.sort(key=lambda e: e.get("date", ""))

    final_fieldnames = ["headline", "date", "market_fact", "impact", "category", "status"]
    with open(FINAL_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=final_fieldnames)
        writer.writeheader()
        for event in clean_events:
            writer.writerow({k: event.get(k, "") for k in final_fieldnames})

    total = len(clean_events)
    print(f"Final dataset: {total} events")
    print(f"Removed: {fabricated_count} fabricated, {dup_count} duplicate events")

    print("\n--- Distribution ---")

    years = Counter(e["date"][:4] for e in clean_events if e.get("date"))
    print("\nBy year:")
    for year in sorted(years):
        print(f"  {year}: {years[year]}")

    cats = Counter(e.get("category", "unknown") for e in clean_events)
    print("\nBy category:")
    for cat, count in cats.most_common():
        print(f"  {cat}: {count} ({count/total*100:.1f}%)")

    impacts = Counter(e.get("impact", "unknown") for e in clean_events)
    print("\nBy impact:")
    for impact, count in impacts.most_common():
        print(f"  {impact}: {count} ({count/total*100:.1f}%)")

    print("\n--- Quality checks ---")

    low_pct = impacts.get("Low", 0) / total * 100
    print(f"  Low impact: {low_pct:.1f}% {'OK' if low_pct >= 20 else 'WARNING (target: 25%+)'}")

    for year in range(START_YEAR, END_YEAR + 1):
        count = years.get(str(year), 0)
        status = "OK" if count >= 300 else "WARNING (target: 400-600)"
        print(f"  {year}: {count} events {status}")

    unverified = sum(1 for e in clean_events if e.get("status") == "unverified")
    if unverified:
        print(f"  Unverified events: {unverified}")

    print(f"\nSaved to: {FINAL_CSV}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ShockChain Event Generation Pipeline")
    parser.add_argument("--phase", choices=["generate", "verify", "merge"], required=True)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT_DEFAULT)
    parser.add_argument("--verify-batch-size", type=int, default=20)
    parser.add_argument("--round", type=int, default=1, help="Generation round (2+ for top-up passes)")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if args.phase in ["generate", "verify"] and not api_key:
        print(f"Error: ANTHROPIC_API_KEY not found. Create .env at:\n  {ENV_PATH}")
        return

    if args.phase == "generate":
        asyncio.run(generate_events_async(args.max_concurrent, resume=not args.no_resume, round=args.round))
    elif args.phase == "verify":
        asyncio.run(verify_events_async(args.max_concurrent, args.verify_batch_size, resume=not args.no_resume))
    elif args.phase == "merge":
        merge_results()


if __name__ == "__main__":
    main()