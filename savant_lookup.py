"""Shared Baseball Savant lookups: the bat-tracking leaderboard, and resolving a
player name to an MLBAM id.

The leaderboard is the only source here that maps names to ids, so anything
accepting a name goes through ``resolve_player``. Names are matched loosely --
accents, case, punctuation and first/last order are all ignored -- but never
ambiguously: two players sharing a name is always surfaced, either as an error
when it cannot be resolved or as a warning when handedness settles it.

    resolve_player("Luis Arraez", 2026, "L")   -> PlayerRef(650333, 'Arraez, Luis', 'L')
    resolve_player(650333, 2026, "L")          -> same, ids pass straight through
"""

from __future__ import annotations

import io
import re
import unicodedata
import warnings
from dataclasses import dataclass

import pandas as pd
import requests

LEADERBOARD_URL = (
    "https://baseballsavant.mlb.com/leaderboard/bat-tracking?gameType=Regular"
    "&groupBy=bat_side&minSwings=1&minGroupSwings=1"
    "&seasonStart={year}&seasonEnd={year}&type=batter&csv=true"
)

_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}
_LEADERBOARD_CACHE: dict[int, pd.DataFrame] = {}


class SavantError(RuntimeError):
    """Base for lookup and extraction failures."""


class AmbiguousPlayerError(SavantError):
    """More than one player answers to the requested name."""

    def __init__(self, query: str, matches: pd.DataFrame):
        self.query = query
        self.matches = matches
        listing = "; ".join(
            f"{r['name']} (id {int(r['id'])}, bats {r['bat_side']})"
            for _, r in matches.iterrows()
        )
        super().__init__(
            f"{matches['id'].nunique()} players match {query!r}: {listing}. "
            "Pass the MLBAM id instead, or narrow it with handedness."
        )


def display_name(leaderboard_name: str) -> str:
    """Turn the leaderboard's 'Caminero, Junior' into 'Junior Caminero'."""
    last, _, first = str(leaderboard_name).partition(",")
    first, last = first.strip(), last.strip()
    return f"{first} {last}" if first else last


@dataclass(frozen=True)
class PlayerRef:
    """A resolved player."""

    mlbam_id: int
    name: str | None
    bat_side: str | None

    @property
    def display_name(self) -> str | None:
        """Name in natural order, for titles."""
        return display_name(self.name) if self.name else None


def fetch_bat_tracking(
    year: int,
    timeout: float = 60.0,
    session: requests.Session | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Bat-tracking leaderboard for one season, one row per player and bat side."""
    year = int(year)
    if use_cache and year in _LEADERBOARD_CACHE:
        return _LEADERBOARD_CACHE[year].copy()

    get = session.get if session is not None else requests.get
    resp = get(
        LEADERBOARD_URL.format(year=year),
        timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))

    required = {"id", "name", "bat_side", "avg_bat_speed", "swing_length"}
    missing = required - set(df.columns)
    if missing:
        raise SavantError(
            f"bat-tracking leaderboard for {year} is missing {sorted(missing)}; "
            "the endpoint's schema may have changed."
        )
    if use_cache:
        _LEADERBOARD_CACHE[year] = df.copy()
    return df


def _normalize(text: str) -> str:
    """Fold accents and punctuation so 'Rodríguez, Julio' matches 'julio rodriguez'."""
    decomposed = unicodedata.normalize("NFKD", str(text))
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", stripped)).strip().lower()


def _name_keys(leaderboard_name: str) -> set[str]:
    """Every spelling of a 'Last, First' leaderboard name we accept as a match."""
    last, _, first = str(leaderboard_name).partition(",")
    last, first = _normalize(last), _normalize(first)
    if not first:  # single-token name
        return {last} if last else set()

    bare = " ".join(w for w in last.split() if w not in _NAME_SUFFIXES)
    keys = {f"{first} {last}", f"{last} {first}", last}
    if bare and bare != last:  # 'acuna' as well as 'acuna jr'
        keys |= {f"{first} {bare}", f"{bare} {first}", bare}
    return keys


def resolve_player(
    player: int | str,
    year: int | str,
    handedness: str | None = None,
    leaderboard: pd.DataFrame | None = None,
    session: requests.Session | None = None,
) -> PlayerRef:
    """Resolve an MLBAM id or a player name to a :class:`PlayerRef`.

    Ids pass through untouched -- no leaderboard fetch is needed unless a name
    is given. Names match on any of first-last, last-first, or last name alone,
    ignoring case, accents, punctuation and generational suffixes.

    Raises :class:`AmbiguousPlayerError` when a name maps to more than one
    player and ``handedness`` does not settle it; warns when handedness *does*
    settle it, so a shared name is never resolved silently.
    """
    handedness = str(handedness).upper() if handedness else None

    text = str(player).strip()
    if text.isdigit():
        return PlayerRef(int(text), None, handedness)

    if leaderboard is None:
        leaderboard = fetch_bat_tracking(year, session=session)

    query = _normalize(text)
    if not query:
        raise SavantError("empty player name")

    hits = leaderboard[leaderboard["name"].map(lambda n: query in _name_keys(n))]
    if hits.empty:
        raise SavantError(
            f"no player matching {text!r} in the {year} bat-tracking leaderboard. "
            "Try 'First Last', or pass the MLBAM id."
        )

    shared = hits["id"].nunique() > 1
    narrowed = hits
    if handedness is not None:
        by_side = hits[hits["bat_side"].str.upper() == handedness]
        if by_side.empty:
            sides = ", ".join(sorted(hits["bat_side"].unique()))
            raise SavantError(
                f"{hits.iloc[0]['name']} has no {handedness} bat side in {year} "
                f"(available: {sides})."
            )
        narrowed = by_side

    if narrowed["id"].nunique() > 1:
        raise AmbiguousPlayerError(text, narrowed)

    row = narrowed.iloc[0]
    if shared:
        others = hits[hits["id"] != row["id"]]
        warnings.warn(
            f"{text!r} is shared by {hits['id'].nunique()} players; resolved to "
            f"id {int(row['id'])} (bats {row['bat_side']}) by handedness. Others: "
            + "; ".join(
                f"id {int(r['id'])} bats {r['bat_side']}" for _, r in others.iterrows()
            ),
            stacklevel=2,
        )
    return PlayerRef(int(row["id"]), str(row["name"]), str(row["bat_side"]))
