"""Impute a swing's real duration from Savant's bat-tracking leaderboard.

The swing-path card plots bat speed against *normalized* time, so it carries no
duration of its own. The bat-tracking leaderboard supplies the missing scale:
it reports ``swing_length`` (feet of barrel travel) alongside ``avg_bat_speed``.

The derivation is one substitution. Swing length is the path integral of speed
over the swing::

    s = int_0^T v dt

and with t = tau * T for normalized tau in [0, 1]::

    s = T * int_0^1 v(tau) d(tau) = T * v_mean   =>   T = s / v_mean

where ``v_mean`` is the time-average bat speed over the swing -- obtained by
integrating that player's own digitized curve, not assumed. So the duration
follows from two measured quantities and the shape of the player's own card.

    from swing_duration import impute_swing_duration
    timing = impute_swing_duration("Junior Caminero", 2026, "R")   # or 691406
    print(timing.duration_ms)

What this assumes
-----------------
1. ``swing_length`` tracks the path of the same point whose speed the card
   plots (the barrel / sweet spot), so ``int v dt`` is exactly that path length.
2. Savant's swing-start definition for ``swing_length`` matches the card's
   t = 0. If the two windows differ, T scales proportionally.
3. Both inputs are season averages for that player and bat side, so the result
   is a representative swing, not any individual one.

``speed_check_mph`` guards assumption 1: the card's impact value and the
leaderboard's ``avg_bat_speed`` are independent measurements of the same
quantity, so a large gap means the two sources are not describing the same
swings and the duration should not be trusted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import requests

from savant_lookup import (  # noqa: F401
    LEADERBOARD_URL,
    SavantError,
    fetch_bat_tracking,
    resolve_player,
)
from swing_path_extract import SwingPathError, get_swing_path_data

MPH_TO_FPS = 5280.0 / 3600.0
_SPEED_CHECK_TOLERANCE = 2.0  # mph


@dataclass(frozen=True)
class SwingTiming:
    """Imputed duration plus the quantities it was derived from."""

    mlbam_id: int
    year: int
    handedness: str
    swing_length_ft: float
    leaderboard_bat_speed_mph: float
    card_impact_mph: float
    mean_bat_speed_mph: float
    shape_ratio: float  # mean speed / impact speed, from the card
    duration_s: float
    speed_check_mph: float  # card impact - leaderboard avg_bat_speed
    name: str | None = None

    @property
    def duration_ms(self) -> float:
        return self.duration_s * 1000.0

    def as_dict(self) -> dict:
        d = asdict(self)
        d["duration_ms"] = self.duration_ms
        return d


def mean_bat_speed(curve: pd.DataFrame) -> float:
    """Time-average bat speed (mph) over the swing, by integrating the curve.

    ``swing_time`` spans exactly 0 to 1, so the integral *is* the mean.
    """
    t = curve["swing_time"].to_numpy(dtype=float)
    v = curve["bat_speed_mph"].to_numpy(dtype=float)
    return float(np.trapz(v, t))


def impute_swing_duration(
    player: int | str,
    year: int | str,
    handedness: str,
    curve: pd.DataFrame | None = None,
    leaderboard: pd.DataFrame | None = None,
    session: requests.Session | None = None,
) -> SwingTiming:
    """Impute the real duration of a swing, in seconds.

    ``player`` is an MLBAM id or a player name. Pass an already-extracted
    ``curve`` and/or a cached ``leaderboard`` to avoid refetching when looping
    over players.
    """
    year = int(year)
    handedness = str(handedness).upper()

    if leaderboard is None:
        leaderboard = fetch_bat_tracking(year, session=session)

    ref = resolve_player(
        player, year, handedness, leaderboard=leaderboard, session=session
    )
    mlbam_id = ref.mlbam_id

    row = leaderboard[
        (leaderboard["id"] == mlbam_id)
        & (leaderboard["bat_side"].str.upper() == handedness)
    ]
    if row.empty:
        raise SwingPathError(
            f"{mlbam_id} ({handedness}) is not in the {year} bat-tracking "
            "leaderboard. Check the id, season, and bat side."
        )
    row = row.iloc[0]

    if curve is None:
        curve = get_swing_path_data(mlbam_id, year, handedness, session=session)

    v_mean_mph = mean_bat_speed(curve)
    impact_mph = float(curve["bat_speed_mph"].iloc[-1])
    length_ft = float(row["swing_length"])

    if v_mean_mph <= 0:
        raise SwingPathError(f"non-positive mean bat speed for {mlbam_id}")

    return SwingTiming(
        mlbam_id=mlbam_id,
        year=year,
        handedness=handedness,
        swing_length_ft=length_ft,
        leaderboard_bat_speed_mph=float(row["avg_bat_speed"]),
        card_impact_mph=impact_mph,
        mean_bat_speed_mph=v_mean_mph,
        shape_ratio=v_mean_mph / impact_mph,
        duration_s=length_ft / (v_mean_mph * MPH_TO_FPS),
        speed_check_mph=impact_mph - float(row["avg_bat_speed"]),
        name=str(row["name"]) if "name" in row else None,
    )


def impute_many(
    year: int | str,
    mlbam_ids=None,
    skip_missing: bool = True,
    warn_speed_check: bool = True,
) -> pd.DataFrame:
    """Impute durations for a whole season's leaderboard, or a subset of ids.

    Players without a swing-path card are skipped when ``skip_missing``.
    """
    year = int(year)
    lb = fetch_bat_tracking(year)
    if mlbam_ids is not None:
        lb = lb[lb["id"].isin([int(i) for i in mlbam_ids])]

    session = requests.Session()
    out = []
    for _, r in lb.iterrows():
        try:
            timing = impute_swing_duration(
                r["id"], year, r["bat_side"], leaderboard=lb, session=session
            )
        except SavantError:
            if not skip_missing:
                raise
            continue
        out.append(timing.as_dict())

    df = pd.DataFrame(out)
    if df.empty:
        return df

    cols = [
        "mlbam_id",
        "name",
        "year",
        "handedness",
        "swing_length_ft",
        "leaderboard_bat_speed_mph",
        "card_impact_mph",
        "mean_bat_speed_mph",
        "shape_ratio",
        "duration_s",
        "duration_ms",
        "speed_check_mph",
    ]
    df = df[cols]

    if warn_speed_check:
        bad = df[df["speed_check_mph"].abs() > _SPEED_CHECK_TOLERANCE]
        if not bad.empty:
            print(
                f"warning: {len(bad)} player(s) differ from the leaderboard bat "
                f"speed by more than {_SPEED_CHECK_TOLERANCE} mph; "
                "their durations are suspect."
            )
    return df


if __name__ == "__main__":
    demo = impute_swing_duration(691406, 2026, "R")
    print("Junior Caminero, 2026 (RHB)")
    print(f"  swing length        {demo.swing_length_ft:.2f} ft")
    print(
        f"  mean bat speed      {demo.mean_bat_speed_mph:.2f} mph "
        f"({demo.shape_ratio:.3f} of impact)"
    )
    print(
        f"  impact: card {demo.card_impact_mph:.2f} vs leaderboard "
        f"{demo.leaderboard_bat_speed_mph:.2f} mph  "
        f"({demo.speed_check_mph:+.2f})"
    )
    print(f"  imputed duration    {demo.duration_ms:.1f} ms")
