"""Bat speed and acceleration for one hitter, from card image to numbers.

``get_swing_profile`` is the whole data pipeline in one call: it resolves the
player, digitizes the swing-path card, imputes the swing's real duration from
the bat-tracking leaderboard, and differentiates bat speed.

    from swing_profile import get_swing_profile

    profile = get_swing_profile("Junior Caminero", 2026, "R")
    print(profile.duration_ms, profile.peak_accel_g)
    profile.data.head()          # or get_swing_frame(...) for the frame alone

Columns: ``MLBAMID``, ``Name``, ``Hand``, ``Season``, ``standardized_time``
(0-1, straight off the card), ``swing_time`` (milliseconds), ``swing_speed``
(mph), ``acceleration`` (g), ``jerk`` (g per second).

Plotting lives in ``swing_plot``; nothing here imports matplotlib. The stages
also remain available on their own -- ``swing_path_extract`` for the image,
``swing_duration`` for the timing, ``add_kinematics`` for the derivative.

On units
--------
Time is unitized exactly as the card presents it: 0 at the start of the swing,
1 at impact. ``add_kinematics`` therefore yields **mph per unit swing time**,
not a physical acceleration -- the card carries no duration, so nothing assumes
one. Supply a duration (``get_swing_profile`` imputes one) and the frame also
gains ``time_s``, ``time_ms``, ``accel_fps2`` and ``accel_g``.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
from scipy.signal import savgol_filter

from savant_lookup import SavantError, display_name, fetch_bat_tracking, resolve_player
from swing_duration import SwingTiming, impute_swing_duration
from swing_path_extract import get_swing_path_data

MPH_TO_FPS = 5280.0 / 3600.0
G_FPS2 = 32.174

#: Schema of the frame this module produces. Identity is carried on every row so
#: frames from several players concatenate without losing track of whose swing
#: is whose. ``standardized_time`` is the card's own 0-1 axis; ``swing_time`` is
#: that axis scaled by the imputed duration, in **milliseconds**.
OUTPUT_COLUMNS = [
    "MLBAMID",
    "Name",
    "Hand",
    "Season",
    "standardized_time",
    "swing_time",
    "swing_speed",
    "acceleration",
    "jerk",
]

#: Most leaderboard rows have no swing-path card -- in 2026, 306 of 713. Card
#: availability tracks playing time but the bands overlap (the thinnest card sits
#: at 123 competitive swings; the busiest player without one has 394), so no
#: threshold is exact. 100 is the safe cut: measured against a full 2026 pull it
#: skips 194 doomed requests and misses zero real cards.
MIN_SWINGS = 100


def add_kinematics(
    df: pd.DataFrame,
    smooth_window: int = 15,
    polyorder: int = 3,
    swing_duration: float | None = None,
) -> pd.DataFrame:
    """Add the bat-speed derivatives to a swing-path curve.

    Returns a copy with ``accel_mph_per_t`` -- d(bat_speed_mph)/d(swing_time),
    in mph per unit swing time -- and ``jerk_mph_per_t2``, its derivative again.
    Both come from a Savitzky-Golay filter, which fits a local polynomial rather
    than differencing neighbors, so it does not amplify the digitizer's residual
    pixel noise. Requires a uniform ``swing_time`` grid.

    The default window spans roughly five of the card's ~32 native markers:
    features narrower than that are upsampling artifacts, not swing mechanics.
    Jerk uses the same window and order, so it is the exact analytic derivative
    of the same local fit that produced the acceleration column -- the two
    cannot disagree. It is still the noisiest quantity here: a third derivative
    of a curve digitized at roughly 0.9 mph resolution. Read its shape, not its
    precise magnitude.

    ``swing_duration`` (seconds) is optional and nothing assumes one: supply it
    and the frame also gains ``time_s``, ``time_ms``, ``accel_fps2``,
    ``accel_g`` and ``jerk_g_per_s``. Derive it from a player's own swing length
    rather than guessing -- see ``swing_duration.impute_swing_duration``.
    """
    for col in ("swing_time", "bat_speed_mph"):
        if col not in df.columns:
            raise KeyError(f"input frame needs a {col!r} column")

    out = df.copy()
    t = out["swing_time"].to_numpy(dtype=float)
    if t.size < 5:
        raise ValueError(f"need at least 5 samples to differentiate, got {t.size}")

    steps = np.diff(t)
    if not np.allclose(steps, steps[0], rtol=1e-6, atol=1e-9):
        raise ValueError("swing_time must be uniformly spaced")

    v = out["bat_speed_mph"].to_numpy(dtype=float)
    dt = steps[0]

    window = min(smooth_window, t.size if t.size % 2 else t.size - 1)
    if window % 2 == 0:
        window -= 1
    if window > polyorder:
        accel = savgol_filter(v, window, polyorder, deriv=1, delta=dt)
        jerk = savgol_filter(v, window, polyorder, deriv=2, delta=dt)
    else:  # too few samples to fit the local polynomial
        accel = np.gradient(v, dt)
        jerk = np.gradient(accel, dt)

    out["accel_mph_per_t"] = accel
    out["jerk_mph_per_t2"] = jerk

    if swing_duration is not None:
        if swing_duration <= 0:
            raise ValueError("swing_duration must be positive")
        out["time_s"] = t * swing_duration
        out["time_ms"] = out["time_s"] * 1000.0
        # mph per unit t -> ft/s per second; each further derivative divides by
        # another factor of the duration.
        out["accel_fps2"] = accel * MPH_TO_FPS / swing_duration
        out["accel_g"] = out["accel_fps2"] / G_FPS2
        out["jerk_fps3"] = jerk * MPH_TO_FPS / swing_duration**2
        out["jerk_g_per_s"] = out["jerk_fps3"] / G_FPS2

    return out


def figure_filename(
    mlbam_id: int | str, handedness: str, suffix: str = "", ext: str = "png"
) -> str:
    """Canonical figure name: ``swing_kinematics_{id}_{hand}.png``.

    Handedness belongs in the name, not just the id: a switch hitter has a card
    per bat side under a single MLBAM id, so an id-only name lets one side
    silently overwrite the other.
    """
    tag = f"_{suffix}" if suffix else ""
    return f"swing_kinematics_{int(mlbam_id)}_{str(handedness).upper()}{tag}.{ext}"


@dataclass(frozen=True)
class SwingProfile:
    """One hitter's swing: the digitized curve, its timing, and its derivative."""

    mlbam_id: int
    name: str | None
    year: int
    handedness: str
    timing: SwingTiming
    data: pd.DataFrame

    @property
    def display_name(self) -> str | None:
        """Name in natural order ('Junior Caminero'), for titles."""
        return display_name(self.name) if self.name else None

    @property
    def duration_s(self) -> float:
        return self.timing.duration_s

    @property
    def duration_ms(self) -> float:
        return self.timing.duration_ms

    @property
    def impact_mph(self) -> float:
        return float(self.data["swing_speed"].iloc[-1])

    @property
    def peak_accel_g(self) -> float:
        return float(self.data["acceleration"].max())

    @property
    def peak_accel_ms(self) -> float:
        """When peak acceleration occurs, in elapsed milliseconds."""
        row = self.data["acceleration"].idxmax()
        return float(self.data.loc[row, "swing_time"])

    @property
    def peak_jerk(self) -> float:
        """Largest positive jerk, in g per second."""
        return float(self.data["jerk"].max())

    @property
    def min_jerk(self) -> float:
        """Largest negative jerk -- the hardest let-off, in g per second."""
        return float(self.data["jerk"].min())

    def filename(self, suffix: str = "", ext: str = "png") -> str:
        """This profile's canonical figure name."""
        return figure_filename(self.mlbam_id, self.handedness, suffix, ext)


def get_swing_profile(
    player: int | str,
    year: int | str,
    handedness: str,
    n_points: int = 101,
    smooth: float = 0.03,
    smooth_window: int = 15,
    swing_duration: float | None = None,
    leaderboard: pd.DataFrame | None = None,
    session=None,
) -> SwingProfile:
    """Card image -> bat-speed curve -> imputed duration -> acceleration.

    ``player`` is an MLBAM id or a name. Pass ``swing_duration`` (seconds) to
    override the imputation; otherwise it is derived from the player's own swing
    length and mean bat speed. Pass a cached ``leaderboard`` and a shared
    ``session`` when looping over players.
    """
    year = int(year)
    handedness = str(handedness).upper()

    ref = resolve_player(
        player, year, handedness, leaderboard=leaderboard, session=session
    )
    curve = get_swing_path_data(
        ref.mlbam_id,
        year,
        handedness,
        n_points=n_points,
        smooth=smooth,
        session=session,
    )
    timing = impute_swing_duration(
        ref.mlbam_id,
        year,
        handedness,
        curve=curve,
        leaderboard=leaderboard,
        session=session,
    )
    duration = timing.duration_s if swing_duration is None else float(swing_duration)
    k = add_kinematics(curve, smooth_window=smooth_window, swing_duration=duration)

    name = ref.name or timing.name
    data = pd.DataFrame(
        {
            "MLBAMID": ref.mlbam_id,
            "Name": display_name(name) if name else None,
            "Hand": handedness,
            "Season": year,
            "standardized_time": k["swing_time"].to_numpy(),
            "swing_time": k["time_ms"].to_numpy(),
            "swing_speed": k["bat_speed_mph"].to_numpy(),
            "acceleration": k["accel_g"].to_numpy(),
            "jerk": k["jerk_g_per_s"].to_numpy(),
        }
    )[OUTPUT_COLUMNS]

    return SwingProfile(
        mlbam_id=ref.mlbam_id,
        name=name,
        year=year,
        handedness=handedness,
        timing=timing,
        data=data,
    )


def get_swing_frame(
    player: int | str, year: int | str, handedness: str, **kwargs
) -> pd.DataFrame:
    """Just the frame -- :func:`get_swing_profile` without the timing metadata."""
    return get_swing_profile(player, year, handedness, **kwargs).data


def peak_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per player and bat side, at that swing's peak acceleration.

    Takes an :data:`OUTPUT_COLUMNS` frame (one player or many) and returns the
    row where ``acceleration`` is greatest for each ``MLBAMID``/``Hand``, so
    ``standardized_time`` tells you *when* in the swing the peak lands.
    """
    keys = [k for k in ("MLBAMID", "Hand") if k in frame.columns]
    if not keys:
        raise KeyError("frame needs MLBAMID and Hand columns")
    idx = frame.groupby(keys)["acceleration"].idxmax()
    return frame.loc[idx].reset_index(drop=True)


_thread_local = threading.local()


def _thread_session() -> requests.Session:
    """One HTTP session per worker -- a Session is not safe to share across threads."""
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = _thread_local.session = requests.Session()
    return session


def get_season_frames(
    year: int | str,
    min_swings: int = MIN_SWINGS,
    max_workers: int = 8,
    mlbam_ids=None,
    leaderboard: pd.DataFrame | None = None,
    verbose: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Pull every player on a season's leaderboard into one frame, in parallel.

    The work is almost entirely waiting on the card images, so it scales close to
    linearly with ``max_workers``: measured cold, a season runs ~5.8 min serially,
    ~45 s at 8 workers. 8 is a reasonable default -- past ~16 the gain flattens
    and it is discourteous to the CDN.

    ``min_swings`` skips leaderboard rows too thin to have a card (see
    :data:`MIN_SWINGS`); pass 0 to attempt every row. Rows without a card are
    dropped. Returns the :data:`OUTPUT_COLUMNS` frame, one block of rows per
    player and bat side, in leaderboard order.
    """
    year = int(year)
    lb = fetch_bat_tracking(year) if leaderboard is None else leaderboard

    rows = lb
    if mlbam_ids is not None:
        rows = rows[rows["id"].isin([int(i) for i in mlbam_ids])]
    if min_swings and "swings_competitive" in rows.columns:
        rows = rows[rows["swings_competitive"] >= min_swings]
    rows = list(rows.itertuples(index=False))

    if verbose:
        print(
            f"{year}: {len(rows)} of {len(lb)} rows pass "
            f"min_swings={min_swings}; {max_workers} workers"
        )

    def pull(row):
        try:
            return get_swing_profile(
                int(row.id),
                year,
                row.bat_side,
                leaderboard=lb,
                session=_thread_session(),
                **kwargs,
            ).data
        except SavantError:
            return None  # no card for this player and bat side
        except Exception as exc:  # network hiccup on one row shouldn't kill the run
            if verbose:
                print(f"  {row.id} ({row.bat_side}): {type(exc).__name__}: {exc}")
            return None

    if max_workers and max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            frames = list(pool.map(pull, rows))  # map preserves input order
    else:
        frames = [pull(r) for r in rows]

    found = [f for f in frames if f is not None]
    if verbose:
        print(f"  {len(found)} cards, {len(rows) - len(found)} without")
    if not found:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return pd.concat(found, ignore_index=True)


if __name__ == "__main__":
    profile = get_swing_profile("Junior Caminero", 2026, "R")

    print(f"{profile.display_name} - {profile.year} ({profile.handedness}HB)")
    print(f"  swing length     {profile.timing.swing_length_ft:.2f} ft")
    print(
        f"  impact           {profile.impact_mph:.2f} mph "
        f"(leaderboard {profile.timing.leaderboard_bat_speed_mph:.2f}, "
        f"{profile.timing.speed_check_mph:+.2f})"
    )
    print(f"  imputed duration {profile.duration_ms:.1f} ms")
    print(
        f"  peak accel       {profile.peak_accel_g:.1f} g at "
        f"{profile.peak_accel_ms:.0f} ms"
    )
    print()
    print(profile.data.iloc[::20].to_string(index=False))
