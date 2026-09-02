"""Extract the bat-speed-vs-swing-time curve from a Baseball Savant swing-path poster.

The "splendid splinter" posterized cards render a small chart in the bottom-left
panel: bat speed (mph) on y, normalized swing time on x, running from the start
of the swing to impact. This module fetches that PNG and digitizes the chart back
into a DataFrame.

    df = get_swing_path_data(691406, 2026, "R")
    df = get_swing_path_data("Luis Arraez", 2026, "L")   # a name works too

Calibration notes (all verified against 13 cards spanning 63-80 mph):
  * Cards are a fixed 1280x720 template, itself a 2x upscale of a 640x360 render,
    so the effective resolution is 0.5px -- about 0.45 mph.
  * The panel background is (4, 28, 64) on every card regardless of team colors,
    so the teal marker color is safe to threshold on.
  * The y scale comes from the 30/60/90 axis gridlines (34 px per 30 mph). Read
    back against each card's printed "Bat Speed" value, impact speeds land within
    0.7 mph.
  * Markers sit on a 6px grid (~32 of them) with a larger marker drawn at impact.
"""

from __future__ import annotations

import io
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import requests
from PIL import Image

from savant_lookup import (  # noqa: F401
    AmbiguousPlayerError,
    PlayerRef,
    SavantError,
    fetch_bat_tracking,
    resolve_player,
)

URL_TEMPLATE = (
    "https://builds.mlbstatic.com/baseballsavant.mlb.com/swing-path/"
    "splendid-splinter/posterized/{mlbam_id}-{year}-{handedness}.png"
)

EXPECTED_SIZE = (1280, 720)

# Search window for the bottom-left panel, in image pixels.
_WIN_X0, _WIN_X1 = 5, 230
_WIN_Y0, _WIN_Y1 = 545, 680

# Axis anchors, in image pixels.
_X_START = 19.5  # center of the first marker  -> swing_time 0.0
_X_IMPACT = 208.5  # center of the impact marker -> swing_time 1.0
_Y_ZERO = 664.2  # row of 0 mph
_PX_PER_MPH = 34.0 / 30.0  # from the 30 / 60 / 90 gridlines

_MIN_TRACED_COLUMNS = 60  # sanity floor; real cards trace 145-170 columns


class SwingPathError(SavantError):
    """Raised when a card is missing or does not match the expected template."""


def build_url(mlbam_id: int | str, year: int | str, handedness: str) -> str:
    """URL of the posterized swing-path card."""
    return URL_TEMPLATE.format(
        mlbam_id=mlbam_id, year=year, handedness=str(handedness).upper()
    )


def fetch_card(
    player: int | str,
    year: int | str,
    handedness: str,
    timeout: float = 20.0,
    session: requests.Session | None = None,
) -> Image.Image:
    """Download the card as an RGB image. ``player`` may be an id or a name."""
    mlbam_id = resolve_player(player, year, handedness, session=session).mlbam_id
    url = build_url(mlbam_id, year, handedness)
    get = session.get if session is not None else requests.get
    resp = get(url, timeout=timeout)
    if resp.status_code == 404:
        raise SwingPathError(
            f"No swing-path card for {mlbam_id}-{year}-{handedness} ({url}). "
            "Check the MLBAM id, season, and batting handedness."
        )
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def _marker_mask(rgb: np.ndarray) -> np.ndarray:
    """Boolean mask of the teal curve markers."""
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    return (g > 80) & (b > 80) & ((g - r) > 35) & (np.abs(g - b) < 35)


def _smooth(t: np.ndarray, v: np.ndarray, bandwidth: float) -> np.ndarray:
    """Local linear regression over swing time; flattens pixel quantization.

    Local *linear* rather than a plain kernel average: a kernel average is biased
    at the ends of the range, which is exactly where the two values worth trusting
    live (0 mph at the start, the printed bat speed at impact).
    """
    if not bandwidth:
        return v
    d = t[None, :] - t[:, None]  # (target, source)
    w = np.exp(-0.5 * (d / bandwidth) ** 2)
    s0 = w.sum(axis=1)
    s1 = (w * d).sum(axis=1)
    s2 = (w * d**2).sum(axis=1)
    t0 = (w * v).sum(axis=1)
    t1 = (w * d * v).sum(axis=1)
    denom = s0 * s2 - s1**2
    # Fall back to the kernel average wherever the local fit is degenerate.
    safe = np.abs(denom) > 1e-12
    out = np.where(safe, (s2 * t0 - s1 * t1) / np.where(safe, denom, 1.0), t0 / s0)
    return out


def extract_swing_curve(
    img: Image.Image, n_points: int = 101, smooth: float = 0.03
) -> pd.DataFrame:
    """Digitize the bat-speed chart out of an already-loaded card.

    `n_points` samples on a uniform swing_time grid; `smooth` is the Gaussian
    bandwidth in swing-time units (set 0 to keep the raw pixel trace).
    """
    if img.size != EXPECTED_SIZE:
        raise SwingPathError(
            f"Expected a {EXPECTED_SIZE[0]}x{EXPECTED_SIZE[1]} card, got {img.size}. "
            "The template may have changed; the pixel anchors would need rechecking."
        )

    window = np.asarray(img)[_WIN_Y0:_WIN_Y1, _WIN_X0:_WIN_X1]
    mask = _marker_mask(window)

    rows = np.arange(mask.shape[0])
    cols, mids = [], []
    for col in range(mask.shape[1]):
        ys = rows[mask[:, col]]
        if ys.size:
            cols.append(col)
            mids.append(0.5 * (ys[0] + ys[-1]))

    if len(cols) < _MIN_TRACED_COLUMNS:
        raise SwingPathError(
            f"Only traced {len(cols)} columns of the curve (expected >= "
            f"{_MIN_TRACED_COLUMNS}). The card may be blank or the template changed."
        )

    x_px = np.asarray(cols, dtype=float) + _WIN_X0
    y_px = np.asarray(mids, dtype=float) + _WIN_Y0

    t = (x_px - _X_START) / (_X_IMPACT - _X_START)
    mph = (_Y_ZERO - y_px) / _PX_PER_MPH
    mph = _smooth(t, mph, smooth)

    grid = np.linspace(0.0, 1.0, n_points)
    return pd.DataFrame(
        {
            "swing_time": grid,
            "bat_speed_mph": np.clip(np.interp(grid, t, mph), 0.0, None).round(3),
        }
    )


def get_swing_path_data(
    player: int | str,
    year: int | str,
    handedness: str,
    n_points: int = 101,
    smooth: float = 0.03,
    include_keys: bool = False,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch a card and return its swing_time / bat_speed_mph curve.

    ``player`` is an MLBAM id or a player name (see
    :func:`savant_lookup.resolve_player`). Set `include_keys` to prepend
    mlbam_id / name / year / handedness columns, which makes frames from several
    players safe to concatenate.
    """
    ref = resolve_player(player, year, handedness, session=session)
    img = fetch_card(ref.mlbam_id, year, handedness, session=session)
    df = extract_swing_curve(img, n_points=n_points, smooth=smooth)
    if include_keys:
        df.insert(0, "handedness", str(handedness).upper())
        df.insert(0, "year", int(year))
        df.insert(0, "name", ref.name)
        df.insert(0, "mlbam_id", ref.mlbam_id)
    return df


def get_many(
    keys: Iterable[Sequence],
    n_points: int = 101,
    smooth: float = 0.03,
    skip_missing: bool = True,
) -> pd.DataFrame:
    """Stack curves for many (player, year, handedness) triples into one frame.

    ``player`` may be an id or a name in each triple.
    """
    session = requests.Session()
    frames = []
    for player, year, handedness in keys:
        try:
            frames.append(
                get_swing_path_data(
                    player,
                    year,
                    handedness,
                    n_points=n_points,
                    smooth=smooth,
                    include_keys=True,
                    session=session,
                )
            )
        except SavantError:
            if not skip_missing:
                raise
    if not frames:
        return pd.DataFrame(
            columns=[
                "mlbam_id",
                "name",
                "year",
                "handedness",
                "swing_time",
                "bat_speed_mph",
            ]
        )
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    demo = get_swing_path_data("Junior Caminero", 2026, "R")
    print(demo.iloc[::10].to_string(index=False))
    print(f"\nimpact bat speed: {demo['bat_speed_mph'].iloc[-1]:.1f} mph")
