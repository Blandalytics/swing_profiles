"""Rendering for swing profiles: bat speed and acceleration as stacked panels.

Pure presentation -- the numbers come from ``swing_profile``, and a
:class:`~swing_profile.SwingProfile` carries its own identity and imputed
duration, so rendering needs no further arguments.

    from swing_profile import get_swing_profile
    from swing_plot import plot_swing_kinematics

    profile = get_swing_profile("Junior Caminero", 2026, "R")
    plot_swing_kinematics(profile, save_path=profile.filename())

A bare DataFrame still works when the labelling is passed alongside it.
"""

from __future__ import annotations

import os
import re
import warnings

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from PIL import Image

from swing_profile import SwingProfile, add_kinematics, peak_rows

# Fixed y-limits, so panels are comparable across players at a glance. The
# acceleration ceiling depends on the panel's units: g when a duration is
# supplied, mph per unit swing time otherwise. The normalized ceiling carries
# more headroom because that unit scales with swing duration, so a long swing
# reaches a larger number for the same physical acceleration.
SPEED_YLIM = 90.0  # mph
ACCEL_YLIM_G = 47.0  # g
ACCEL_YLIM_NORMALIZED = 200.0  # mph per unit swing time

# Jerk is asymmetric league-wide (2026: -2577 to +1511 g/s) because every
# hitter's sharpest event is the late let-off, not the build. The band is fixed
# and lopsided to match, rather than centred on zero.
JERK_YLIM = (-2700.0, 1600.0)  # g per second

_THEMES = {
    "pitcherlist": dict(
        surface="#292C42",  # page + panel ground
        text="#FFFFFF",  # labels, ticks, annotations
        header="#00D4FF",
        subheader="#8D96B3",
        chrome="#8D96B3",  # axes, spines, gridlines, footer
        velocity="#00D4FF",
        accel="#F1C647",
        jerk="#FFFFFF",
    ),
}

# Gridlines take the chrome color at low alpha: same hue as the axes, but
# recessive enough that the data reads first.
_GRID_ALPHA = 0.3

_ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
_FONT_FAMILY = "DM Sans"
_FONT_URL = (
    "https://raw.githubusercontent.com/google/fonts/main/ofl/dmsans/"
    "DMSans%5Bopsz%2Cwght%5D.ttf"
)
_FALLBACK_FONTS = ["Segoe UI", "DejaVu Sans", "sans-serif"]
_WORDMARK_URL = (
    "https://res.cloudinary.com/dduabusaf/image/upload/v1772839288/"
    "PitcherList_Stats_watermark_with_logo_k9e3xa.webp"
)

_wordmark_cache = None  # None = untried, False = unavailable, else ndarray
_fonts_registered = False


def _cached_download(url: str, filename: str, timeout: float = 30.0) -> str | None:
    """Fetch an asset once into ./assets, returning its path (None if it fails)."""
    path = os.path.join(_ASSET_DIR, filename)
    if os.path.exists(path):
        return path
    try:
        os.makedirs(_ASSET_DIR, exist_ok=True)
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        with open(path, "wb") as fh:
            fh.write(resp.content)
        return path
    except Exception:
        return None


def _static_instance(var_path: str, out_path: str, weight: int, subfamily: str) -> bool:
    """Freeze the variable font at one weight into a static TTF.

    Google ships DM Sans only as a variable font, and matplotlib cannot select
    a weight along a variable axis -- it reads one face per file. Without this,
    asking for bold silently renders regular.
    """
    try:
        from fontTools.ttLib import TTFont
        from fontTools.varLib import instancer

        font = TTFont(var_path)
        axes = {a.axisTag: a.defaultValue for a in font["fvar"].axes}
        pins = {"wght": weight}
        if "opsz" in axes:
            pins["opsz"] = axes["opsz"]
        inst = instancer.instantiateVariableFont(font, pins, inplace=False)
        inst["OS/2"].usWeightClass = weight
        # Pin the names so matplotlib sees one family with two weights, rather
        # than two families ("DM Sans" and "DM Sans Bold").
        for nid, value in (
            (1, _FONT_FAMILY),
            (2, subfamily),
            (4, f"{_FONT_FAMILY} {subfamily}"),
            (6, f"DMSans-{subfamily}"),
            (16, _FONT_FAMILY),
            (17, subfamily),
        ):
            inst["name"].setName(value, nid, 3, 1, 0x409)
        inst.save(out_path)
        return True
    except Exception:
        return False


def _font_stack() -> list[str]:
    """DM Sans (regular + bold) first, fetching and building it once if needed."""
    global _fonts_registered
    if _fonts_registered:
        return [_FONT_FAMILY] + _FALLBACK_FONTS

    have = {f.weight for f in fm.fontManager.ttflist if f.name == _FONT_FAMILY}
    if not {400, 700} <= have:
        var_path = _cached_download(_FONT_URL, "DMSans.ttf")
        if var_path:
            for weight, subfamily in ((400, "Regular"), (700, "Bold")):
                static = os.path.join(_ASSET_DIR, f"DMSans-{subfamily}.ttf")
                if not os.path.exists(static):
                    _static_instance(var_path, static, weight, subfamily)
                if os.path.exists(static):
                    try:
                        fm.fontManager.addfont(static)
                    except Exception:
                        pass

    have = {f.weight for f in fm.fontManager.ttflist if f.name == _FONT_FAMILY}
    if not have:
        warnings.warn(
            f"{_FONT_FAMILY} unavailable; falling back to " f"{_FALLBACK_FONTS[0]}.",
            stacklevel=3,
        )
        return _FALLBACK_FONTS
    if 700 not in have:
        warnings.warn(
            f"{_FONT_FAMILY} bold unavailable; the header will render "
            "at regular weight.",
            stacklevel=3,
        )

    _fonts_registered = True
    return [_FONT_FAMILY] + _FALLBACK_FONTS


def _load_wordmark() -> np.ndarray | None:
    """The PitcherList wordmark as RGBA, or None if it cannot be fetched."""
    global _wordmark_cache
    if _wordmark_cache is None:
        path = _cached_download(_WORDMARK_URL, "wordmark.webp")
        try:
            _wordmark_cache = np.asarray(Image.open(path).convert("RGBA"))
        except Exception:
            _wordmark_cache = False
    return None if _wordmark_cache is False else _wordmark_cache


def _break_after_first_sentence(text: str) -> str:
    """Put the opening sentence on its own line when more than one follows."""
    parts = re.split(r"(?<=\.)\s+", text.strip(), maxsplit=1)
    return "\n".join(parts)


def _style_axis(ax, c):
    """Recessive grid and axes in the chrome color; tick labels in text color."""
    ax.set_facecolor(c["surface"])
    ax.grid(True, color=c["chrome"], linewidth=0.8, alpha=_GRID_ALPHA, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["chrome"])
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=c["text"], labelsize=9, length=0)


def plot_swing_kinematics(
    data: SwingProfile | pd.DataFrame,
    player_name: str | None = None,
    year: int | str | None = None,
    handedness: str | None = None,
    theme: str = "pitcherlist",
    smooth_window: int = 15,
    swing_duration: float | None = None,
    speed_ylim: float = SPEED_YLIM,
    accel_ylim: float | None = None,
    show_jerk: bool = False,
    jerk_ylim: tuple | None = JERK_YLIM,
    save_path: str | None = None,
    figsize: tuple | None = None,
    show_wordmark: bool = True,
):
    """Plot bat speed and its derivative on a shared x-axis.

    ``data`` is a :class:`~swing_profile.SwingProfile` -- whose identity and
    imputed duration fill in the header, subheader and units automatically -- or
    a bare DataFrame with the labelling passed alongside it.

    Two measures on different scales get two panels, never one pair of twinned
    y-axes. Both panels always share one x-axis in one unit: normalized swing
    time by default, or elapsed milliseconds when a duration is known -- which
    also puts acceleration in g.

    Both y-axes are fixed (``speed_ylim`` mph, ``accel_ylim`` in the panel's own
    units) so players can be compared across figures. A series that would exceed
    its ceiling is warned about rather than silently clipped.
    Returns ``(fig, (ax_velocity, ax_accel))``.
    """
    if theme not in _THEMES:
        raise ValueError(f"theme must be one of {sorted(_THEMES)}")
    c = _THEMES[theme]
    if figsize is None:
        figsize = (8.0, 9.2) if show_jerk else (8.0, 7.0)

    # A profile already knows who it is and how long the swing took, so the
    # labels cannot drift out of step with the curve being drawn.
    if isinstance(data, SwingProfile):
        profile, df = data, data.data
        player_name = player_name if player_name is not None else profile.display_name
        year = year if year is not None else profile.year
        handedness = handedness if handedness is not None else profile.handedness
        if swing_duration is None:
            swing_duration = profile.duration_s
    else:
        df = data

    # Two schemas reach here and both spell a column "swing_time" -- milliseconds
    # in swing_profile's output, 0-1 in add_kinematics' raw frame. Tell them
    # apart explicitly; guessing would silently plot one as the other.
    output_schema = "standardized_time" in df.columns

    if output_schema:
        frame = df
        if swing_duration is None:
            swing_duration = float(df["swing_time"].iloc[-1]) / 1000.0
        real_time = True
        v = frame["swing_speed"].to_numpy()
        a = frame["acceleration"].to_numpy()
        j = frame["jerk"].to_numpy() if "jerk" in frame.columns else None
        t = frame["swing_time"].to_numpy()
    else:
        real_time = swing_duration is not None
        needs = "accel_g" if real_time else "accel_mph_per_t"
        frame = (
            df
            if needs in df.columns
            else add_kinematics(
                df, smooth_window=smooth_window, swing_duration=swing_duration
            )
        )
        v = frame["bat_speed_mph"].to_numpy()
        a = frame["accel_g" if real_time else "accel_mph_per_t"].to_numpy()
        jcol = "jerk_g_per_s" if real_time else "jerk_mph_per_t2"
        j = frame[jcol].to_numpy() if jcol in frame.columns else None
        t = (frame["time_ms"] if real_time else frame["swing_time"]).to_numpy()

    if show_jerk and j is None:
        raise KeyError(
            "no jerk column in the input; re-run add_kinematics or "
            "get_swing_profile to add it"
        )

    # One x vector drives both panels, so they cannot drift into different units.
    if real_time:
        x_label = "Elapsed time (ms)"
        peak_fmt = "peak {a:.0f} g at {t:.0f} ms"
    else:
        x_label = "Swing time (0 = start, 1 = impact)"
        peak_fmt = "peak {a:.0f} at t = {t:.2f}"

    if accel_ylim is None:
        accel_ylim = ACCEL_YLIM_G if real_time else ACCEL_YLIM_NORMALIZED
    for series, ceiling, what in (
        (v, speed_ylim, "bat speed"),
        (a, accel_ylim, "acceleration"),
    ):
        if series.max() > ceiling:
            warnings.warn(
                f"{what} peaks at {series.max():.1f}, above the fixed axis limit "
                f"of {ceiling:g}; the curve is clipped. Raise the limit to show it.",
                stacklevel=2,
            )

    span = t[-1] - t[0]

    def label_side(x):
        """Put a marker label on whichever side keeps it inside the axes."""
        if (x - t[0]) / span > 0.62:
            return "right", x - 0.02 * span
        return "left", x + 0.02 * span

    with plt.rc_context(
        {"font.family": "sans-serif", "font.sans-serif": _font_stack()}
    ):
        n_panels = 3 if show_jerk else 2
        fig, axes = plt.subplots(
            n_panels,
            1,
            figsize=figsize,
            sharex=True,
            gridspec_kw={"height_ratios": [1] * n_panels, "hspace": 0.22},
        )
        ax_v, ax_a = axes[0], axes[1]
        ax_j = axes[2] if show_jerk else None
        fig.patch.set_facecolor(c["surface"])
        for ax in axes:
            _style_axis(ax, c)

        # --- velocity ------------------------------------------------------
        ax_v.plot(
            t, v, color=c["velocity"], linewidth=2.0, solid_capstyle="round", zorder=3
        )
        ax_v.plot(
            [t[-1]],
            [v[-1]],
            "o",
            markersize=8,
            color=c["velocity"],
            markeredgecolor=c["surface"],
            markeredgewidth=2,
            zorder=4,
        )
        # Sit the label just above the point, but never push it past the ceiling.
        ax_v.annotate(
            f"{v[-1]:.1f} mph at impact",
            xy=(t[-1] - 0.01 * span, min(v[-1] + 0.03 * speed_ylim, 0.93 * speed_ylim)),
            ha="right",
            va="bottom",
            fontsize=10,
            color=c["text"],
        )
        ax_v.set_ylabel("Bat speed (mph)", fontsize=10, color=c["text"])
        ax_v.set_ylim(0, speed_ylim)

        # --- derivative ----------------------------------------------------
        peak = int(np.argmax(a))
        ax_a.axhline(0, color=c["chrome"], linewidth=1.0, zorder=1)
        ax_a.plot(
            t[4:-5], a[4:-5], color=c["accel"], linewidth=2.0, solid_capstyle="round", zorder=3
        )
        ax_a.plot(
            [t[peak]],
            [a[peak]],
            "o",
            markersize=8,
            color=c["accel"],
            markeredgecolor=c["surface"],
            markeredgewidth=2,
            zorder=4,
        )
        ha_a, x_a = label_side(t[peak])
        ax_a.annotate(
            peak_fmt.format(a=a[peak], t=t[peak]),
            xy=(x_a, min(a[peak] + 0.02 * accel_ylim, 0.93 * accel_ylim)),
            ha=ha_a,
            va="bottom",
            fontsize=10,
            color=c["text"],
        )
        ax_a.set_ylabel(
            "Acceleration (g)" if real_time else "Acceleration (mph per unit t)",
            fontsize=10,
            color=c["text"],
        )
        ax_a.set_ylim(min(0, a.min() * 1.1), accel_ylim)
        ax_a.set_xlim(t[0], t[-1])

        # --- jerk ----------------------------------------------------------
        if show_jerk:
            trough = int(np.argmin(j))
            ax_j.axhline(0, color=c["chrome"], linewidth=1.0, zorder=1)
            ax_j.plot(
                t[9:-10], j[9:-10], color=c["jerk"], linewidth=2.0, solid_capstyle="round", zorder=3
            )
            ax_j.plot(
                [t[trough]],
                [j[trough]],
                "o",
                markersize=8,
                color=c["jerk"],
                markeredgecolor=c["surface"],
                markeredgewidth=2,
                zorder=4,
            )
            lo, hi = jerk_ylim if jerk_ylim else (j.min() * 1.15, j.max() * 1.25)
            # The let-off is the informative extreme, so label the trough.
            label = (
                f"let-off {j[trough]:,.0f} g/s at {t[trough]:.0f} ms"
                if real_time
                else f"min {j[trough]:,.0f} at t = {t[trough]:.2f}"
            )
            ha_j, x_j = label_side(t[trough])
            ax_j.annotate(
                label,
                xy=(x_j, max(j[trough] - 0.02 * (hi - lo), lo + 0.06 * (hi - lo))),
                ha=ha_j,
                va="top",
                fontsize=10,
                color=c["text"],
            )
            ax_j.set_ylabel(
                "Jerk (g per second)" if real_time else "Jerk (mph per unit t squared)",
                fontsize=10,
                color=c["text"],
            )
            ax_j.set_ylim(lo, hi)
            ax_j.set_xlim(t[0], t[-1])
            if j.max() > hi or j.min() < lo:
                warnings.warn(
                    f"jerk spans {j.min():.0f} to {j.max():.0f}, outside the fixed "
                    f"axis {lo:.0f} to {hi:.0f}; the curve is clipped.",
                    stacklevel=2,
                )

        axes[-1].set_xlabel(x_label, fontsize=10, color=c["text"])

        # The player is the title; each panel's y-label names its own series, so
        # neither panel needs a title or a legend box of its own.
        # Keep the header block a fixed distance from the top in inches, so it
        # does not drift when a third panel makes the figure taller.
        head_y = 1 - 0.28 / figsize[1]
        sub_y = 1 - 0.62 / figsize[1]
        if player_name:
            fig.text(
                0.10,
                head_y,
                str(player_name),
                ha="left",
                va="top",
                fontsize=16,
                fontweight="bold",
                color=c["header"],
            )
        if year is not None and handedness:
            hand = f"{str(handedness).upper()}HB"
            fig.text(
                0.10,
                sub_y,
                f"{year} Swing Metrics over time, {hand}",
                ha="left",
                va="top",
                fontsize=10,
                color=c["subheader"],
            )

        note = (
            (
                "Derived from Baseball Savant images and data. Duration imputed "
                f"as swing length / mean bat speed = {swing_duration * 1000:.0f} ms."
            )
            if real_time
            else (
                "Derived from Baseball Savant images and data. Time is normalized, "
                "so the derivative is mph per unit swing time, not acceleration."
            )
        )
        fig.text(
            0.10,
            0.28 / figsize[1],
            _break_after_first_sentence(note),
            ha="left",
            va="bottom",
            fontsize=7.5,
            color=c["chrome"],
            linespacing=1.6,
        )

        fig.subplots_adjust(
            top=1 - 0.95 / figsize[1], bottom=1.08 / figsize[1], left=0.10, right=0.96
        )

        if show_wordmark:
            mark = _load_wordmark()
            if mark is not None:
                w_frac = 0.18
                h_frac = (
                    w_frac * (figsize[0] / figsize[1]) * (mark.shape[0] / mark.shape[1])
                )
                ax_logo = fig.add_axes(
                    [0.96 - w_frac, 0.24 / figsize[1], w_frac, h_frac], zorder=5
                )
                ax_logo.imshow(mark, interpolation="antialiased")
                ax_logo.axis("off")
                ax_logo.patch.set_alpha(0)
        if save_path:
            fig.savefig(save_path, dpi=200, facecolor=c["surface"], edgecolor="none")

    return fig, tuple(axes)


def plot_peak_time_kde(
    data: pd.DataFrame,
    year: int | str | None = None,
    theme: str = "pitcherlist",
    bw_adjust: float = 1.0,
    bins: float = 0.02,
    save_path: str | None = None,
    figsize: tuple = (8.0, 4.8),
    show_wordmark: bool = True,
):
    """KDE of *when* in the swing each hitter reaches peak acceleration.

    ``data`` is either an :data:`~swing_profile.OUTPUT_COLUMNS` frame (peaks are
    taken per player and bat side) or an already-reduced frame from
    :func:`~swing_profile.peak_rows`. One observation per player-side.

    The histogram behind the curve is the actual data: a KDE alone hides sample
    size and the card's 0.01 grid, and would imply more resolution than exists.
    """
    from scipy.stats import gaussian_kde

    if theme not in _THEMES:
        raise ValueError(f"theme must be one of {sorted(_THEMES)}")
    c = _THEMES[theme]

    # Already one row per player-side? Then it came from peak_rows; otherwise
    # it is a full curve frame and the peaks still need picking out.
    keys = ["MLBAMID", "Hand"]
    reduced = all(k in data.columns for k in keys) and not data.duplicated(keys).any()
    peaks = data if reduced else peak_rows(data)
    t = peaks["standardized_time"].to_numpy(dtype=float)
    if t.size < 2:
        raise ValueError("need at least two players to estimate a density")

    if year is None and "Season" in peaks.columns and peaks["Season"].nunique() == 1:
        year = int(peaks["Season"].iloc[0])

    kde = gaussian_kde(t)
    kde.set_bandwidth(kde.factor * bw_adjust)
    grid = np.linspace(0.0, 1.0, 400)
    density = kde(grid)

    edges = np.arange(
        np.floor(t.min() / bins) * bins, np.ceil(t.max() / bins) * bins + bins / 2, bins
    )
    counts, edges = np.histogram(t, bins=edges, density=True)
    median = float(np.median(t))

    with plt.rc_context(
        {"font.family": "sans-serif", "font.sans-serif": _font_stack()}
    ):
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(c["surface"])
        _style_axis(ax, c)

        ax.bar(
            edges[:-1],
            counts,
            width=bins,
            align="edge",
            color=c["accel"],
            alpha=0.38,
            linewidth=0,
            zorder=2,
        )
        ax.plot(
            grid,
            density,
            color=c["accel"],
            linewidth=2.0,
            solid_capstyle="round",
            zorder=3,
        )

        top = max(density.max(), counts.max()) * 1.22
        ax.plot(
            [median, median],
            [0, kde(median)[0]],
            color=c["chrome"],
            linewidth=1.2,
            linestyle=(0, (4, 3)),
            zorder=4,
        )
        ax.annotate(
            f"median {median:.2f}",
            xy=(median, kde(median)[0]),
            xytext=(6, 6),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=10,
            color=c["text"],
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, top)
        ax.set_xlabel(
            "Standardized swing time at peak acceleration " "(0 = start, 1 = impact)",
            fontsize=10,
            color=c["text"],
        )
        ax.set_ylabel("Density", fontsize=10, color=c["text"])

        fig.text(
            0.10,
            0.965,
            "Peak Acceleration Timing",
            ha="left",
            va="top",
            fontsize=16,
            fontweight="bold",
            color=c["header"],
        )
        sub = f"{year} Swing Metrics, " if year else ""
        fig.text(
            0.10,
            0.892,
            f"{sub}{len(t)} player-sides",
            ha="left",
            va="top",
            fontsize=10,
            color=c["subheader"],
        )

        fig.text(
            0.10,
            0.045,
            _break_after_first_sentence(
                "Derived from Baseball Savant images and data. Each hitter "
                "contributes one peak, taken from their own swing-path card."
            ),
            ha="left",
            va="bottom",
            fontsize=7.5,
            color=c["chrome"],
            linespacing=1.6,
        )

        fig.subplots_adjust(top=0.80, bottom=0.235, left=0.10, right=0.96)

        if show_wordmark:
            mark = _load_wordmark()
            if mark is not None:
                w_frac = 0.18
                h_frac = (
                    w_frac * (figsize[0] / figsize[1]) * (mark.shape[0] / mark.shape[1])
                )
                ax_logo = fig.add_axes([0.96 - w_frac, 0.038, w_frac, h_frac], zorder=5)
                ax_logo.imshow(mark, interpolation="antialiased")
                ax_logo.axis("off")
                ax_logo.patch.set_alpha(0)
        if save_path:
            fig.savefig(save_path, dpi=200, facecolor=c["surface"], edgecolor="none")

    return fig, ax


if __name__ == "__main__":
    from swing_profile import get_swing_profile

    demo = get_swing_profile("Junior Caminero", 2026, "R")
    out = demo.filename()
    plot_swing_kinematics(demo, save_path=out)
    print(
        f"{demo.display_name}: {demo.duration_ms:.1f} ms, "
        f"peak {demo.peak_accel_g:.1f} g -> {out}"
    )
