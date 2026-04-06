from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Resistivity Probe Lab",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)


THEMES: dict[str, dict[str, str]] = {
    "Slate Lab": {
        "background": "#0b1220",
        "panel": "#111a2e",
        "panel_alt": "#16223a",
        "text": "#edf4ff",
        "muted": "#9eb1cc",
        "accent": "#55c1ff",
        "accent_2": "#8ef0ff",
        "grid": "rgba(158, 177, 204, 0.16)",
        "border": "rgba(142, 240, 255, 0.18)",
        "shadow": "0 18px 48px rgba(0, 0, 0, 0.28)",
        "sample": "rgba(85, 193, 255, 0.18)",
        "substrate": "rgba(255, 255, 255, 0.06)",
        "success": "#6be3b3",
        "warning": "#ffb86b",
    },
    "Paper Lab": {
        "background": "#f4f7fb",
        "panel": "#ffffff",
        "panel_alt": "#eef4fb",
        "text": "#122033",
        "muted": "#5f728a",
        "accent": "#1378c5",
        "accent_2": "#17b6d4",
        "grid": "rgba(18, 32, 51, 0.10)",
        "border": "rgba(19, 120, 197, 0.14)",
        "shadow": "0 18px 40px rgba(18, 32, 51, 0.08)",
        "sample": "rgba(19, 120, 197, 0.12)",
        "substrate": "rgba(18, 32, 51, 0.05)",
        "success": "#118a63",
        "warning": "#c9781b",
    },
}


def inject_theme(theme: dict[str, str]) -> None:
    st.markdown(
        f"""
        <style>
            :root {{
                --bg: {theme["background"]};
                --panel: {theme["panel"]};
                --panel-alt: {theme["panel_alt"]};
                --text: {theme["text"]};
                --muted: {theme["muted"]};
                --accent: {theme["accent"]};
                --accent2: {theme["accent_2"]};
                --border: {theme["border"]};
                --shadow: {theme["shadow"]};
                --success: {theme["success"]};
                --warning: {theme["warning"]};
            }}
            .stApp {{
                background:
                    radial-gradient(circle at top left, rgba(85, 193, 255, 0.12), transparent 28%),
                    radial-gradient(circle at top right, rgba(23, 182, 212, 0.10), transparent 24%),
                    linear-gradient(180deg, var(--bg), var(--bg));
                color: var(--text);
            }}
            [data-testid="stSidebar"] {{
                background:
                    linear-gradient(180deg, rgba(255, 255, 255, 0.02), rgba(255, 255, 255, 0.00)),
                    var(--panel);
                border-right: 1px solid var(--border);
            }}
            [data-testid="stSidebar"] * {{
                color: var(--text);
            }}
            .block-container {{
                padding-top: 1.7rem;
                padding-bottom: 2rem;
            }}
            .hero {{
                padding: 1.6rem 1.8rem;
                border-radius: 24px;
                background:
                    linear-gradient(135deg, rgba(85, 193, 255, 0.13), rgba(23, 182, 212, 0.04)),
                    var(--panel);
                border: 1px solid var(--border);
                box-shadow: var(--shadow);
                margin-bottom: 1rem;
            }}
            .hero h1 {{
                margin: 0 0 0.35rem 0;
                font-size: 2.15rem;
                letter-spacing: -0.02em;
            }}
            .hero p {{
                margin: 0;
                color: var(--muted);
                line-height: 1.55;
                max-width: 58rem;
            }}
            .chip-row {{
                display: flex;
                flex-wrap: wrap;
                gap: 0.55rem;
                margin-top: 1rem;
            }}
            .chip {{
                padding: 0.38rem 0.72rem;
                border-radius: 999px;
                border: 1px solid var(--border);
                background: rgba(255, 255, 255, 0.03);
                font-size: 0.86rem;
                color: var(--text);
            }}
            .panel {{
                background: var(--panel);
                border: 1px solid var(--border);
                box-shadow: var(--shadow);
                border-radius: 22px;
                padding: 1.1rem 1.15rem;
                margin-bottom: 1rem;
            }}
            .section-title {{
                font-size: 1.02rem;
                font-weight: 700;
                letter-spacing: 0.01em;
                margin-bottom: 0.6rem;
            }}
            .section-copy {{
                color: var(--muted);
                line-height: 1.55;
                margin-bottom: 0.6rem;
            }}
            .metric-card {{
                background: linear-gradient(180deg, rgba(255, 255, 255, 0.02), rgba(255, 255, 255, 0.01)), var(--panel);
                border: 1px solid var(--border);
                border-radius: 20px;
                padding: 1rem 1rem 0.95rem 1rem;
                box-shadow: var(--shadow);
                min-height: 132px;
            }}
            .metric-label {{
                color: var(--muted);
                font-size: 0.85rem;
                letter-spacing: 0.05em;
                text-transform: uppercase;
            }}
            .metric-value {{
                font-size: 1.6rem;
                font-weight: 700;
                margin: 0.3rem 0 0.25rem 0;
                color: var(--text);
                letter-spacing: -0.03em;
            }}
            .metric-note {{
                color: var(--muted);
                font-size: 0.87rem;
                line-height: 1.45;
            }}
            .info-box {{
                border-radius: 20px;
                padding: 1rem 1.05rem;
                background: linear-gradient(135deg, rgba(107, 227, 179, 0.10), rgba(85, 193, 255, 0.06));
                border: 1px solid rgba(107, 227, 179, 0.18);
                line-height: 1.55;
            }}
            .formula-box {{
                border-radius: 20px;
                padding: 0.85rem 1rem;
                background: var(--panel-alt);
                border: 1px solid var(--border);
            }}
            div[data-testid="stDataFrame"] {{
                border-radius: 18px;
                overflow: hidden;
            }}
            .small-note {{
                color: var(--muted);
                font-size: 0.9rem;
            }}
            [data-testid="stNumberInput"] input,
            [data-testid="stTextInput"] input,
            [data-testid="stDataEditor"] input {{
                color: #111111 !important;
                -webkit-text-fill-color: #111111 !important;
                caret-color: #111111 !important;
                font-weight: 600 !important;
            }}
            [data-testid="stNumberInput"] input::placeholder,
            [data-testid="stTextInput"] input::placeholder,
            [data-testid="stDataEditor"] input::placeholder {{
                color: #111111 !important;
                -webkit-text-fill-color: #111111 !important;
                opacity: 0.82 !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def fmt(value: float, unit: str = "", precision: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    magnitude = abs(value)
    if magnitude == 0:
        rendered = "0"
    elif magnitude >= 1_000 or magnitude < 1e-3:
        rendered = f"{value:.{precision}e}"
    else:
        rendered = f"{value:.{precision}f}".rstrip("0").rstrip(".")
    return f"{rendered} {unit}".strip()


def metric_card(label: str, value: str, note: str) -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """


def default_dataset(current: float, voltage: float, count: int) -> pd.DataFrame:
    current = max(current, 1e-6)
    resistance = voltage / current if current else 0.0
    start = max(current * 0.35, 1e-6)
    stop = max(current * 1.65, start + 1e-6)
    currents = np.linspace(start, stop, count)
    voltages = currents * resistance
    return pd.DataFrame(
        {
            "Current (A)": np.round(currents, 6),
            "Voltage (V)": np.round(voltages, 6),
        }
    )


def sanitize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    for column in ["Current (A)", "Voltage (V)"]:
        if column not in clean.columns:
            clean[column] = np.nan
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean = clean[["Current (A)", "Voltage (V)"]]
    clean = clean.dropna(how="all")
    return clean.reset_index(drop=True)


def fit_line(currents: np.ndarray, voltages: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(currents) & np.isfinite(voltages)
    x = currents[mask]
    y = voltages[mask]
    if len(x) == 0:
        return {"slope": math.nan, "intercept": math.nan, "r2": math.nan}
    if len(np.unique(x)) < 2:
        non_zero = x != 0
        if np.any(non_zero):
            slope = float(np.mean(y[non_zero] / x[non_zero]))
            intercept = 0.0
            y_pred = slope * x
        else:
            slope = math.nan
            intercept = math.nan
            y_pred = np.full_like(y, np.nan, dtype=float)
    else:
        slope, intercept = np.polyfit(x, y, 1)
        slope = float(slope)
        intercept = float(intercept)
        y_pred = slope * x + intercept
    if len(x) > 1 and np.all(np.isfinite(y_pred)):
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    else:
        r2 = math.nan
    return {"slope": slope, "intercept": intercept, "r2": r2}


def compute_resistivity_two_probe(
    voltage: float,
    current: float,
    length: float,
    area: float,
    contact_resistance: float = 0.0,
    wire_resistance: float = 0.0,
    fitted_resistance: float | None = None,
) -> dict[str, float]:
    point_resistance = voltage / current if current else math.nan
    measured_resistance = (
        float(fitted_resistance)
        if fitted_resistance is not None and np.isfinite(fitted_resistance)
        else point_resistance
    )
    parasitic = 2 * contact_resistance + wire_resistance
    sample_resistance = measured_resistance - parasitic if np.isfinite(measured_resistance) else math.nan
    resistivity = sample_resistance * area / length if length > 0 and area > 0 else math.nan
    return {
        "point_resistance": point_resistance,
        "measured_resistance": measured_resistance,
        "sample_resistance": sample_resistance,
        "resistivity": resistivity,
        "parasitic_resistance": parasitic,
    }


def compute_resistivity_four_probe(
    voltage: float,
    current: float,
    probe_spacing: float,
    thickness: float,
    model: str = "Bulk",
    fitted_resistance: float | None = None,
    contact_resistance: float = 0.0,
    wire_resistance: float = 0.0,
) -> dict[str, float]:
    point_resistance = voltage / current if current else math.nan
    sensed_resistance = (
        float(fitted_resistance)
        if fitted_resistance is not None and np.isfinite(fitted_resistance)
        else point_resistance
    )
    hypothetical_two_probe = sensed_resistance + (2 * contact_resistance + wire_resistance)
    if model == "Thin Film":
        resistivity = (math.pi / math.log(2)) * thickness * sensed_resistance if thickness > 0 else math.nan
    else:
        resistivity = 2 * math.pi * probe_spacing * sensed_resistance if probe_spacing > 0 else math.nan
    return {
        "point_resistance": point_resistance,
        "measured_resistance": sensed_resistance,
        "resistivity": resistivity,
        "hypothetical_two_probe": hypothetical_two_probe,
    }


def build_derived_dataset(
    df: pd.DataFrame,
    method: str,
    contact_resistance: float,
    wire_resistance: float,
) -> pd.DataFrame:
    derived = sanitize_dataset(df)
    if derived.empty:
        return derived
    currents = derived["Current (A)"].to_numpy(dtype=float)
    voltages = derived["Voltage (V)"].to_numpy(dtype=float)
    parasitic_drop = currents * (2 * contact_resistance + wire_resistance)
    with np.errstate(divide="ignore", invalid="ignore"):
        derived["Resistance (Ohm)"] = np.where(currents != 0, voltages / currents, np.nan)
    if method == "Two-Probe":
        derived["Ideal Voltage (V)"] = voltages - parasitic_drop
        with np.errstate(divide="ignore", invalid="ignore"):
            derived["Corrected R_sample (Ohm)"] = np.where(
                currents != 0, derived["Ideal Voltage (V)"] / currents, np.nan
            )
    else:
        derived["Ideal Voltage (V)"] = voltages
        derived["Hypothetical Two-Probe V (V)"] = voltages + parasitic_drop
        with np.errstate(divide="ignore", invalid="ignore"):
            derived["Hypothetical Two-Probe R (Ohm)"] = np.where(
                currents != 0,
                derived["Hypothetical Two-Probe V (V)"] / currents,
                np.nan,
            )
    return derived


def plot_interactive_graph(
    df: pd.DataFrame,
    method: str,
    theme: dict[str, str],
    show_ideal_vs_real: bool,
    contact_resistance: float,
    wire_resistance: float,
) -> tuple[go.Figure, dict[str, float], dict[str, float] | None]:
    data = sanitize_dataset(df)
    currents = data["Current (A)"].to_numpy(dtype=float) if not data.empty else np.array([])
    voltages = data["Voltage (V)"].to_numpy(dtype=float) if not data.empty else np.array([])
    measured_fit = fit_line(currents, voltages)
    comparison_fit: dict[str, float] | None = None

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=currents,
            y=voltages,
            mode="markers",
            name="Measured data",
            marker=dict(size=10, color=theme["accent"], line=dict(width=1.4, color=theme["panel"])),
            hovertemplate="I = %{x:.4e} A<br>V = %{y:.4e} V<extra>Measured</extra>",
        )
    )

    if np.isfinite(measured_fit["slope"]):
        x_line = np.linspace(np.nanmin(currents), np.nanmax(currents), 200) if len(currents) else np.array([0, 1])
        y_line = measured_fit["slope"] * x_line + measured_fit["intercept"]
        figure.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name=f"Fit: slope {measured_fit['slope']:.4e} Ohm",
                line=dict(color=theme["accent"], width=3),
                hovertemplate="Fit voltage = %{y:.4e} V<extra>Linear fit</extra>",
            )
        )

    if show_ideal_vs_real and not data.empty:
        parasitic_drop = currents * (2 * contact_resistance + wire_resistance)
        if method == "Two-Probe":
            comparison_y = voltages - parasitic_drop
            comparison_name = "Ideal sample-only response"
        else:
            comparison_y = voltages + parasitic_drop
            comparison_name = "Hypothetical two-probe response"
        comparison_fit = fit_line(currents, comparison_y)
        figure.add_trace(
            go.Scatter(
                x=currents,
                y=comparison_y,
                mode="markers+lines",
                name=comparison_name,
                marker=dict(size=7, color=theme["accent_2"]),
                line=dict(color=theme["accent_2"], width=2.2, dash="dash"),
                hovertemplate="I = %{x:.4e} A<br>V = %{y:.4e} V<extra>Comparison</extra>",
            )
        )

    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Current I (A)",
        yaxis_title="Voltage V (V)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        font=dict(color=theme["text"]),
        hoverlabel=dict(bgcolor=theme["panel"], font=dict(color=theme["text"])),
    )
    figure.update_xaxes(showgrid=True, gridcolor=theme["grid"], zeroline=False)
    figure.update_yaxes(showgrid=True, gridcolor=theme["grid"], zeroline=False)
    return figure, measured_fit, comparison_fit


def draw_probe_diagram(method: str, model: str, theme: dict[str, str], show_ideal_vs_real: bool) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=20, b=10),
        showlegend=False,
        font=dict(color=theme["text"]),
    )
    figure.update_xaxes(visible=False, range=[0, 10])
    figure.update_yaxes(visible=False, range=[0, 10], scaleanchor="x", scaleratio=1)

    if method == "Two-Probe":
        figure.add_shape(type="rect", x0=0.8, y0=4.2, x1=9.2, y1=5.8, line=dict(color=theme["accent"], width=2), fillcolor=theme["sample"])
        probe_x = [2.2, 7.8]
        labels = ["I+, V+", "I-, V-"]
        for x_pos, label in zip(probe_x, labels):
            figure.add_shape(type="line", x0=x_pos, y0=8.4, x1=x_pos, y1=5.8, line=dict(color=theme["accent_2"], width=4))
            figure.add_shape(type="circle", x0=x_pos - 0.22, y0=8.45, x1=x_pos + 0.22, y1=8.89, line=dict(color=theme["accent_2"], width=2), fillcolor=theme["panel_alt"])
            figure.add_annotation(x=x_pos, y=9.25, text=label, showarrow=False, font=dict(size=12))
        figure.add_annotation(
            x=7.4,
            y=6.9,
            ax=2.6,
            ay=6.9,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            text="Current flow",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.1,
            arrowwidth=2,
            arrowcolor=theme["accent"],
        )
        figure.add_annotation(x=5.0, y=8.0, text="Voltage measured at the same contacts", showarrow=False, font=dict(size=12, color=theme["muted"]))
        figure.add_shape(type="line", x0=2.2, y0=7.45, x1=7.8, y1=7.45, line=dict(color=theme["accent_2"], width=2, dash="dot"))
        if show_ideal_vs_real:
            figure.add_annotation(
                x=5.0,
                y=3.1,
                text="Real reading includes contact and lead drops",
                showarrow=False,
                font=dict(size=12, color=theme["warning"]),
            )
    else:
        if model == "Thin Film":
            figure.add_shape(type="rect", x0=0.8, y0=3.7, x1=9.2, y1=4.3, line=dict(color=theme["muted"], width=1), fillcolor=theme["substrate"])
            figure.add_shape(type="rect", x0=0.8, y0=4.3, x1=9.2, y1=5.0, line=dict(color=theme["accent"], width=2), fillcolor=theme["sample"])
        else:
            figure.add_shape(type="rect", x0=0.8, y0=4.0, x1=9.2, y1=6.0, line=dict(color=theme["accent"], width=2), fillcolor=theme["sample"])
        probe_x = [1.6, 3.8, 6.2, 8.4]
        labels = ["I+", "V+", "V-", "I-"]
        colors = [theme["accent"], theme["accent_2"], theme["accent_2"], theme["accent"]]
        for x_pos, label, color in zip(probe_x, labels, colors):
            figure.add_shape(type="line", x0=x_pos, y0=8.5, x1=x_pos, y1=5.0 if model == "Thin Film" else 6.0, line=dict(color=color, width=4))
            figure.add_shape(type="circle", x0=x_pos - 0.2, y0=8.48, x1=x_pos + 0.2, y1=8.88, line=dict(color=color, width=2), fillcolor=theme["panel_alt"])
            figure.add_annotation(x=x_pos, y=9.2, text=label, showarrow=False, font=dict(size=12))
        figure.add_annotation(
            x=8.0,
            y=6.9,
            ax=2.0,
            ay=6.9,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            text="Outer probes drive current",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.1,
            arrowwidth=2,
            arrowcolor=theme["accent"],
        )
        figure.add_shape(type="line", x0=3.8, y0=7.35, x1=6.2, y1=7.35, line=dict(color=theme["accent_2"], width=2.5, dash="dot"))
        figure.add_annotation(x=5.0, y=7.95, text="Inner probes sense voltage only", showarrow=False, font=dict(size=12, color=theme["muted"]))
        if show_ideal_vs_real:
            figure.add_annotation(
                x=5.0,
                y=2.8,
                text="Contact drops stay outside the sensed voltage window",
                showarrow=False,
                font=dict(size=12, color=theme["success"]),
            )
    return figure


def render_formulas(
    method: str,
    model: str,
    calc: dict[str, float],
    measured_fit: dict[str, float],
    controls: dict[str, Any],
) -> None:
    st.markdown('<div class="formula-box">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Dynamic Formulas</div>', unsafe_allow_html=True)
    if method == "Two-Probe":
        st.latex(r"R_{\mathrm{measured}} = \frac{V}{I}")
        st.latex(rf"R_{{\mathrm{{sample}}}} = R_{{\mathrm{{measured}}}} - (2R_c + R_w)")
        st.latex(r"\rho = R_{\mathrm{sample}} \cdot \frac{A}{L}")
        st.latex(
            rf"R_{{\mathrm{{fit}}}} = {measured_fit['slope']:.4e}\ \Omega,\quad "
            rf"R_{{\mathrm{{sample}}}} = {calc['sample_resistance']:.4e}\ \Omega"
        )
        st.latex(
            rf"\rho = {calc['sample_resistance']:.4e}\times\frac{{{controls['area']:.4e}}}{{{controls['length']:.4e}}}"
            rf" = {calc['resistivity']:.4e}\ \Omega\cdot m"
        )
    else:
        st.latex(r"R = \frac{V}{I}")
        if model == "Thin Film":
            st.latex(r"\rho = \frac{\pi}{\ln 2}\,t\,\frac{V}{I}")
            st.latex(
                rf"\rho = \frac{{\pi}}{{\ln 2}}\times {controls['thickness']:.4e}\times {measured_fit['slope']:.4e}"
                rf" = {calc['resistivity']:.4e}\ \Omega\cdot m"
            )
        else:
            st.latex(r"\rho = 2\pi s \frac{V}{I}")
            st.latex(
                rf"\rho = 2\pi\times {controls['spacing']:.4e}\times {measured_fit['slope']:.4e}"
                rf" = {calc['resistivity']:.4e}\ \Omega\cdot m"
            )
    st.markdown("</div>", unsafe_allow_html=True)


def render_header(method: str, model: str, show_ideal_vs_real: bool) -> None:
    chips = [
        f"<span class='chip'>Method: {method}</span>",
        f"<span class='chip'>Model: {model if method == 'Four-Probe' else 'Bulk sample path'}</span>",
        f"<span class='chip'>Comparison: {'Ideal vs Real' if show_ideal_vs_real else 'Measurement only'}</span>",
    ]
    st.markdown(
        f"""
        <div class="hero">
            <h1>Two-Probe and Four-Probe Resistivity Workbench</h1>
            <p>
                Explore how geometry, probe architecture, and parasitic resistances reshape resistivity extraction.
                The interface combines editable laboratory readings, automated V-I fitting, and a scientific instrument
                schematic so the measurement logic is visible as well as numerical.
            </p>
            <div class="chip-row">{''.join(chips)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_controls() -> dict[str, Any]:
    st.sidebar.markdown("## Experiment Console")
    theme_name = st.sidebar.radio("Theme", list(THEMES.keys()), index=0, horizontal=True)
    method = st.sidebar.selectbox("Probe method", ["Two-Probe", "Four-Probe"])
    model = "Bulk"
    if method == "Four-Probe":
        model = st.sidebar.radio("Four-probe model", ["Bulk", "Thin Film"], horizontal=True)

    st.sidebar.markdown("### Electrical inputs")
    current = st.sidebar.number_input("Current I (A)", min_value=1e-6, value=0.010, step=0.001, format="%.6f")
    voltage = st.sidebar.number_input("Voltage V (V)", min_value=0.0, value=0.120, step=0.010, format="%.6f")

    st.sidebar.markdown("### Sample parameters")
    length = st.sidebar.number_input("Length L (m)", min_value=1e-6, value=0.020, step=0.001, format="%.6f")
    area = st.sidebar.number_input("Cross-sectional area A (m^2)", min_value=1e-12, value=1.00e-6, step=1e-7, format="%.6e")
    thickness = st.sidebar.number_input("Thickness t (m)", min_value=1e-12, value=2.00e-7, step=1e-8, format="%.6e")
    spacing = st.sidebar.number_input("Probe spacing s (m)", min_value=1e-6, value=1.00e-3, step=1e-4, format="%.6e")

    st.sidebar.markdown("### Non-ideal elements")
    contact_resistance = st.sidebar.number_input("Contact resistance Rc (Ohm)", min_value=0.0, value=0.800, step=0.050, format="%.4f")
    wire_resistance = st.sidebar.number_input("Wire resistance Rw (Ohm)", min_value=0.0, value=0.200, step=0.050, format="%.4f")

    st.sidebar.markdown("### Data controls")
    reading_count = st.sidebar.slider("Auto-generated readings", min_value=4, max_value=12, value=6)
    show_ideal_vs_real = st.sidebar.toggle("Show Ideal vs Real Measurement", value=True)
    regenerate = st.sidebar.button("Regenerate dataset", width="stretch")
    append_reference = st.sidebar.button("Append sidebar reading", width="stretch")

    return {
        "theme_name": theme_name,
        "method": method,
        "model": model,
        "current": current,
        "voltage": voltage,
        "length": length,
        "area": area,
        "thickness": thickness,
        "spacing": spacing,
        "contact_resistance": contact_resistance,
        "wire_resistance": wire_resistance,
        "reading_count": reading_count,
        "show_ideal_vs_real": show_ideal_vs_real,
        "regenerate": regenerate,
        "append_reference": append_reference,
    }


def main() -> None:
    bootstrap_theme = THEMES["Slate Lab"]
    inject_theme(bootstrap_theme)
    controls = sidebar_controls()
    theme = THEMES[controls["theme_name"]]
    inject_theme(theme)

    method = controls["method"]
    model = controls["model"]

    render_header(method, model, controls["show_ideal_vs_real"])

    datasets = st.session_state.setdefault("datasets", {})
    dataset_key = f"{method}_{model}"
    if controls["regenerate"] or dataset_key not in datasets:
        datasets[dataset_key] = default_dataset(
            current=controls["current"],
            voltage=controls["voltage"],
            count=controls["reading_count"],
        )
    if controls["append_reference"]:
        datasets[dataset_key] = pd.concat(
            [
                sanitize_dataset(datasets[dataset_key]),
                pd.DataFrame(
                    [{"Current (A)": controls["current"], "Voltage (V)": controls["voltage"]}]
                ),
            ],
            ignore_index=True,
        )

    summary_left, summary_right = st.columns([1.55, 1.0], gap="large")

    with summary_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Measurement Dataset</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-copy">Edit readings directly, add rows, or regenerate a clean sweep from the sidebar reference point.</div>',
            unsafe_allow_html=True,
        )
        edited_df = st.data_editor(
            datasets[dataset_key],
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            column_config={
                "Current (A)": st.column_config.NumberColumn("Current (A)", format="%.6e"),
                "Voltage (V)": st.column_config.NumberColumn("Voltage (V)", format="%.6e"),
            },
            key=f"editor_{dataset_key}",
        )
        datasets[dataset_key] = sanitize_dataset(edited_df)
        st.markdown("</div>", unsafe_allow_html=True)

    with summary_right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Experiment Snapshot</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="small-note">
                Selected reading: <strong>{fmt(controls["current"], "A")}</strong> and <strong>{fmt(controls["voltage"], "V")}</strong><br>
                Geometry: L = {fmt(controls["length"], "m")}, A = {fmt(controls["area"], "m^2")}, t = {fmt(controls["thickness"], "m")}, s = {fmt(controls["spacing"], "m")}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    dataset = sanitize_dataset(datasets[dataset_key])
    derived_dataset = build_derived_dataset(
        dataset,
        method=method,
        contact_resistance=controls["contact_resistance"],
        wire_resistance=controls["wire_resistance"],
    )

    chart, measured_fit, comparison_fit = plot_interactive_graph(
        dataset,
        method=method,
        theme=theme,
        show_ideal_vs_real=controls["show_ideal_vs_real"],
        contact_resistance=controls["contact_resistance"],
        wire_resistance=controls["wire_resistance"],
    )

    if method == "Two-Probe":
        calc = compute_resistivity_two_probe(
            voltage=controls["voltage"],
            current=controls["current"],
            length=controls["length"],
            area=controls["area"],
            contact_resistance=controls["contact_resistance"],
            wire_resistance=controls["wire_resistance"],
            fitted_resistance=measured_fit["slope"],
        )
        comparison_note = f"Parasitic contribution = {fmt(calc['parasitic_resistance'], 'Ohm')}"
        tertiary_value = fmt(calc["sample_resistance"], "Ohm")
        tertiary_label = "Corrected sample R"
        tertiary_note = "Measured slope after removing two contacts and wire path."
    else:
        calc = compute_resistivity_four_probe(
            voltage=controls["voltage"],
            current=controls["current"],
            probe_spacing=controls["spacing"],
            thickness=controls["thickness"],
            model=model,
            fitted_resistance=measured_fit["slope"],
            contact_resistance=controls["contact_resistance"],
            wire_resistance=controls["wire_resistance"],
        )
        comparison_note = f"Hypothetical two-probe = {fmt(calc['hypothetical_two_probe'], 'Ohm')}"
        tertiary_value = model
        tertiary_label = "Geometry model"
        tertiary_note = "Bulk uses probe spacing; thin film uses thickness correction."

    metric_columns = st.columns(4, gap="medium")
    metric_html = [
        metric_card(
            "Resistance R (V/I)",
            fmt(calc["point_resistance"], "Ohm"),
            "Single selected reading from the sidebar inputs.",
        ),
        metric_card(
            "Slope From Graph",
            fmt(measured_fit["slope"], "Ohm"),
            f"Linear fit on edited V-I data. R^2 = {fmt(measured_fit['r2'])}",
        ),
        metric_card(
            tertiary_label,
            tertiary_value,
            tertiary_note,
        ),
        metric_card(
            "Resistivity rho",
            fmt(calc["resistivity"], "Ohm*m"),
            comparison_note,
        ),
    ]
    for column, html in zip(metric_columns, metric_html):
        with column:
            st.markdown(html, unsafe_allow_html=True)

    if method == "Two-Probe" and np.isfinite(calc["sample_resistance"]) and calc["sample_resistance"] <= 0:
        st.warning("Corrected sample resistance is non-physical. Reduce Rc/Rw or increase measured V/I data.")

    visual_left, visual_right = st.columns([1.6, 1.0], gap="large")
    with visual_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Interactive V-I Characterization</div>', unsafe_allow_html=True)
        st.plotly_chart(chart, width="stretch", config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with visual_right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Probe Arrangement</div>', unsafe_allow_html=True)
        st.plotly_chart(
            draw_probe_diagram(method, model, theme, controls["show_ideal_vs_real"]),
            width="stretch",
            config={"displayModeBar": False},
        )
        st.markdown(
            """
            <div class="info-box">
                <strong>Why four-probe is more accurate</strong><br>
                In a two-probe test, the same contacts carry current and sense voltage, so contact and lead drops are mixed into the sample reading.
                Four-probe separates those jobs: outer probes push current while inner probes measure only the sample potential difference, which sharply reduces parasitic error.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    theory_left, theory_right = st.columns([1.1, 1.2], gap="large")
    with theory_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        render_formulas(method, model, calc, measured_fit, controls)
        st.markdown("</div>", unsafe_allow_html=True)

    with theory_right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Reading Analysis</div>', unsafe_allow_html=True)
        if derived_dataset.empty:
            st.info("Add at least one current-voltage pair to compute derived values.")
        else:
            if method == "Two-Probe":
                styled = derived_dataset.style.format(
                    {
                        "Current (A)": "{:.4e}",
                        "Voltage (V)": "{:.4e}",
                        "Resistance (Ohm)": "{:.4e}",
                        "Ideal Voltage (V)": "{:.4e}",
                        "Corrected R_sample (Ohm)": "{:.4e}",
                    }
                )
            else:
                styled = derived_dataset.style.format(
                    {
                        "Current (A)": "{:.4e}",
                        "Voltage (V)": "{:.4e}",
                        "Resistance (Ohm)": "{:.4e}",
                        "Ideal Voltage (V)": "{:.4e}",
                        "Hypothetical Two-Probe V (V)": "{:.4e}",
                        "Hypothetical Two-Probe R (Ohm)": "{:.4e}",
                    }
                )
            st.dataframe(styled, width="stretch", hide_index=True)

            if controls["show_ideal_vs_real"] and comparison_fit is not None and np.isfinite(comparison_fit["slope"]):
                if method == "Two-Probe":
                    st.caption(
                        f"Ideal sample-only fit slope: {fmt(comparison_fit['slope'], 'Ohm')} after subtracting Rc and Rw."
                    )
                else:
                    st.caption(
                        f"Hypothetical two-probe fit slope: {fmt(comparison_fit['slope'], 'Ohm')} if contact drops were included."
                    )
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
