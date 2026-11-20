"""Streamlit dashboard of the maintenance digital twin (spec §7.14).

Rendering only; logic lives in src/dashboard/. Structure (farms,
turbines, models, alerts, fault events) comes from the RDF graph, time
series from the turbine Parquet files (only the columns a view needs).
Run with `make app`.
"""

import json
import sys
import time
from datetime import datetime, timedelta
from datetime import time as clock_time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import load_config  # noqa: E402
from src.dashboard import replay as rp  # noqa: E402
from src.dashboard.data import (  # noqa: E402
    component_scores,
    detail_columns,
    farm_normal,
    read_columns,
    with_model_ids,
)
from src.dashboard.health import (  # noqa: E402
    component_level,
    farm_health,
    health_table,
)
from src.dashboard.ontology_view import (  # noqa: E402
    component_keys,
    for_components,
    lineage_dot,
    schema_dot,
)
from src.dashboard.tables import (  # noqa: E402
    drift_view,
    evaluation_view,
    turbine_metrics,
    wake_view,
)
from src.experiments.cross_site import summarise  # noqa: E402
from src.graph.graph_utils import (  # noqa: E402
    ALERTS_QUERY,
    FAULT_EVENTS_QUERY,
    LINEAGE_SUBJECT_QUERIES,
    LINEAGE_TRIPLES_QUERY,
    MODELS_QUERY,
    SCHEMA_CLASSES_QUERY,
    SCHEMA_PROPERTIES_QUERY,
    SENSORS_QUERY,
    TURBINES_QUERY,
    get_alerts,
    get_farms,
    get_fault_events,
    get_lineage,
    get_models,
    get_schema,
    get_sensors,
    get_turbines,
    load_graph,
)
from src.health.scores import residual_models  # noqa: E402
from src.models.modelling import (  # noqa: E402
    prediction_column,
    split_times,
    turbine_paths,
)
from src.report.comparison import cross_site_table, metric_frame  # noqa: E402

CONFIG = load_config()
DASH = CONFIG["dashboard"]
VARIANT_COLORS = CONFIG["report"]["colors"]["variants"]
INK = CONFIG["report"]["colors"]["text"]
MUTED = CONFIG["report"]["colors"]["muted"]
ONTOLOGY = CONFIG["paths"]["ontology"]
MODELS_DIR = Path(CONFIG["paths"]["models"])
FIGURES_DIR = Path(CONFIG["report"]["figures"])
INTERVAL = CONFIG["sampling"]["interval_minutes"]
TRAIN_END, CALIBRATION_END = split_times(CONFIG["train_test_split"])
THERMAL_TARGETS = CONFIG["thermal"]["targets"]  # target -> component


@st.cache_resource
def graph():
    """The RDF graph (read once per server)."""
    return load_graph(ONTOLOGY)


@st.cache_data
def structure(farm_code: str) -> dict:
    """Turbines, models, alerts, fault events and Parquet paths of a farm."""
    g = graph()
    return {
        "turbines": get_turbines(g, farm_code),
        "models": with_model_ids(get_models(g, farm_code)),
        "alerts": get_alerts(g, farm_code),
        "events": get_fault_events(g, farm_code),
        "paths": turbine_paths(g, farm_code),
    }


@st.cache_data
def farms() -> pd.DataFrame:
    """Farms with their dataset attribution."""
    return get_farms(graph())


@st.cache_data
def normal_frame(farm_code: str) -> pd.DataFrame:
    """timestamp, turbine_id, is_normal of every turbine of the farm."""
    return farm_normal(structure(farm_code)["paths"])


@st.cache_data(max_entries=6)
def detail(path: str, columns: tuple[str, ...]) -> pd.DataFrame:
    """The replay columns of one turbine."""
    return read_columns(Path(path), list(columns))


@st.cache_data(max_entries=6, show_spinner="Scoring the farm's models ...")
def scores(farm_code: str, component: str) -> dict[str, pd.Series]:
    """Problem scores of every model of one component of the farm."""
    farm = structure(farm_code)
    models = farm["models"][farm["models"]["component_key"] == component]
    return component_scores(CONFIG, models, farm["paths"])


@st.cache_data
def results(name: str):
    """A models/<name>.json result file (None if missing)."""
    path = MODELS_DIR / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def lineage(farm_code: str, turbine_id: str) -> pd.DataFrame:
    """Lineage triples of one turbine."""
    return get_lineage(graph(), farm_code, turbine_id)


@st.cache_data
def schema() -> str:
    """DOT class diagram of the graph's schema."""
    return schema_dot(*get_schema(graph()))


@st.cache_data
def sensors(farm_code: str, turbine_id: str) -> pd.DataFrame:
    """Sensors of one turbine."""
    return get_sensors(graph(), farm_code, turbine_id)


def thresholds() -> dict[str, float]:
    """Alert threshold per model id."""
    return {
        mid: r.get("threshold", float("nan")) for mid, r in results("alerts").items()
    }


def to_naive(ts: pd.Timestamp) -> datetime:
    """UTC timestamp as a naive datetime for the widgets."""
    return ts.tz_convert("UTC").tz_localize(None).to_pydatetime()


def fmt(value, pattern: str) -> str:
    """Format a value, a dash when missing."""
    return "–" if value is None or pd.isna(value) else pattern.format(value)


def set_playing(playing: bool) -> None:
    """Button callback: start or stop the replay."""
    st.session_state.playing = playing


def set_replay_time(first: pd.Timestamp, last: pd.Timestamp) -> None:
    """Date/clock callback: replay time from the picked date and time."""
    state = st.session_state
    picked = pd.Timestamp(
        datetime.combine(state.replay_date, state.replay_clock), tz="UTC"
    )
    state.replay_time = to_naive(rp.clamp(picked, first, last))


def status_map(turbines: pd.DataFrame, statuses: dict[str, str]) -> go.Figure:
    """Map of the turbines coloured by their status at the replay time."""
    data = turbines.assign(status=turbines["turbine_id"].map(statuses))
    colors = DASH["status_colors"]
    fig = px.scatter_map(
        data,
        lat="lat",
        lon="long",
        color="status",
        color_discrete_map=colors,
        category_orders={"status": list(colors)},
        hover_name="turbine_id",
        text="turbine_id",
        center=rp.map_center(turbines),
        zoom=DASH["map_zoom"],
        height=330,
        map_style=DASH["map_style"],
    )
    fig.update_traces(
        marker={"size": 16},
        textposition="top right",
        textfont={"color": "white", "size": 13},
    )
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0}, legend={"orientation": "h", "y": 1.02}
    )
    return fig


def shade(fig: go.Figure, periods: pd.DataFrame, start, end, color: str) -> None:
    """Shade alert periods inside [start, end]."""
    for period in rp.alerts_overlapping(periods, start, end).itertuples():
        fig.add_vrect(
            x0=max(period.start, start),
            x1=min(period.end, end),
            fillcolor=color,
            opacity=0.12,
            line_width=0,
        )


def series_chart(
    frame: pd.DataFrame,
    target: str,
    variants: list[str],
    alerts: pd.DataFrame,
    span: tuple[pd.Timestamp, pd.Timestamp],
    title: str,
    unit: str,
) -> go.Figure:
    """Measured signal and expected values per variant, alerts shaded."""
    fig = go.Figure()
    shade(fig, alerts, *span, DASH["status_colors"]["alert"])
    fig.add_scatter(
        x=frame["timestamp"],
        y=frame[target],
        name="measured",
        mode="lines",
        line={"color": INK, "width": 2},
    )
    for variant in variants:
        column = prediction_column(target, variant)
        if column in frame:
            fig.add_scatter(
                x=frame["timestamp"],
                y=frame[column],
                name=variant,
                mode="lines",
                line={"color": VARIANT_COLORS[variant], "width": 1.5},
            )
    fig.update_layout(
        height=320,
        margin={"l": 0, "r": 0, "t": 36, "b": 0},
        title=title,
        yaxis_title=unit,
        legend={"orientation": "h", "y": -0.2},
    )
    return fig


def power_curve(frame: pd.DataFrame, current: pd.Series | None) -> go.Figure:
    """Power curve of (a sample of) normal rows, current sample highlighted."""
    normal = rp.sample_rows(frame[frame["is_normal"]], DASH["power_curve_max_points"])
    fig = go.Figure()
    fig.add_scattergl(
        x=normal["wind_speed"],
        y=normal["power"],
        mode="markers",
        marker={"size": 2, "color": MUTED, "opacity": 0.3},
        name="normal samples",
    )
    if current is not None:
        # WebGL traces draw above SVG ones: the highlight must be WebGL too
        fig.add_scattergl(
            x=[current["wind_speed"]],
            y=[current["power"]],
            mode="markers",
            name="replay time",
            marker={"size": 14, "color": DASH["status_colors"]["alert"]},
        )
    fig.update_layout(
        height=320,
        margin={"l": 0, "r": 0, "t": 36, "b": 0},
        title="Power curve (sample of normal rows)",
        xaxis_title="Wind speed (m/s)",
        yaxis_title="kW",
        legend={"orientation": "h", "y": -0.2},
    )
    return fig


def render_replay(farm_code, farm, turbine_id, target, ts, statuses) -> None:
    """Replay tab: map, turbine state, power and temperature vs expected,
    power curve and the alerts active at the replay time."""
    st.plotly_chart(status_map(farm["turbines"], statuses), width="stretch")
    turbine = farm["turbines"].set_index("turbine_id").loc[turbine_id]
    models = farm["models"][farm["models"]["turbine_id"] == turbine_id]
    power_variants = sorted(
        models.loc[models["model_class"] == "PowerModel", "variant"].unique(),
        key=list(VARIANT_COLORS).index,
    )
    thermal_variants = (
        sorted(
            models.loc[
                (models["model_class"] == "ThermalModel")
                & (models["target"] == target),
                "variant",
            ].unique(),
            key=list(VARIANT_COLORS).index,
        )
        if target
        else []
    )
    columns = detail_columns(power_variants, target, thermal_variants)
    frame = detail(str(farm["paths"][turbine_id]), tuple(columns))
    current = rp.row_at(frame, ts)
    alerts = farm["alerts"][farm["alerts"]["turbine_id"] == turbine_id]

    st.subheader(
        f"{turbine['label']} ({farm_code} {turbine_id}): {statuses[turbine_id]}"
    )
    period = rp.period_of(ts, TRAIN_END, CALIBRATION_END)
    st.badge(
        f"{period} period",
        color={"training": "gray", "calibration": "orange", "test": "blue"}[period],
    )
    cols = st.columns(4)
    cols[0].metric("Actual power", fmt(rp.value_of(current, "power"), "{:,.0f} kW"))
    cols[1].metric(
        "Expected power (physics)",
        fmt(rp.value_of(current, prediction_column("power", "physics")), "{:,.0f} kW"),
    )
    cols[2].metric("Wind speed", fmt(rp.value_of(current, "wind_speed"), "{:.1f} m/s"))
    if target:
        cols[3].metric(
            target.replace("_", " "), fmt(rp.value_of(current, target), "{:.1f} °C")
        )

    shown_power = st.multiselect(
        "Expected power variants",
        power_variants,
        default=[v for v in ("physics", "baseline") if v in power_variants],
    )
    span = (ts - pd.Timedelta(days=DASH["chart_days"]), ts)
    recent = rp.window(frame, ts, DASH["chart_days"])
    left, right = st.columns(2)
    left.plotly_chart(
        series_chart(
            recent,
            "power",
            shown_power,
            alerts[alerts["model_class"] == "PowerModel"],
            span,
            f"Power, last {DASH['chart_days']} days (power alerts red)",
            "kW",
        ),
        width="stretch",
    )
    if target:
        right.plotly_chart(
            series_chart(
                recent,
                target,
                thermal_variants,
                alerts[alerts["target"] == target],
                span,
                f"{target.replace('_', ' ')}, last {DASH['chart_days']} days",
                "°C",
            ),
            width="stretch",
        )
    else:
        right.info("This component has no temperature model.")
    st.plotly_chart(power_curve(frame, current), width="stretch")

    st.subheader("Alerts active at the replay time")
    now = rp.active_alerts(farm["alerts"], ts)
    st.dataframe(
        now[
            [
                "turbine_id",
                "component_key",
                "model_class",
                "target",
                "variant",
                "start",
                "end",
            ]
        ],
        hide_index=True,
        width="stretch",
    )
    with st.expander("SPARQL: turbines and alerts"):
        st.code(TURBINES_QUERY.strip(), language="sparql")
        st.code(ALERTS_QUERY.strip(), language="sparql")


def score_chart(score, threshold, alerts, events, ts, title) -> go.Figure:
    """A model's problem score with its threshold, alerts shaded, the
    component's fault events marked and the replay time."""
    fig = go.Figure()
    if not score.empty:
        shade(
            fig,
            alerts,
            score.index.min(),
            score.index.max(),
            DASH["status_colors"]["alert"],
        )
    fig.add_scatter(
        x=score.index,
        y=score.values,
        name="score",
        mode="lines",
        line={"color": INK, "width": 1.5},
    )
    fig.add_hline(
        y=threshold,
        line={"color": DASH["status_colors"]["alert"], "dash": "dash"},
        annotation_text="alert threshold",
    )
    if not events.empty:
        fig.add_scatter(
            x=events["start"],
            y=[score.min() if not score.empty else 0] * len(events),
            mode="markers",
            name="fault events",
            text=events["message"],
            marker={"symbol": "triangle-up", "size": 9, "color": MUTED},
        )
    fig.add_vline(x=ts, line={"color": MUTED, "width": 1})
    fig.update_layout(
        height=340,
        margin={"l": 0, "r": 0, "t": 36, "b": 0},
        title=title,
        legend={"orientation": "h", "y": -0.2},
    )
    return fig


def render_health(farm_code, farm, turbine_id, component, ts) -> None:
    """Component health tab: farm overview, the turbine's models and one
    model's score over time with alerts and fault events."""
    all_scores = scores(farm_code, component)
    limits = thresholds()
    models = residual_models(
        farm["models"][farm["models"]["component_key"] == component]
    )
    if models.empty:
        st.info(
            "No residual model for this component (yaw alerts come from offset drift)."
        )
        return
    watch = DASH["health_watch_ratio"]
    overview = farm_health(models, all_scores, limits, ts, watch)
    st.subheader(f"{component}: health index per turbine at the replay time")
    st.caption(
        "Health index = problem score / alert threshold; ≥ 1 is an alert, "
        f"≥ {watch} watch. Scores are fleet-relative 24 h means, as in the alerts."
    )
    st.plotly_chart(
        px.bar(
            overview,
            x="turbine_id",
            y="health_index",
            color="level",
            color_discrete_map=DASH["health_colors"],
            height=260,
        )
        .add_hline(y=1, line={"dash": "dash", "color": MUTED})
        .update_layout(margin={"l": 0, "r": 0, "t": 10, "b": 0}),
        width="stretch",
    )
    mine = models[models["turbine_id"] == turbine_id]
    table = health_table(mine, all_scores, limits, ts, watch)
    index, level = component_level(table, watch)
    st.metric(
        f"{turbine_id} {component} health index",
        fmt(index, "{:.2f}"),
        level,
        delta_color="off",
    )
    st.dataframe(table.round(3), hide_index=True, width="stretch")
    chosen = st.selectbox("Model", table["model_id"])
    if chosen:
        alerts = farm["alerts"][farm["alerts"]["model"].str.endswith("#" + chosen)]
        events = farm["events"][
            (farm["events"]["turbine_id"] == turbine_id)
            & (farm["events"]["component_key"] == component)
        ]
        st.plotly_chart(
            score_chart(
                all_scores.get(chosen, pd.Series(dtype=float)),
                limits.get(chosen, float("nan")),
                alerts,
                events,
                ts,
                f"{chosen}: problem score (alerts red, fault events ▲)",
            ),
            width="stretch",
        )


def render_comparison(farm_code) -> None:
    """Model comparison tab: metrics, evaluation, drift, cross-site, wakes."""
    metrics = metric_frame(results("metrics"))
    st.subheader("Test error per turbine and variant")
    st.markdown(
        "Power: nRMSE (RMSE / rated power); pitch: RMSE in degrees above "
        "rated; temperatures: RMSE in °C. Test period 2022+."
    )
    for model_class, value in (
        ("PowerModel", "nrmse"),
        ("PitchModel", "rmse"),
        ("ThermalModel", "rmse"),
    ):
        st.markdown(f"**{model_class}**")
        st.dataframe(
            turbine_metrics(
                metrics, farm_code, model_class, value, VARIANT_COLORS
            ).round(4),
            width="stretch",
        )
    st.subheader("Event-timed detection (alert ≤ 30 days before a fault episode)")
    scope = st.radio("Scope", ["component", "signal"], horizontal=True)
    st.dataframe(
        evaluation_view(results("evaluation")["summary"], farm_code, scope).round(3),
        hide_index=True,
        width="stretch",
    )
    st.subheader("Drift alerts (≥ 3 months above the calibration year)")
    st.dataframe(
        drift_view(results("drift_alerts"), farm_code).round(2),
        hide_index=True,
        width="stretch",
    )
    cross = results("cross_site")
    if cross is not None:
        st.subheader("Cross-site: trained on the test farm → trained on the other farm")
        st.dataframe(
            cross_site_table(summarise(pd.DataFrame(cross)), list(VARIANT_COLORS)),
            width="stretch",
        )
        figure = FIGURES_DIR / "cross_site.png"
        if figure.exists():
            st.image(str(figure))
    wake = results("wake")
    if wake is not None:
        st.subheader(
            f"Wake-aware power: residual bias (%) waked vs free, k = "
            f"{wake['k'].get(farm_code, float('nan')):.3f}"
        )
        st.dataframe(wake_view(wake, farm_code).round(1), width="stretch")
    with st.expander("SPARQL: models"):
        st.code(MODELS_QUERY.strip(), language="sparql")


def render_ontology(farm_code, turbine_id, ts, alerts) -> None:
    """Ontology tab: lineage of the turbine (models with an active alert in
    red), its sensors, the schema and the queries."""
    st.subheader(f"Lineage of {farm_code} {turbine_id}")
    st.markdown(
        "Where the turbine's specs, readings and models come from: static "
        "file, raw SCADA files inside the Zenodo archives, sensors, components "
        "and the models of each component. Models with an alert active at the "
        "replay time are red."
    )
    triples = lineage(farm_code, turbine_id)
    keys = component_keys(triples)
    shown = st.multiselect(
        "Components", keys, default=[k for k in keys if k == "gearbox"] or keys[:1]
    )
    active = set(
        rp.active_alerts(alerts[alerts["turbine_id"] == turbine_id], ts)["model"]
    )
    st.graphviz_chart(
        lineage_dot(for_components(triples, set(shown)), active), width="stretch"
    )
    with st.expander("Lineage SPARQL queries"):
        st.markdown("One query per kind of subject (farm and turbine bound):")
        for kind, sparql in LINEAGE_SUBJECT_QUERIES.items():
            st.markdown(f"*{kind}*")
            st.code(sparql, language="sparql")
        st.markdown("Then the triples of all those subjects:")
        st.code(LINEAGE_TRIPLES_QUERY.strip(), language="sparql")
    with st.expander("Sensors of the turbine"):
        st.code(SENSORS_QUERY.strip(), language="sparql")
        st.dataframe(sensors(farm_code, turbine_id), hide_index=True, width="stretch")
    with st.expander("Fault events query"):
        st.code(FAULT_EVENTS_QUERY.strip(), language="sparql")
    st.subheader("Ontology schema")
    st.graphviz_chart(schema(), width="stretch")
    with st.expander("Schema SPARQL queries"):
        st.code(SCHEMA_CLASSES_QUERY.strip(), language="sparql")
        st.code(SCHEMA_PROPERTIES_QUERY.strip(), language="sparql")


def attribution() -> str:
    """Footer text for both datasets, from the graph."""
    parts = [
        f"{r.label}: {r.publisher}, DOI [{r.doi}](https://doi.org/{r.doi}), {r.license}"
        for r in farms().itertuples()
    ]
    return "Data: " + "; ".join(parts) + "."


def main() -> None:
    """Sidebar controls, the four tabs, the footer and the replay clock."""
    st.set_page_config(page_title="Wind Maintenance Twin", layout="wide")
    farm_rows = farms()
    labels = dict(zip(farm_rows["farm_code"], farm_rows["label"], strict=True))
    st.sidebar.header("Farm and turbine")
    farm_code = st.sidebar.selectbox("Farm", list(labels), format_func=labels.get)
    farm = structure(farm_code)
    normal = normal_frame(farm_code)
    first, last = normal["timestamp"].min(), normal["timestamp"].max()
    turbine_id = st.sidebar.selectbox("Turbine", farm["turbines"]["turbine_id"])
    components = sorted(farm["models"]["component_key"].unique())
    component = st.sidebar.selectbox(
        "Component",
        components,
        index=components.index("gearbox") if "gearbox" in components else 0,
    )
    targets = [t for t, c in THERMAL_TARGETS.items() if c == component]
    target = st.sidebar.selectbox("Temperature", targets) if targets else None

    state = st.session_state
    if state.get("farm") != farm_code:  # new farm: restart at the test period
        state.farm = farm_code
        state.playing = False
        state.replay_time = to_naive(
            rp.replay_start(first, last, CALIBRATION_END, INTERVAL)
        )
    st.sidebar.header("Replay")
    speed = st.sidebar.radio(
        "Speed (per tick)", list(DASH["speeds"]), index=3, horizontal=True
    )
    play, pause = st.sidebar.columns(2)
    play.button("▶ Play", on_click=set_playing, args=(True,), width="stretch")
    pause.button("⏸ Pause", on_click=set_playing, args=(False,), width="stretch")
    if state.playing:
        following, keep = rp.advance(
            pd.Timestamp(state.replay_time, tz="UTC"),
            DASH["speeds"][speed],
            INTERVAL,
            last,
        )
        state.replay_time, state.playing = to_naive(following), keep
    state.replay_date = state.replay_time.date()
    state.replay_clock = state.replay_time.time()
    st.sidebar.date_input(
        "Replay date (UTC)",
        min_value=to_naive(first).date(),
        max_value=to_naive(last).date(),
        key="replay_date",
        on_change=set_replay_time,
        args=(first, last),
    )
    st.sidebar.slider(
        "Time of day (UTC)",
        min_value=clock_time(0, 0),
        max_value=(datetime.min + timedelta(days=1, minutes=-INTERVAL)).time(),
        step=timedelta(minutes=INTERVAL),
        format="HH:mm",
        key="replay_clock",
        on_change=set_replay_time,
        args=(first, last),
    )
    st.sidebar.caption("Playing" if state.playing else "Paused")
    ts = pd.Timestamp(state.replay_time, tz="UTC")
    statuses = rp.farm_statuses(normal, farm["alerts"], ts)

    st.title(f"{labels[farm_code]}: maintenance digital twin")
    st.caption(f"Replay time {ts:%Y-%m-%d %H:%M} UTC")
    tabs = st.tabs(
        ["Replay", "Component health", "Model comparison", "Ontology & lineage"]
    )
    with tabs[0]:
        render_replay(farm_code, farm, turbine_id, target, ts, statuses)
    with tabs[1]:
        render_health(farm_code, farm, turbine_id, component, ts)
    with tabs[2]:
        render_comparison(farm_code)
    with tabs[3]:
        render_ontology(farm_code, turbine_id, ts, farm["alerts"])
    st.divider()
    st.caption(attribution())
    if state.playing:
        time.sleep(DASH["tick_seconds"])
        st.rerun()


main()
