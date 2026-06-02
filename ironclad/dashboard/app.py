"""Streamlit betting intelligence dashboard for ironclad-v2."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ironclad.eval import metrics as m
from ironclad.store.connection import get_connection

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ironclad | Betting Intelligence",
    page_icon="🏈",
    layout="wide",
)


# ── DB connection (one per Streamlit server process) ─────────────────────────

@st.cache_resource
def _conn():
    return get_connection(read_only=True)


# ── Cached data loaders (TTL = 5 min) ────────────────────────────────────────

@st.cache_data(ttl=300)
def load_edges() -> pd.DataFrame:
    try:
        return _conn().execute(
            "SELECT * FROM gold.betting_edges ORDER BY analyzed_at DESC"
        ).df()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_results() -> pd.DataFrame:
    try:
        return _conn().execute(
            "SELECT * FROM gold.betting_results ORDER BY settled_at ASC"
        ).df()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_backtest() -> pd.DataFrame:
    try:
        row = _conn().execute(
            "SELECT backtest_run_id FROM gold.backtest_predictions "
            "ORDER BY predicted_at DESC LIMIT 1"
        ).fetchone()
        if not row:
            return pd.DataFrame()
        return _conn().execute(
            "SELECT * FROM gold.backtest_predictions WHERE backtest_run_id = ? "
            "ORDER BY season, week",
            [row[0]],
        ).df()
    except Exception:
        return pd.DataFrame()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _empty(msg: str) -> None:
    st.info(msg)


def _metrics(cols, items: list[tuple[str, str]]) -> None:
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)


# ── App shell ─────────────────────────────────────────────────────────────────

st.title("ironclad  |  Betting Intelligence")
tab_edges, tab_pnl, tab_health = st.tabs(["Edges", "P&L", "Model Health"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1: EDGES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_edges:
    df_e = load_edges()

    if df_e.empty:
        _empty("No betting edges found. Run: **ironclad edges --save** to analyze props and persist results.")
    else:
        with st.sidebar:
            st.header("Edge Filters")
            game_ids = ["All"] + sorted(df_e["game_id"].dropna().unique().tolist())
            sel_game = st.selectbox("Game ID", game_ids)
            stat_types = sorted(df_e["stat_type"].dropna().unique().tolist())
            sel_stats = st.multiselect("Stat Types", stat_types, default=stat_types)
            max_edge = float(df_e["edge"].dropna().max()) if df_e["edge"].notna().any() else 1.0
            min_edge = st.slider("Min Edge", 0.0, max(max_edge, 0.01), 0.0, step=0.01)

        fdf = df_e.copy()
        if sel_game != "All":
            fdf = fdf[fdf["game_id"] == sel_game]
        if sel_stats:
            fdf = fdf[fdf["stat_type"].isin(sel_stats)]
        fdf = fdf[fdf["edge"].fillna(0) >= min_edge]

        c1, c2, c3, c4 = st.columns(4)
        _metrics([c1, c2, c3, c4], [
            ("Total Edges", str(len(fdf))),
            ("Avg Edge", f"{fdf['edge'].mean():.1%}" if not fdf.empty else "—"),
            ("Avg EV", f"{fdf['ev'].mean():.3f}" if not fdf.empty else "—"),
            ("Avg Kelly", f"{fdf['kelly'].mean():.3f}" if not fdf.empty else "—"),
        ])

        st.divider()

        if not fdf.empty and fdf["edge"].notna().any():
            fig = px.histogram(
                fdf.dropna(subset=["edge"]),
                x="edge", nbins=20,
                title="Edge Distribution",
                labels={"edge": "Edge"},
                color_discrete_sequence=["#3b82f6"],
            )
            fig.update_layout(bargap=0.05)
            st.plotly_chart(fig, use_container_width=True)

        TABLE_COLS = [
            "player_name", "stat_type", "side", "market_line",
            "odds", "model_prob", "market_prob", "edge", "ev", "kelly", "game_id",
        ]
        show = [c for c in TABLE_COLS if c in fdf.columns]

        def _highlight_edge(val):
            if not isinstance(val, float):
                return ""
            return ("background-color:#d1fae5" if val > 0.05
                    else "background-color:#fef9c3" if val > 0.0 else "")

        st.dataframe(
            fdf[show].style.map(_highlight_edge, subset=["edge"]),
            use_container_width=True, hide_index=True,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2: P&L
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_pnl:
    df_r = load_results()

    if df_r.empty:
        _empty(
            "No settled bets yet. Use **ironclad edges --save** then **ironclad settle** to record results."
        )
    else:
        total = len(df_r)
        wins = int((df_r["result"] == "win").sum())
        total_profit = float(df_r["profit_units"].sum())
        total_wagered = float(df_r["units_wagered"].sum())
        win_rate = wins / total if total > 0 else 0.0
        roi = total_profit / total_wagered if total_wagered > 0 else 0.0

        c1, c2, c3, c4 = st.columns(4)
        _metrics([c1, c2, c3, c4], [
            ("Total Bets", str(total)),
            ("Win Rate", f"{win_rate:.1%}"),
            ("Total Profit (units)", f"{total_profit:+.2f}"),
            ("ROI", f"{roi:+.1%}"),
        ])

        st.divider()
        col_l, col_r = st.columns(2)

        with col_l:
            df_sorted = df_r.sort_values("settled_at").copy()
            df_sorted["cum_profit"] = df_sorted["profit_units"].cumsum()
            fig_pnl = px.line(
                df_sorted, x="settled_at", y="cum_profit",
                title="Cumulative Profit (units)",
                labels={"settled_at": "Date", "cum_profit": "Units"},
                color_discrete_sequence=["#10b981"],
            )
            fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_pnl, use_container_width=True)

        with col_r:
            wr_by_stat = (
                df_r.assign(is_win=df_r["result"] == "win")
                .groupby("stat_type")["is_win"]
                .mean()
                .reset_index(name="win_rate")
            )
            fig_wr = px.bar(
                wr_by_stat, x="stat_type", y="win_rate",
                title="Win Rate by Stat Type",
                labels={"stat_type": "Stat", "win_rate": "Win Rate"},
                color_discrete_sequence=["#6366f1"],
            )
            fig_wr.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig_wr, use_container_width=True)

        st.subheader("Bet Results")
        RES_COLS = [
            "result_id", "stat_type", "side", "market_line", "odds",
            "model_prob", "ev", "units_wagered", "result", "profit_units", "settled_at",
        ]
        show = [c for c in RES_COLS if c in df_r.columns]
        st.dataframe(df_r[show], use_container_width=True, hide_index=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3: MODEL HEALTH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_health:
    df_bt = load_backtest()

    if df_bt.empty:
        _empty("No backtest data found. Run: **ironclad backtest** to generate walk-forward predictions.")
    else:
        valid = df_bt.dropna(subset=["home_win_prob", "home_win_actual"])

        if valid.empty:
            _empty("Backtest loaded but no predictions with completed actuals yet.")
        else:
            probs = valid["home_win_prob"].to_numpy(float)
            actuals = valid["home_win_actual"].astype(int).to_numpy()
            overall_brier = m.brier_score(probs, actuals)

            vegas_brier: float | None = None
            if "home_win_prob_vegas" in valid.columns:
                vv = valid.dropna(subset=["home_win_prob_vegas"])
                if not vv.empty:
                    vegas_brier = m.brier_score(
                        vv["home_win_prob_vegas"].to_numpy(float),
                        vv["home_win_actual"].astype(int).to_numpy(),
                    )

            ats_valid = df_bt.dropna(subset=["vegas_spread", "home_margin_actual"])
            ats = None
            if not ats_valid.empty:
                ats = m.ats_record(
                    ats_valid["home_margin_pred"].to_numpy(float),
                    ats_valid["vegas_spread"].to_numpy(float),
                    ats_valid["home_margin_actual"].to_numpy(float),
                )

            c1, c2, c3 = st.columns(3)
            vegas_display = f"{vegas_brier:.4f}" if vegas_brier is not None else "0.2107*"
            ats_str = (
                f"{ats['wins']}-{ats['losses']}-{ats['pushes']} ({ats['pct']:.1%})"
                if ats else "—"
            )
            _metrics([c1, c2, c3], [
                ("Brier Score", f"{overall_brier:.4f}"),
                ("Vegas Baseline", vegas_display),
                ("ATS Record", ats_str),
            ])
            if vegas_brier is None:
                st.caption("* Vegas baseline is the published NFL market average; no Vegas probs in this run.")

            st.divider()
            col_l, col_r = st.columns(2)

            with col_l:
                brier_rows = []
                for season, sg in df_bt.groupby("season"):
                    sv = sg.dropna(subset=["home_win_prob", "home_win_actual"])
                    if len(sv) < 2:
                        continue
                    brier_rows.append({
                        "season": int(season),
                        "brier": m.brier_score(
                            sv["home_win_prob"].to_numpy(float),
                            sv["home_win_actual"].astype(int).to_numpy(),
                        ),
                        "n_games": len(sv),
                    })
                if brier_rows:
                    bdf = pd.DataFrame(brier_rows)
                    fig_b = px.bar(
                        bdf, x="season", y="brier",
                        title="Brier Score by Season",
                        labels={"season": "Season", "brier": "Brier Score"},
                        color_discrete_sequence=["#f59e0b"],
                        text="n_games",
                    )
                    fig_b.update_traces(texttemplate="%{text} games", textposition="outside")
                    st.plotly_chart(fig_b, use_container_width=True)

            with col_r:
                cal = m.calibration_curve(probs, actuals, n_bins=10)
                if not cal.empty:
                    fig_cal = go.Figure()
                    fig_cal.add_trace(go.Scatter(
                        x=cal["predicted_prob"], y=cal["observed_rate"],
                        mode="markers+lines", name="Model",
                        marker=dict(size=cal["count"] / cal["count"].max() * 20 + 5),
                        line=dict(color="#3b82f6"),
                    ))
                    fig_cal.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1], mode="lines",
                        name="Perfect calibration",
                        line=dict(dash="dash", color="gray"),
                    ))
                    fig_cal.update_layout(
                        title="Calibration Curve",
                        xaxis_title="Predicted Probability",
                        yaxis_title="Observed Win Rate",
                        xaxis=dict(range=[0, 1]),
                        yaxis=dict(range=[0, 1]),
                    )
                    st.plotly_chart(fig_cal, use_container_width=True)
