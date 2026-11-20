"""
Video Game Sales Explorer
A dark, polished Streamlit dashboard for exploring the vgsales.csv dataset
(Kaggle: gregorut/videogamesales, scraped from vgchartz.com).

Run locally:   streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Video Game Sales Explorer",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Design tokens
# ----------------------------------------------------------------------------
BG = "#0A0B1E"
SURFACE = "#12142B"
BORDER = "#262A4F"
GRID = "#1E2245"
TEXT = "#E8EAFF"
MUTED = "#8D93BD"

VIOLET = "#8B5CF6"
CYAN = "#22D3EE"
PINK = "#F472B6"
AMBER = "#FBBF24"
GREEN = "#34D399"

REGIONS = {
    "NA_Sales": "North America",
    "EU_Sales": "Europe",
    "JP_Sales": "Japan",
    "Other_Sales": "Other",
}
REGION_COLORS = {
    "North America": CYAN,
    "Europe": VIOLET,
    "Japan": PINK,
    "Other": AMBER,
}
GENRE_COLORS = [
    "#8B5CF6", "#22D3EE", "#F472B6", "#FBBF24", "#34D399", "#60A5FA",
    "#FB923C", "#A78BFA", "#2DD4BF", "#F87171", "#C084FC", "#A3E635",
]
SEQ = ["#12142B", "#3B1F8F", "#8B5CF6", "#22D3EE"]
DIV_SCALE = [[0.0, PINK], [0.5, SURFACE], [1.0, CYAN]]

# Newer Streamlit versions replaced use_container_width with width="stretch".
try:
    _v = tuple(int(p) for p in st.__version__.split(".")[:2])
except ValueError:
    _v = (99, 0)
WIDE = {"width": "stretch"} if _v >= (1, 50) else {"use_container_width": True}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3, .hero-title, .kpi-value, .brand-name { font-family: 'Space Grotesk', sans-serif !important; }

.stApp {
  background:
    radial-gradient(1100px 560px at 85% -10%, rgba(139,92,246,.20), transparent 60%),
    radial-gradient(900px 500px at 0% 0%, rgba(34,211,238,.10), transparent 55%),
    #0A0B1E;
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #0D0F26; border-right: 1px solid #1E2245; }
.brand { display:flex; align-items:center; gap:12px; margin: 4px 0 18px 0; }
.brand-logo { width:44px; height:44px; border-radius:12px; display:flex; align-items:center;
  justify-content:center; font-size:22px;
  background: linear-gradient(135deg, rgba(139,92,246,.35), rgba(34,211,238,.25));
  border:1px solid rgba(139,92,246,.5); }
.brand-name { font-size:20px; font-weight:700; color:#E8EAFF; line-height:1.1; }
.brand-sub { font-size:11px; letter-spacing:1.6px; color:#8D93BD; text-transform:uppercase; }
.side-card { background:#12142B; border:1px solid #262A4F; border-radius:14px; padding:14px 16px; margin-bottom:18px; }
.side-card-title { font-size:11px; letter-spacing:1.6px; color:#8D93BD; text-transform:uppercase; margin-bottom:8px; }
.side-row { display:flex; justify-content:space-between; font-size:13.5px; padding:5px 0;
  border-bottom:1px solid #1E2245; color:#C9CDF2; }
.side-row:last-child { border-bottom:none; }
.side-row b { color:#E8EAFF; font-variant-numeric: tabular-nums; }
.side-section { font-size:11px; letter-spacing:1.6px; color:#8D93BD; text-transform:uppercase; margin:6px 0 2px 0; }
span[data-baseweb="tag"] { background:#3B1F8F !important; color:#EDE9FE !important; border-radius:8px !important; }

/* Hero */
.hero { display:flex; justify-content:space-between; align-items:flex-end; flex-wrap:wrap; gap:12px; margin-bottom:22px; }
.hero-title { font-size:40px; font-weight:700; color:#F5F6FF; letter-spacing:-0.5px; line-height:1.1; margin:0; }
.hero-sub { color:#8D93BD; font-size:15.5px; margin-top:6px; }
.pill { display:inline-flex; align-items:center; gap:8px; padding:7px 14px; border-radius:999px;
  background: rgba(52,211,153,.10); border:1px solid rgba(52,211,153,.35); color:#6EE7B7; font-size:13.5px; }
.pill-dot { width:8px; height:8px; border-radius:50%; background:#34D399; box-shadow:0 0 8px #34D399; }

/* KPI cards */
.kpi { background: linear-gradient(160deg, #15183A 0%, #101228 100%); border:1px solid #262A4F;
  border-radius:20px; padding:20px 22px; position:relative; overflow:hidden; min-height:158px; }
.kpi::before { content:""; position:absolute; left:0; top:0; right:0; height:3px; background: var(--tone); }
.kpi::after { content:""; position:absolute; right:-40px; top:-40px; width:130px; height:130px; border-radius:50%;
  background: var(--tone); opacity:.10; filter: blur(8px); }
.kpi-top { display:flex; justify-content:space-between; align-items:center; }
.kpi-label { font-size:12px; letter-spacing:1.6px; color:#8D93BD; text-transform:uppercase; }
.kpi-icon { width:38px; height:38px; border-radius:11px; display:flex; align-items:center; justify-content:center;
  font-size:18px; background: rgba(255,255,255,.05); border:1px solid #2E3262; }
.kpi-value { font-size:31px; font-weight:700; color:#F5F6FF; margin:14px 0 12px 0; white-space:nowrap;
  overflow:hidden; text-overflow:ellipsis; }
.kpi-badge { display:inline-block; padding:4px 11px; border-radius:999px; font-size:12.5px;
  background: rgba(255,255,255,.06); border:1px solid #2E3262; color: var(--tone); }

/* Chart containers */
div[data-testid="stVerticalBlockBorderWrapper"] { background:#101228; border-color:#262A4F !important; border-radius:18px; }
.ct { font-family:'Space Grotesk',sans-serif; font-size:18px; font-weight:700; color:#F5F6FF; margin-bottom:2px; }
.cs { font-size:13px; color:#8D93BD; margin-bottom:6px; }

/* Insights */
.insight { display:flex; gap:12px; padding:11px 0; border-bottom:1px solid #1E2245; color:#C9CDF2; font-size:14.5px; line-height:1.45; }
.insight:last-child { border-bottom:none; }
.insight .dot { width:9px; height:9px; margin-top:7px; border-radius:50%; flex:0 0 9px; background:var(--tone); box-shadow:0 0 8px var(--tone); }
.insight b { color:#F5F6FF; }

/* Tabs */
button[data-baseweb="tab"] { font-size:15px; font-weight:500; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Data loading & cleaning
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading data…")
def load_data():
    path = Path(__file__).parent / "vgsales.csv"
    raw = pd.read_csv(path, na_values=["N/A"], encoding="latin1")

    info = {
        "rows": int(len(raw)),
        "cols": int(raw.shape[1]),
        "duplicates": int(raw.duplicated().sum()),
        "dup_name_platform": int(raw.duplicated(["Name", "Platform"]).sum()),
        "missing": raw.isna().sum().to_dict(),
    }

    df = raw.drop_duplicates().copy()
    for c in ["Name", "Platform", "Genre", "Publisher"]:
        df[c] = df[c].astype("string").str.strip()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    info["missing_year"] = int(df["Year"].isna().sum())
    info["missing_publisher"] = int(df["Publisher"].isna().sum())
    info["late_years"] = int((df["Year"] > 2016).sum())
    df["Publisher"] = df["Publisher"].fillna("Unknown")
    df = df.astype({"Name": "object", "Platform": "object", "Genre": "object", "Publisher": "object"})
    return df, info


df, info = load_data()


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def style(fig, height=360, legend=True):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXT, size=13),
        margin=dict(l=8, r=8, t=10, b=8),
        colorway=GENRE_COLORS,
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=BORDER, font_color=TEXT),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, title_font_color=MUTED)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, title_font_color=MUTED)
    return fig


def show(fig, key):
    st.plotly_chart(fig, key=key, config={"displayModeBar": False}, **WIDE)


def head(title, sub=""):
    st.markdown(f"<div class='ct'>{title}</div><div class='cs'>{sub}</div>", unsafe_allow_html=True)


def kpi(col, icon, label, value, badge, tone):
    col.markdown(
        f"""<div class="kpi" style="--tone:{tone}">
              <div class="kpi-top"><span class="kpi-label">{label}</span><span class="kpi-icon">{icon}</span></div>
              <div class="kpi-value">{value}</div>
              <span class="kpi-badge">{badge}</span>
            </div>""",
        unsafe_allow_html=True,
    )


def millions(x):
    return f"{x:,.1f}M"


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """<div class="brand"><div class="brand-logo">🎮</div>
           <div><div class="brand-name">Game Sales</div><div class="brand-sub">Data exploration</div></div></div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<div class="side-card"><div class="side-card-title">Dataset</div>
            <div class="side-row"><span>Source</span><b>vgchartz (Kaggle)</b></div>
            <div class="side-row"><span>Format</span><b>CSV (latin1)</b></div>
            <div class="side-row"><span>Rows</span><b>{info['rows']:,}</b></div>
            <div class="side-row"><span>Columns</span><b>{info['cols']}</b></div>
            <div class="side-row"><span>Duplicates removed</span><b>{info['duplicates']}</b></div>
            </div>""",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='side-section'>Filters</div>", unsafe_allow_html=True)
    genres = st.multiselect("Genre", sorted(df["Genre"].unique()), placeholder="All genres")
    platforms = st.multiselect("Platform", sorted(df["Platform"].unique()), placeholder="All platforms")
    publishers = st.multiselect(
        "Publisher", sorted(df["Publisher"].unique()), placeholder="All publishers"
    )
    y_min, y_max = int(df["Year"].min()), int(df["Year"].max())
    years = st.slider("Release year", y_min, y_max, (y_min, y_max))
    include_unknown = st.checkbox("Include games with unknown year", value=True)

# Apply filters (an empty selection means "all")
f = df.copy()
if genres:
    f = f[f["Genre"].isin(genres)]
if platforms:
    f = f[f["Platform"].isin(platforms)]
if publishers:
    f = f[f["Publisher"].isin(publishers)]
year_mask = f["Year"].between(years[0], years[1])
if include_unknown:
    year_mask = year_mask | f["Year"].isna()
f = f[year_mask]

with st.sidebar:
    st.markdown("<div class='side-section'>Export</div>", unsafe_allow_html=True)
    st.download_button(
        "Download filtered CSV",
        data=f.to_csv(index=False).encode("utf-8"),
        file_name="vgsales_filtered.csv",
        mime="text/csv",
        **WIDE,
    )

# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
st.markdown(
    f"""<div class="hero"><div>
          <div class="hero-title">Video Game Sales Explorer</div>
          <div class="hero-sub">Which genres, platforms and publishers shaped four decades of game sales?</div>
        </div>
        <div class="pill"><span class="pill-dot"></span>{len(f):,} of {len(df):,} games</div></div>""",
    unsafe_allow_html=True,
)

if f.empty:
    st.warning("No games match these filters. Widen the selection in the sidebar.")
    st.stop()

# ----------------------------------------------------------------------------
# Aggregates shared by several charts
# ----------------------------------------------------------------------------
total_sales = f["Global_Sales"].sum()
genre_sales = f.groupby("Genre")["Global_Sales"].sum().sort_values(ascending=False)
platform_sales = f.groupby("Platform")["Global_Sales"].sum().sort_values(ascending=False)
known_pub = f[f["Publisher"] != "Unknown"]
pub_sales = known_pub.groupby("Publisher")["Global_Sales"].sum().sort_values(ascending=False)
region_totals = f[list(REGIONS)].sum().rename(index=REGIONS)

yearly = (
    f.dropna(subset=["Year"])
    .assign(Year=lambda d: d["Year"].astype(int))
    .groupby("Year")
    .agg(Games=("Name", "count"), Sales=("Global_Sales", "sum"))
    .reset_index()
)

# ----------------------------------------------------------------------------
# KPI row
# ----------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
kpi(c1, "💰", "Total global sales", millions(total_sales),
    f"{total_sales / len(f):.2f}M avg per game", CYAN)
kpi(c2, "🕹️", "Games", f"{len(f):,}",
    f"{f['Platform'].nunique()} platforms · {f['Genre'].nunique()} genres", VIOLET)
kpi(c3, "🏆", "Top genre", genre_sales.index[0],
    f"{genre_sales.iloc[0] / total_sales:.1%} of sales", PINK)
if pub_sales.empty:
    kpi(c4, "🏢", "Top publisher", "n/a", "no publisher data", AMBER)
else:
    kpi(c4, "🏢", "Top publisher", pub_sales.index[0],
        f"{pub_sales.iloc[0] / total_sales:.1%} of sales", AMBER)

st.write("")

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_overview, tab_regions, tab_dist, tab_data = st.tabs(
    ["Overview", "Regions", "Distributions & outliers", "Data & cleaning"]
)

# ---------------------------- Overview -------------------------------------
with tab_overview:
    a, b = st.columns(2)
    with a, st.container(border=True):
        head("Sales by genre", "Total global sales in millions of copies")
        g = genre_sales.reset_index()
        fig = px.bar(g, x="Global_Sales", y="Genre", orientation="h",
                     color="Global_Sales", color_continuous_scale=SEQ, text_auto=".0f",
                     labels={"Global_Sales": "Global sales (M)"})
        fig.update_layout(coloraxis_showscale=False, yaxis=dict(categoryorder="total ascending"))
        fig.update_traces(textfont_color=TEXT, textposition="outside", cliponaxis=False)
        show(style(fig, 400, legend=False), "genre_bar")

    with b, st.container(border=True):
        head("Sales and releases over time", "Bars: sales (M) · Line: number of games released")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=yearly["Year"], y=yearly["Sales"], name="Sales (M)",
                             marker_color=VIOLET, opacity=0.85), secondary_y=False)
        fig.add_trace(go.Scatter(x=yearly["Year"], y=yearly["Games"], name="Games released",
                                 mode="lines+markers", line=dict(color=CYAN, width=2.5),
                                 marker=dict(size=5)), secondary_y=True)
        fig.update_yaxes(title_text="Sales (M)", secondary_y=False)
        fig.update_yaxes(title_text="Games", secondary_y=True, showgrid=False)
        show(style(fig, 400), "year_combo")

    a, b = st.columns(2)
    with a, st.container(border=True):
        head("Top 10 platforms", "Which consoles sold the most")
        p = platform_sales.head(10).reset_index()
        fig = px.bar(p, x="Platform", y="Global_Sales", color="Global_Sales",
                     color_continuous_scale=SEQ, text_auto=".0f",
                     labels={"Global_Sales": "Global sales (M)"})
        fig.update_layout(coloraxis_showscale=False)
        fig.update_traces(textfont_color=TEXT, textposition="outside", cliponaxis=False)
        show(style(fig, 360, legend=False), "platform_bar")

    with b, st.container(border=True):
        head("Top 10 publishers", "Unknown publishers excluded")
        if pub_sales.empty:
            st.info("No publisher data for this selection.")
        else:
            p = pub_sales.head(10).reset_index()
            fig = px.bar(p, x="Global_Sales", y="Publisher", orientation="h",
                         color="Global_Sales", color_continuous_scale=SEQ, text_auto=".0f",
                         labels={"Global_Sales": "Global sales (M)"})
            fig.update_layout(coloraxis_showscale=False, yaxis=dict(categoryorder="total ascending"))
            fig.update_traces(textfont_color=TEXT, textposition="outside", cliponaxis=False)
            show(style(fig, 360, legend=False), "publisher_bar")

    a, b = st.columns([3, 2])
    with a, st.container(border=True):
        head("Best-selling games", "Top 10 titles by global sales")
        t = f.nlargest(10, "Global_Sales").copy()
        t["Label"] = t["Name"] + " (" + t["Platform"] + ")"
        fig = px.bar(t, x="Global_Sales", y="Label", orientation="h", color="Genre",
                     labels={"Global_Sales": "Global sales (M)", "Label": ""},
                     color_discrete_sequence=GENRE_COLORS)
        fig.update_layout(yaxis=dict(categoryorder="total ascending"))
        show(style(fig, 400), "top_games")

    with b, st.container(border=True):
        head("Key insights", "Generated from the current filters")
        s = f["Global_Sales"].sort_values(ascending=False)
        k = max(1, int(len(s) * 0.10))
        top_share = s.iloc[:k].sum() / s.sum() if s.sum() else 0
        lead_region = region_totals.idxmax()
        lead_share = region_totals.max() / region_totals.sum()
        items = [
            (CYAN, f"<b>{genre_sales.index[0]}</b> leads with {genre_sales.iloc[0] / total_sales:.0%} of all sales."),
            (VIOLET, f"The top 10% of titles earn <b>{top_share:.0%}</b> of total sales, a heavy-tailed market."),
            (PINK, f"<b>{lead_region}</b> is the biggest market at {lead_share:.0%} of sales."),
        ]
        if not yearly.empty:
            peak = yearly.loc[yearly["Sales"].idxmax()]
            items.append((AMBER, f"Sales peaked in <b>{int(peak['Year'])}</b> with {peak['Sales']:.0f}M copies."))
        items.append((GREEN, f"<b>{platform_sales.index[0]}</b> is the best-selling platform "
                             f"({platform_sales.iloc[0]:,.0f}M)."))
        html = "".join(
            f"<div class='insight' style='--tone:{c}'><span class='dot'></span><span>{t}</span></div>"
            for c, t in items
        )
        st.markdown(html, unsafe_allow_html=True)

# ---------------------------- Regions --------------------------------------
with tab_regions:
    a, b = st.columns([2, 3])
    with a, st.container(border=True):
        head("Share of sales by region", "Where the money comes from")
        r = region_totals.reset_index()
        r.columns = ["Region", "Sales"]
        fig = px.pie(r, names="Region", values="Sales", hole=0.62, color="Region",
                     color_discrete_map=REGION_COLORS)
        fig.update_traces(textinfo="percent", textfont_color="#0A0B1E",
                          marker=dict(line=dict(color=BG, width=3)))
        show(style(fig, 380), "region_donut")

    with b, st.container(border=True):
        head("Regional sales by genre", "Toggle between absolute sales and each genre's regional mix")
        as_pct = st.toggle("Show as % share", value=False, key="pct_toggle")
        gr = f.groupby("Genre")[list(REGIONS)].sum().rename(columns=REGIONS)
        gr = gr.loc[gr.sum(axis=1).sort_values(ascending=False).index]
        long = gr.reset_index().melt(id_vars="Genre", var_name="Region", value_name="Sales")
        fig = px.bar(long, x="Genre", y="Sales", color="Region", color_discrete_map=REGION_COLORS,
                     barmode="stack",
                     labels={"Sales": "Share (%)" if as_pct else "Sales (M)"})
        if as_pct:
            fig.update_layout(barnorm="percent")
        show(style(fig, 330), "region_genre")

    a, b = st.columns(2)
    with a, st.container(border=True):
        head("Regional trends over time", "Stacked sales by release year")
        ry = (f.dropna(subset=["Year"]).assign(Year=lambda d: d["Year"].astype(int))
              .groupby("Year")[list(REGIONS)].sum().rename(columns=REGIONS).reset_index()
              .melt(id_vars="Year", var_name="Region", value_name="Sales"))
        fig = px.area(ry, x="Year", y="Sales", color="Region", color_discrete_map=REGION_COLORS,
                      labels={"Sales": "Sales (M)"})
        show(style(fig, 360), "region_area")

    with b, st.container(border=True):
        head("Genre taste by region", "Each row: % of that genre's sales coming from each region")
        mix = gr.div(gr.sum(axis=1), axis=0) * 100
        fig = px.imshow(mix.round(0), text_auto=".0f", aspect="auto",
                        color_continuous_scale=SEQ, labels=dict(color="% of sales"))
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(side="top")
        show(style(fig, 360, legend=False), "taste_heat")

# ---------------------------- Distributions --------------------------------
with tab_dist:
    a, b = st.columns(2)
    with a, st.container(border=True):
        head("Distribution of game sales", "log₁₀ of global sales (M). Most games sell little, a few sell a lot")
        pos = f[f["Global_Sales"] > 0].assign(log_sales=lambda d: np.log10(d["Global_Sales"]))
        fig = px.histogram(pos, x="log_sales", nbins=50, color_discrete_sequence=[VIOLET],
                           labels={"log_sales": "log₁₀(global sales, millions)"})
        fig.update_traces(marker_line_color=BG, marker_line_width=1)
        fig.update_yaxes(title_text="Number of games")
        show(style(fig, 360, legend=False), "hist")
        st.caption(
            f"Skewness: **{f['Global_Sales'].skew():.1f}** · median **{f['Global_Sales'].median():.2f}M** "
            f"vs mean **{f['Global_Sales'].mean():.2f}M**. The mean is pulled up by blockbusters."
        )

    with b, st.container(border=True):
        head("Sales spread by genre", "Box plot on a log scale; dots are outlier titles")
        fig = px.box(f[f["Global_Sales"] > 0], x="Genre", y="Global_Sales", color="Genre",
                     points="outliers", log_y=True, color_discrete_sequence=GENRE_COLORS,
                     labels={"Global_Sales": "Global sales (M)"})
        show(style(fig, 360, legend=False), "box")

    a, b = st.columns(2)
    with a, st.container(border=True):
        head("North America vs Europe", "Each dot is a game (log scales, zero-sales excluded)")
        sc = f[(f["NA_Sales"] > 0) & (f["EU_Sales"] > 0)]
        fig = px.scatter(sc, x="NA_Sales", y="EU_Sales", color="Genre", log_x=True, log_y=True,
                         hover_name="Name", hover_data=["Platform", "Year"], opacity=0.7,
                         color_discrete_sequence=GENRE_COLORS,
                         labels={"NA_Sales": "NA sales (M)", "EU_Sales": "EU sales (M)"})
        show(style(fig, 380), "scatter")

    with b, st.container(border=True):
        head("Correlation matrix", "How strongly the numeric columns move together")
        corr = f[["Year", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]].corr()
        fig = px.imshow(corr.round(2), text_auto=".2f", aspect="auto", zmin=-1, zmax=1,
                        color_continuous_scale=DIV_SCALE)
        fig.update_layout(coloraxis_colorbar=dict(title="r", thickness=12))
        show(style(fig, 380, legend=False), "corr")

# ---------------------------- Data & cleaning ------------------------------
with tab_data:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows (raw)", f"{info['rows']:,}")
    m2.metric("Columns", info["cols"])
    m3.metric("Missing cells (raw)", f"{sum(info['missing'].values()):,}")
    m4.metric("Exact duplicates", info["duplicates"])

    a, b = st.columns(2)
    with a, st.container(border=True):
        head("Cleaning log", "What was found and what was done")
        log = pd.DataFrame({
            "Issue": [
                "Missing Year",
                "Missing Publisher",
                "Exact duplicate rows",
                "Same Name + Platform repeated",
                "Release year after 2016",
            ],
            "Count": [
                info["missing_year"], info["missing_publisher"], info["duplicates"],
                info["dup_name_platform"], info["late_years"],
            ],
            "Action": [
                "Kept; excluded from time charts, toggle in sidebar",
                "Filled with 'Unknown'; excluded from publisher rankings",
                "Dropped",
                "Kept; flagged for review (may be different releases)",
                "Kept; the source mostly covers up to 2016",
            ],
        })
        st.dataframe(log, hide_index=True, **WIDE)

    with b, st.container(border=True):
        head("Missing values by column", "Raw file, before cleaning")
        miss = pd.Series(info["missing"]).reset_index()
        miss.columns = ["Column", "Missing"]
        fig = px.bar(miss, x="Missing", y="Column", orientation="h", color_discrete_sequence=[PINK],
                     text_auto=True)
        fig.update_layout(yaxis=dict(autorange="reversed"))
        fig.update_traces(textfont_color=TEXT, cliponaxis=False)
        show(style(fig, 280, legend=False), "missing")

    with st.container(border=True):
        head("Summary statistics", "Numeric columns for the current selection")
        st.dataframe(f.describe().T.round(2), **WIDE)

    with st.container(border=True):
        head("Browse the data", "Click a column header to sort")
        st.dataframe(f.reset_index(drop=True), hide_index=True, height=380, **WIDE)

# ----------------------------------------------------------------------------
# Methodology expander
# ----------------------------------------------------------------------------
with st.expander("Data summary & methodology"):
    t1, t2, t3 = st.tabs(["Summary stats", "Key findings", "Data ethics"])
    with t1:
        st.markdown("**Metric distribution overview**")
        st.dataframe(f[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]]
                     .describe().T.round(2), **WIDE)
    with t2:
        top_game = f.sort_values("Global_Sales", ascending=False).iloc[0]
        reg_tot = f[list(REGIONS)].sum().rename(REGIONS)
        top_reg = reg_tot.idxmax()
        best_yr = yearly.sort_values("Sales", ascending=False).iloc[0] if len(yearly) else None
        k1, k2 = st.columns(2)
        with k1:
            st.markdown(
                f"- **Top genre:** {genre_sales.index[0]} with {genre_sales.iloc[0]:,.0f}M copies "
                f"({genre_sales.iloc[0] / total_sales:.1%} of sales).\n"
                f"- **Best-selling game:** {top_game['Name']} ({top_game['Platform']}) at "
                f"{top_game['Global_Sales']:.1f}M copies.\n"
                f"- **Biggest region:** {top_reg} with {reg_tot[top_reg] / total_sales:.1%} of sales.")
        with k2:
            top10 = f.nlargest(max(1, len(f) // 10), "Global_Sales")["Global_Sales"].sum()
            st.markdown(
                f"- **Concentration:** the top 10% of titles earn {top10 / total_sales:.0%} of all sales.\n"
                + (f"- **Peak year:** {int(best_yr['Year'])} with {best_yr['Sales']:,.0f}M copies sold.\n"
                   if best_yr is not None else "")
                + f"- **Correlation:** NA and EU sales have r = "
                  f"{f['NA_Sales'].corr(f['EU_Sales']):.2f}, so hits tend to travel across markets.")
    with t3:
        e1, e2 = st.columns(2)
        with e1:
            st.markdown("##### Color accessibility")
            st.markdown("- Region colors are distinct in both hue and brightness, so they stay readable "
                        "for people with color-vision deficiencies.\n"
                        "- Sequential scales move from dark to bright, so they also work in grayscale.")
        with e2:
            st.markdown("##### Zero-baseline integrity")
            st.markdown("- Bar charts start at zero, because bar length encodes the value.\n"
                        "- Heavily skewed data is shown on a log scale only where labeled, "
                        "and zero-sales titles are excluded there.")

st.download_button("Download filtered data (CSV)", f.to_csv(index=False).encode("utf-8"),
                   file_name="vgsales_filtered.csv", mime="text/csv")

st.caption("Data: Kaggle · Video Game Sales (gregorut), scraped from vgchartz.com. Sales are in millions of copies.")
