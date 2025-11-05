from importlib import import_module
from typing import Any, cast


st = cast(Any, import_module("streamlit"))
pd = cast(Any, import_module("pandas"))
px = cast(Any, import_module("plotly.express"))
go = cast(Any, import_module("plotly.graph_objects"))

st.set_page_config(layout="wide")

st.title("World Map Visualization with Country-Specific Data")

# Example data (replace with your actual data)
# Ensure 'Country Code' column uses ISO-3 country codes for accurate mapping
data = {
    "Country": [
        "United States",
        "Canada",
        "Mexico",
        "Brazil",
        "Germany",
        "France",
        "India",
        "China",
        "Australia",
        "Japan",
    ],
    "Country Code": ["USA", "CAN", "MEX", "BRA", "DEU", "FRA", "IND", "CHN", "AUS", "JPN"],
    "Data Value": [100, 80, 50, 70, 90, 60, 120, 150, 75, 110],
    "Population (M)": [331, 38, 128, 214, 83, 67, 1400, 1440, 25, 125],
}
df = pd.DataFrame(data)

# Map color gradient
color_scale = ["#0f172a", "#1d4ed8", "#38bdf8", "#a5f3fc"]

map_fig = px.choropleth(
    df,
    locations="Country Code",
    locationmode="ISO-3",
    color="Data Value",
    hover_name="Country",
    projection="natural earth",
    color_continuous_scale=color_scale,
)

map_fig.update_traces(
    customdata=df[["Population (M)"]],
    hovertemplate=(
        "<b>%{hovertext}</b><br>" "Value: %{z}<br>Population: %{customdata[0]}M<extra></extra>"
    ),
)

map_fig.update_geos(
    showframe=False,
    showcountries=True,
    countrycolor="rgba(255,255,255,0.25)",
    showcoastlines=True,
    coastlinecolor="rgba(56,189,248,0.3)",
    projection_scale=0.95,
    bgcolor="rgba(0,0,0,0)",
)

map_fig.update_layout(
    title={"text": "Global Data Distribution", "x": 0.01, "xanchor": "left", "font": {"size": 24, "color": "#e2e8f0"}},
    paper_bgcolor="#0f172a",
    plot_bgcolor="#0f172a",
    font=dict(color="#cbd5f5"),
    margin=dict(l=0, r=0, t=60, b=0),
    height=700,
    coloraxis_colorbar=dict(
        title="Value",
        tickcolor="#cbd5f5",
        tickfont=dict(color="#cbd5f5"),
        outlinecolor="rgba(0,0,0,0)",
    ),
)

st.plotly_chart(map_fig, width="stretch")

# Line chart for population trends
line_fig = go.Figure()
line_fig.add_trace(
    go.Scatter(
        x=df["Country"],
        y=df["Population (M)"],
        mode="lines+markers",
        line=dict(color="#38bdf8", width=3),
        marker=dict(size=8, color="#a5f3fc", line=dict(color="#1d4ed8", width=1)),
        hovertemplate="<b>%{x}</b><br>Population: %{y}M<extra></extra>",
    )
)

line_fig.update_layout(
    title={"text": "Population by Country", "x": 0.01, "xanchor": "left", "font": {"size": 22, "color": "#e2e8f0"}},
    xaxis=dict(title="Country", tickangle=-30, showgrid=False, color="#cbd5f5"),
    yaxis=dict(title="Population (Millions)", gridcolor="rgba(148,163,184,0.25)", color="#cbd5f5"),
    paper_bgcolor="#0f172a",
    plot_bgcolor="#0f172a",
    font=dict(color="#cbd5f5"),
    margin=dict(l=40, r=20, t=60, b=40),
)

st.plotly_chart(line_fig, width="stretch")