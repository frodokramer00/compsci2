"""
app.py – Flask server for

1. Netflix actor-collaboration network
   • Colour nodes by Country or by Genre
   • Interactive Plotly graph

2. Co-author community network
   • Reads final_abstracts.csv, final_authors.csv, final_papers.csv
   • Detects communities using label-propagation

3. NLP page for WordCloud of Netflix descriptions by genre
"""

from flask import Flask, render_template, request, send_from_directory
from markupsafe import Markup
import os
from typing import Optional
import pandas as pd
import networkx as nx
from itertools import combinations
from collections import Counter
import matplotlib.cm as cm
import plotly.graph_objects as go
import plotly.io as pio
import ast
from wordcloud import WordCloud, STOPWORDS  # ✅ fixed STOPWORDS import
import matplotlib.pyplot as plt
import io, base64

app = Flask(__name__)

@app.route("/assets/<path:filename>")
def assets(filename: str):
    return send_from_directory(os.path.join(app.root_path, "assets"), filename)

# ─── Netflix actor data ────────────────────────────────────────────────
df_netflix = pd.read_csv("netflix_titles.csv")
df_netflix = df_netflix.dropna(subset=["cast"])


def _nx_to_plotly(G, *, node_colours, node_sizes, node_text, title):
    pos = nx.spring_layout(G, k=0.3, seed=42)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=0.4, color="#888"), hoverinfo="none")
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_trace = go.Scatter(x=node_x, y=node_y, mode="markers", hoverinfo="text", text=node_text,
                            marker=dict(size=node_sizes, color=node_colours, line=dict(width=0)))
    fig = go.Figure(
    data=[edge_trace, node_trace],
    layout=go.Layout(
        title=title,
        title_x=0.5,
        hovermode="closest",
        showlegend=False,
        margin=dict(l=0, r=0, t=60, b=0),  # slightly more top margin
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=1000
    )
)
    fig.update_layout(template="plotly_white", autosize=True)
    return Markup(
    pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs="cdn",
        config={"responsive": True}
    )
)
    #return Markup(pio.to_html(fig, full_html=False, include_plotlyjs="cdn", config={"responsive": True}))

def _actor_collab_graph(top_n=450):
    df_actor = df_netflix.dropna(subset=["cast"]).copy()
    df_actor["cast"] = df_actor["cast"].str.split(", ")
    edges = []
    for cast_list in df_actor["cast"]:
        if isinstance(cast_list, list) and len(cast_list) > 1:
            edges += combinations(sorted(set(cast_list)), 2)
    collab_df = pd.DataFrame(Counter(edges).items(), columns=["pair", "weight"])
    collab_df[["actor_1", "actor_2"]] = pd.DataFrame(collab_df["pair"].tolist(), index=collab_df.index)
    G = nx.Graph()
    for _, row in collab_df.iterrows():
        G.add_edge(row.actor_1, row.actor_2, weight=row.weight)
    top_nodes = sorted(G.degree, key=lambda t: t[1], reverse=True)[:top_n]
    return G.subgraph([n for n, _ in top_nodes]).copy()


@app.route("/data")
def data_info():
    preview_rows = df_netflix[["title", "type", "cast", "country", "release_year", "listed_in", "description"]].dropna(subset=["cast"]).head(10).to_dict(orient="records")
    return render_template("data.html", preview_rows=preview_rows)

@app.route("/download/netflix_titles.csv")
def download_netflix_csv():
    return send_from_directory(
        directory=os.path.join(app.root_path, "archive_flask_app"),
        filename="netflix_titles.csv",  # ✅ correct
        as_attachment=True
    )


def build_country_graph_html(top_n=450):
    G = _actor_collab_graph(top_n)
    df_map = df_netflix.dropna(subset=["cast", "country"]).copy()
    df_map["cast"] = df_map["cast"].str.split(", ")
    df_map["country"] = df_map["country"].str.split(", ")
    df_map = df_map.explode("cast").explode("country")
    actor_to_country = df_map.groupby("cast")["country"].agg(lambda x: x.value_counts().idxmax()).to_dict()
    palette = cm.get_cmap("tab20").colors
    cats = sorted({actor_to_country.get(a, "Unknown") for a in G.nodes()})
    colour_for = {c: palette[i % len(palette)] for i, c in enumerate(cats)}
    rgba = lambda c: f"rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},0.9)"
    node_colours = [rgba(colour_for[actor_to_country.get(a, 'Unknown')]) for a in G.nodes()]
    node_sizes = [3 + sum(d["weight"] for _, _, d in G.edges(n, data=True)) for n in G.nodes()]
    node_text = [f"<b>{n}</b><br>Country: {actor_to_country.get(n,'Unknown')}<br>Degree: {G.degree[n]}" for n in G.nodes()]
    return _nx_to_plotly(G, node_colours=node_colours, node_sizes=node_sizes, node_text=node_text,
                         title=f"Netflix Actor Collaboration – coloured by Country (top {top_n})")

def build_genre_graph_html(top_n=450):
    G = _actor_collab_graph(top_n)
    df_map = df_netflix.dropna(subset=["cast", "listed_in"]).copy()
    df_map["cast"] = df_map["cast"].str.split(", ")
    df_map["listed_in"] = df_map["listed_in"].str.split(", ")
    df_map = df_map.explode("cast").explode("listed_in")
    df_map["listed_in"] = df_map["listed_in"].str.strip()
    actor_to_genre = df_map.groupby("cast")["listed_in"].agg(lambda x: x.value_counts().idxmax()).to_dict()
    cats = sorted({actor_to_genre.get(a, "Unknown") for a in G.nodes()})
    cmap = cm.get_cmap("nipy_spectral", len(cats))
    colour_for = {g: cmap(i / len(cats)) for i, g in enumerate(cats)}
    rgba = lambda c: f"rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},0.9)"
    node_colours = [rgba(colour_for[actor_to_genre.get(a, 'Unknown')]) for a in G.nodes()]
    node_sizes = [3 + sum(d["weight"] for _, _, d in G.edges(n, data=True)) for n in G.nodes()]
    node_text = [f"<b>{n}</b><br>Genre: {actor_to_genre.get(n,'Unknown')}<br>Degree: {G.degree[n]}" for n in G.nodes()]
    return _nx_to_plotly(G, node_colours=node_colours, node_sizes=node_sizes, node_text=node_text,
                         title=f"Netflix Actor Collaboration – coloured by Genre (top {top_n})")

# ─── Word cloud per genre ──────────────────────────────────────────────────
def _description_wordcloud(genre: str) -> Markup:
    df_sub = df_netflix[df_netflix["listed_in"].str.contains(genre, na=False)]
    if df_sub.empty:
        return Markup(f"<div class='alert alert-warning'>No titles found for genre <b>{genre}</b>.</div>")
    text = " ".join(df_sub["description"].dropna().tolist())
    wc = WordCloud(
        width=1600,
        height=800,
        background_color="white",
        colormap="nipy_spectral",
        max_words=200,
        stopwords=STOPWORDS.union({"film", "movie", "series", "story", "new", "one", "two", "year"})
    ).generate(text)
    buf = io.BytesIO()
    plt.figure(figsize=(16, 8))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return Markup(f"<img class='img-fluid shadow rounded' src='data:image/png;base64,{b64}' alt='Word cloud'>")

@app.route("/nlp", methods=["GET", "POST"])
def nlp_page():
    genres = (
        df_netflix["listed_in"]
        .str.split(",")
        .explode()
        .str.strip()
        .dropna()
        .sort_values()
        .unique()
        .tolist()
    )
    chosen = None
    cloud_html = None
    if request.method == "POST":
        chosen = request.form.get("genre")
        cloud_html = _description_wordcloud(chosen)
    return render_template("nlp.html", genres=genres, chosen=chosen, cloud_html=cloud_html)

# ─── Co-author community network ───────────────────────────────────────────
community_cache: dict = {}

def _load_community_data():
    if community_cache:
        return community_cache
    required = ["final_abstracts.csv", "final_authors.csv", "final_papers.csv"]
    if not all(os.path.exists(f) for f in required):
        community_cache["error"] = "Required CSV files (final_*.csv) not found."
        return community_cache
    abstracts = pd.read_csv("final_abstracts.csv")
    papers = pd.read_csv("final_papers.csv")
    merged = pd.merge(papers, abstracts, on="id", how="inner")
    merged["author_ids"] = merged["author_ids"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    G = nx.Graph()
    for _, row in merged.iterrows():
        for i in range(len(row["author_ids"])):
            for j in range(i + 1, len(row["author_ids"])):
                G.add_edge(row["author_ids"][i], row["author_ids"][j])
    comms = list(nx.community.label_propagation_communities(G))
    a2c = {a: cid for cid, comm in enumerate(comms) for a in comm}
    community_cache.update({"graph": G, "a2c": a2c, "comms": comms})
    return community_cache

def build_coauthor_graph_html(top_n: int = 500) -> Markup:
    data = _load_community_data()
    if "error" in data:
        return Markup(f"<div class='alert alert-danger'>{data['error']}</div>")
    G = data["graph"].copy()
    a2c = data["a2c"]
    G = G.subgraph([n for n, _ in sorted(G.degree, key=lambda t: t[1], reverse=True)[:top_n]])
    cats = sorted({a2c.get(a, -1) for a in G.nodes()})
    cmap = cm.get_cmap("nipy_spectral", len(cats))
    colour_for = {cid: cmap(i / len(cats)) for i, cid in enumerate(cats)}
    rgba = lambda c: f"rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},0.9)"
    node_colours = [rgba(colour_for[a2c.get(a, -1)]) for a in G.nodes()]
    node_sizes = [3 + sum(d["weight"] for _, _, d in G.edges(n, data=True)) for n in G.nodes()]
    node_text = [f"<b>Author {n}</b><br>Community: {a2c.get(n)}<br>Degree: {G.degree[n]}" for n in G.nodes()]
    return _nx_to_plotly(G, node_colours, node_sizes, node_text,
                         f"Co-author Network – coloured by Community (top {top_n})")

# ─── Routes ────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    graph_type = "country"
    top_n = 450
    graph_html: Optional[Markup] = None
    if request.method == "POST":
        graph_type = request.form.get("graph_type", "country")
        top_n = int(request.form.get("top_n", 450))
        graph_html = (build_genre_graph_html(top_n) if graph_type == "genre" else build_country_graph_html(top_n))
    return render_template("index.html", graph_type=graph_type, top_n=top_n, graph_html=graph_html)

@app.route("/community", methods=["GET", "POST"])
def community():
    top_n = int(request.form.get("top_n", 500)) if request.method == "POST" else 500
    graph_html = build_coauthor_graph_html(top_n)
    return render_template("community.html", top_n=top_n, graph_html=graph_html)

# ─── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
