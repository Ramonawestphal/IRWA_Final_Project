"""
EDA for IRWA Part 1 – Fashion Products Dataset
Generates summary statistics, token frequencies, plots, and a word cloud.
Outputs are stored in project_progress/part_1/
"""

import os
import json
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ---------- Paths ----------
INP = "data/processed/products_clean.parquet"
OUTDIR = "project_progress/part_1"
os.makedirs(OUTDIR, exist_ok=True)

# ---------- Load data ----------
df = pd.read_parquet(INP)

# ---------- Summary statistics ----------
summary = {
    "docs": int(len(df)),
    "unique_brands": int(df["brand"].nunique()),
    "unique_categories": int(df["category"].nunique()),
    "avg_price": float(df["selling_price"].dropna().mean()),
    "avg_discount_frac": float(df["discount_frac"].dropna().mean()),
    "avg_rating": float(df["average_rating"].dropna().mean()),
    "out_of_stock_pct": float(100 * df["out_of_stock"].mean()),
}

print(summary)
with open(os.path.join(OUTDIR, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

# ---------- Top brands ----------
top_brands = (
    df["brand"]
    .value_counts()
    .head(20)
    .rename_axis("brand")
    .reset_index(name="count")
)
top_brands.to_csv(os.path.join(OUTDIR, "top_brands.csv"), index=False)
print(top_brands.head(10))

# ---------- Token frequencies ----------
v_title = Counter(t for ts in df["title_tokens"] for t in ts)
v_desc = Counter(t for ts in df["desc_tokens"] for t in ts)

pd.DataFrame(v_title.most_common(50), columns=["term", "freq"]).to_csv(
    os.path.join(OUTDIR, "top_terms_title.csv"), index=False
)
pd.DataFrame(v_desc.most_common(50), columns=["term", "freq"]).to_csv(
    os.path.join(OUTDIR, "top_terms_desc.csv"), index=False
)

# ---------- Distributions ----------
# Selling price
ax = df["selling_price"].dropna().plot(
    kind="hist", bins=40, title="Selling Price Distribution"
)
ax.set_xlabel("Price")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "fig_price_hist.png"))
plt.clf()

# Discount
ax = df["discount_frac"].dropna().plot(
    kind="hist", bins=40, title="Discount Fraction Distribution"
)
ax.set_xlabel("Discount (0–1)")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "fig_discount_hist.png"))
plt.clf()

# Rating
ax = df["average_rating"].dropna().plot(
    kind="hist", bins=20, title="Average Rating Distribution"
)
ax.set_xlabel("Rating")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "fig_rating_hist.png"))
plt.clf()

# ---------- Description and title length ----------
df["desc_len"] = df["desc_tokens"].apply(len)
df["title_len"] = df["title_tokens"].apply(len)

df["desc_len"].plot(kind="hist", bins=40, title="Description Length Distribution")
plt.xlabel("Number of tokens")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "fig_desc_length.png"))
plt.clf()

# ---------- Vocabulary statistics ----------
vocab = Counter(t for ts in df["title_tokens"] for t in ts)
print("Vocabulary size:", len(vocab))
print("Most frequent tokens:", vocab.most_common(20))

# ---------- Top brands and categories ----------
df["brand"].value_counts().head(5).plot(kind="barh", title="Top 5 Brands")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "fig_top_brands.png"))
plt.clf()

df["category"].value_counts().head(4).plot(kind="barh", title="Categories")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "fig_top_categories.png"))
plt.clf()

# ---------- Word cloud ----------
wc = WordCloud(width=1000, height=600).generate(
    " ".join(t for ts in df["title_tokens"] for t in ts)
)
plt.imshow(wc)
plt.axis("off")
plt.title("Most Common Tokens in Product Titles")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "fig_wordcloud_titles.png"))
plt.clf()

print("EDA complete. Outputs saved to:", OUTDIR)
