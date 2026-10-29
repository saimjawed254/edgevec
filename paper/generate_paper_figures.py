"""
EdgeVec Publication Figure Generator
Generates publication-quality figures for IEEE Transactions on Knowledge and Data Engineering / IEEE Computer Society.
All figures are saved in both PDF and 300 DPI PNG in paper/figures/.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# Set up IEEE publication styling
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Palatino", "DejaVu Serif"],
    "font.size": 9.5,
    "axes.labelsize": 10,
    "axes.titlesize": 10.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "figure.titlesize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# =============================================================================
# FIGURE 1: SYSTEM ARCHITECTURE (Centralized vs. EdgeVec Disaggregated)
# =============================================================================
def generate_fig1_architecture():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.85))

    # Panel A: Centralized ANNS Architecture
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis("off")
    ax1.set_title("(a) Conventional Centralized In-Memory ANNS", fontsize=9.5, fontweight="bold", pad=8)

    # Client
    ax1.add_patch(patches.FancyBboxPatch((0.5, 6.8), 3.2, 2.4, boxstyle="round,pad=0.2", facecolor="#f1f5f9", edgecolor="#64748b", lw=1.2))
    ax1.text(2.1, 8.3, "Thin Client Browser", ha="center", va="center", fontweight="bold", fontsize=8.5)
    ax1.text(2.1, 7.4, "Plaintext Query: 'breach'\n(Zero Computation)", ha="center", va="center", fontsize=7.5, color="#334155")

    # Network Arrow
    ax1.annotate("", xy=(5.2, 8.0), xytext=(3.7, 8.0), arrowprops=dict(arrowstyle="->", color="#ef4444", lw=1.5))
    ax1.text(4.45, 8.4, "Raw Query\n(Privacy Leak)", ha="center", va="bottom", fontsize=7, color="#dc2626", fontweight="bold")

    # Central Server Box
    ax1.add_patch(patches.FancyBboxPatch((5.3, 1.2), 4.3, 8.0, boxstyle="round,pad=0.2", facecolor="#fef2f2", edgecolor="#ef4444", lw=1.5))
    ax1.text(7.45, 8.6, "Monolithic Server Node", ha="center", va="center", fontweight="bold", fontsize=9, color="#991b1b")

    # Sub-blocks inside server
    ax1.add_patch(patches.Rectangle((5.6, 6.4), 3.7, 1.5, facecolor="#fee2e2", edgecolor="#f87171", lw=1))
    ax1.text(7.45, 7.15, "CPU Inference & Distance Engine\n(Intel Xeon / GPU Cluster)", ha="center", va="center", fontsize=7.5)

    ax1.add_patch(patches.Rectangle((5.6, 4.2), 3.7, 1.6, facecolor="#fecaca", edgecolor="#ef4444", lw=1.2, hatch="//"))
    ax1.text(7.45, 5.0, "MEMORY BUS BOTTLENECK\n307 GB/s Bus Saturation\n(Queue L = \u03bbW Blowup)", ha="center", va="center", fontsize=7.5, fontweight="bold", color="#7f1d1d")

    ax1.add_patch(patches.Rectangle((5.6, 1.8), 3.7, 1.8, facecolor="#fee2e2", edgecolor="#f87171", lw=1))
    ax1.text(7.45, 2.7, "Pinned In-Memory Index\n1.53 TB - 15.3 TB DRAM\nCost: $140 - $42,000/mo", ha="center", va="center", fontsize=7.5)

    # Panel B: EdgeVec Disaggregated Architecture
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")
    ax2.set_title("(b) EdgeVec Disaggregated Architecture", fontsize=9.5, fontweight="bold", pad=8)

    # Client Tier
    ax2.add_patch(patches.FancyBboxPatch((0.4, 1.2), 4.4, 8.0, boxstyle="round,pad=0.2", facecolor="#ecfdf5", edgecolor="#059669", lw=1.5))
    ax2.text(2.6, 8.6, "Client Browser (WASM SIMD Tier)", ha="center", va="center", fontweight="bold", fontsize=9, color="#065f46")

    ax2.add_patch(patches.Rectangle((0.7, 6.5), 3.8, 1.4, facecolor="#d1fae5", edgecolor="#34d399", lw=1))
    ax2.text(2.6, 7.2, "Local WASM SIMD Embedding\n(all-MiniLM-L6-v2 in 18 ms)", ha="center", va="center", fontsize=7.5)

    ax2.add_patch(patches.Rectangle((0.7, 4.6), 3.8, 1.4, facecolor="#d1fae5", edgecolor="#34d399", lw=1))
    ax2.text(2.6, 5.3, "Voronoi Routing & Multi-Probe\n(\u0394_boundary < 0.045 test)", ha="center", va="center", fontsize=7.5)

    ax2.add_patch(patches.Rectangle((0.7, 2.7), 3.8, 1.4, facecolor="#a7f3d0", edgecolor="#059669", lw=1.2))
    ax2.text(2.6, 3.4, "K-Anonymity Chaffing Engine\n(1 Genuine + 4 Decoy Requests)", ha="center", va="center", fontsize=7.5, fontweight="bold", color="#064e3b")

    ax2.add_patch(patches.Rectangle((0.7, 1.5), 3.8, 0.9, facecolor="#d1fae5", edgecolor="#34d399", lw=1))
    ax2.text(2.6, 1.95, "Local Int8 SIMD Cosine Ranking", ha="center", va="center", fontsize=7.5)

    # Arrow to CDN
    ax2.annotate("", xy=(5.8, 6.0), xytext=(4.8, 6.0), arrowprops=dict(arrowstyle="->", color="#0284c7", lw=1.5))
    ax2.text(5.3, 6.4, "Static HTTP GET\n/bucket_97.json", ha="center", va="bottom", fontsize=7, color="#0369a1", fontweight="bold")

    # CDN Edge Tier
    ax2.add_patch(patches.FancyBboxPatch((5.9, 2.2), 3.7, 6.5, boxstyle="round,pad=0.2", facecolor="#f0f9ff", edgecolor="#0284c7", lw=1.5))
    ax2.text(7.75, 8.1, "Stateless Anycast CDN", ha="center", va="center", fontweight="bold", fontsize=9, color="#075985")
    ax2.text(7.75, 7.4, "Fastly / Cloudflare Edge PoP", ha="center", va="center", fontsize=7.5, color="#0369a1")

    ax2.add_patch(patches.Rectangle((6.2, 4.8), 3.1, 2.1, facecolor="#e0f2fe", edgecolor="#38bdf8", lw=1))
    ax2.text(7.75, 5.85, "100 Static Inverted\nVoronoi Shards\n(Signed Int8 Quantized)", ha="center", va="center", fontsize=7.5)

    ax2.add_patch(patches.Rectangle((6.2, 2.6), 3.1, 1.8, facecolor="#bae6fd", edgecolor="#0284c7", lw=1.2))
    ax2.text(7.75, 3.5, "SERVER LOAD: 0 MB\nEdge Hit Rate: 99.99%\nCost: $0.00/month", ha="center", va="center", fontsize=7.5, fontweight="bold", color="#0c4a6e")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_architecture.pdf"))
    plt.savefig(os.path.join(FIG_DIR, "fig1_architecture.png"))
    plt.close()
    print("Generated Fig 1: Architecture Comparison")

# =============================================================================
# FIGURE 2: VORONOI GEOMETRY & MULTI-PROBE MARGIN
# =============================================================================
def generate_fig2_voronoi():
    fig, ax = plt.subplots(figsize=(3.5, 3.2))

    # Centroids
    c1 = np.array([2.5, 6.0])
    c2 = np.array([7.5, 6.0])
    c3 = np.array([5.0, 1.5])

    # Voronoi Boundary between c1 and c2 (perpendicular bisector)
    ax.plot([5.0, 5.0], [3.75, 9.5], "k--", lw=1.5, label="Voronoi Boundary $\mathcal{H}_{1,2}$")
    ax.plot([5.0, 1.5], [3.75, 0.5], "k--", lw=1.5)
    ax.plot([5.0, 8.5], [3.75, 0.5], "k--", lw=1.5)

    # Boundary threshold zone (tau = 0.045)
    ax.fill_betweenx([3.75, 9.5], 4.4, 5.6, color="#fef08a", alpha=0.4, label="Boundary Zone ($\pm\u03c4$)")

    # Plot Centroids
    ax.scatter([c1[0], c2[0], c3[0]], [c1[1], c2[1], c3[1]], color="#1e40af", s=70, zorder=5)
    ax.text(c1[0] - 0.4, c1[1] + 0.3, "Centroid $c_{(1)}$", fontsize=8, fontweight="bold")
    ax.text(c2[0] + 0.3, c2[1] + 0.3, "Centroid $c_{(2)}$", fontsize=8, fontweight="bold")
    ax.text(c3[0], c3[1] - 0.6, "Centroid $c_{(3)}$", fontsize=8, ha="center")

    # Query point q in boundary zone
    q = np.array([4.7, 7.2])
    ax.scatter([q[0]], [q[1]], color="#dc2626", s=90, marker="*", zorder=6, label="Query vector $q$")
    ax.text(q[0] - 0.3, q[1] + 0.4, "Query $q$", fontsize=8.5, fontweight="bold", color="#dc2626")

    # Distance lines from q to c1 and c2
    ax.plot([q[0], c1[0]], [q[1], c1[1]], "r:", lw=1.2)
    ax.plot([q[0], c2[0]], [q[1], c2[1]], "b:", lw=1.2)
    ax.text(3.4, 6.8, "$d_{(1)}$", color="#dc2626", fontsize=8)
    ax.text(6.3, 6.8, "$d_{(2)}$", color="#1e40af", fontsize=8)

    # Stewart's line between A and B
    ax.plot([c1[0], c2[0]], [c1[1], c2[1]], "g-", lw=1.2, alpha=0.7)
    o_point = c1 + 0.4 * (c2 - c1)
    ax.scatter([o_point[0]], [o_point[1]], color="#047857", s=40, zorder=5)
    ax.text(o_point[0] - 0.2, o_point[1] - 0.5, "Fine $O$", fontsize=7.5, color="#047857", fontweight="bold")

    ax.set_xlim(0.5, 9.5)
    ax.set_ylim(0.5, 9.8)
    ax.set_aspect("equal")
    ax.set_xlabel("Embedding Manifold Dimension $x_1$")
    ax.set_ylabel("Embedding Manifold Dimension $x_2$")
    ax.set_title("Voronoi Partitioning & Multi-Probe Margin", fontsize=9.5, fontweight="bold")
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig2_voronoi.pdf"))
    plt.savefig(os.path.join(FIG_DIR, "fig2_voronoi.png"))
    plt.close()
    print("Generated Fig 2: Voronoi Geometry")

# =============================================================================
# FIGURE 3: CONCURRENCY SCALING (Throughput & P95 Tail Latency)
# =============================================================================
def generate_fig3_concurrency():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.7))

    users = np.array([10, 50, 100, 250, 500, 1000, 2500, 5000, 10000])

    # Empirical data & queue model
    edgevec_qps = users * 22.0
    edgevec_p95 = np.array([44.2, 45.5, 46.8, 48.2, 49.8, 51.5, 52.2, 52.8, 53.0])

    # Centralized baseline (collapses due to memory bus & queue saturation)
    tkde_qps = np.array([98.0, 210.0, 257.4, 215.0, 160.0, 112.8, 65.0, 48.0, 42.1])
    tkde_p95 = np.array([4.7, 12.0, 24.5, 120.0, 580.0, 1850.0, 6200.0, 12400.0, 18500.0])

    faiss_qps = np.array([55.0, 125.0, 157.8, 130.0, 105.0, 86.9, 42.0, 28.0, 21.0])
    faiss_p95 = np.array([12.5, 28.0, 53.0, 240.0, 1150.0, 3420.0, 11000.0, 21500.0, 32000.0])

    # (a) Throughput QPS
    ax1.plot(users, edgevec_qps, "g-o", lw=1.8, ms=4, label="EdgeVec (Disaggregated)")
    ax1.plot(users, tkde_qps, "r-s", lw=1.8, ms=4, label="Fu et al. TKDE '26 GI")
    ax1.plot(users, faiss_qps, "b-^", lw=1.8, ms=4, label="Centralized FAISS-IVF")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Concurrent Virtual Users ($U$)")
    ax1.set_ylabel("Aggregate Throughput (QPS)")
    ax1.set_title("(a) Throughput Scaling Under Load", fontsize=9.5, fontweight="bold")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=7.5)

    # (b) P95 Latency
    ax2.plot(users, edgevec_p95, "g-o", lw=1.8, ms=4, label="EdgeVec (Invariant ~48ms)")
    ax2.plot(users, tkde_p95, "r-s", lw=1.8, ms=4, label="Fu et al. TKDE '26 GI")
    ax2.plot(users, faiss_p95, "b-^", lw=1.8, ms=4, label="Centralized FAISS-IVF")
    ax2.axhline(y=1000, color="gray", linestyle=":", label="1-sec SLA Threshold")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Concurrent Virtual Users ($U$)")
    ax2.set_ylabel("P95 Tail Latency (ms)")
    ax2.set_title("(b) P95 Latency Invariance vs. Collapse", fontsize=9.5, fontweight="bold")
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", fontsize=7.5)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig3_concurrency.pdf"))
    plt.savefig(os.path.join(FIG_DIR, "fig3_concurrency.png"))
    plt.close()
    print("Generated Fig 3: Concurrency Scaling")

# =============================================================================
# FIGURE 4: RECALL PARETO FRONTIER & QUANTIZATION FIDELITY
# =============================================================================
def generate_fig4_pareto():
    fig, ax = plt.subplots(figsize=(3.5, 2.7))

    configs = [
        {"name": "1-Probe (F32)", "rec5": 96.9, "payload": 425, "color": "#3b82f6", "marker": "o"},
        {"name": "2-Probe Fixed (F32)", "rec5": 100.0, "payload": 850, "color": "#6366f1", "marker": "s"},
        {"name": "3-Probe Fixed (F32)", "rec5": 100.0, "payload": 1275, "color": "#8b5cf6", "marker": "^"},
        {"name": "EdgeVec Adaptive (F32)", "rec5": 100.0, "payload": 720, "color": "#059669", "marker": "*"},
        {"name": "EdgeVec Adaptive (Int8)", "rec5": 91.5, "payload": 180, "color": "#10b981", "marker": "D"},
    ]

    for c in configs:
        ax.scatter(c["payload"], c["rec5"], color=c["color"], s=75 if c["marker"] != "*" else 120,
                   marker=c["marker"], label=c["name"], zorder=5)

    # Annotation callouts
    ax.annotate("EdgeVec Int8\n(4x compression, 91.5%)", xy=(180, 91.5), xytext=(220, 89.0),
                arrowprops=dict(arrowstyle="->", color="#059669", lw=1), fontsize=7.5, fontweight="bold")
    ax.annotate("EdgeVec Float32\n(100% Recall@5)", xy=(720, 100.0), xytext=(550, 97.5),
                arrowprops=dict(arrowstyle="->", color="#059669", lw=1), fontsize=7.5, fontweight="bold")

    ax.set_xlabel("Partition Shard Network Transfer (KB)")
    ax.set_ylabel("Top-5 Recall Accuracy (%)")
    ax.set_title("Recall Accuracy vs. Network Payload", fontsize=9.5, fontweight="bold")
    ax.set_ylim(87, 102)
    ax.set_xlim(100, 1400)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig4_pareto.pdf"))
    plt.savefig(os.path.join(FIG_DIR, "fig4_pareto.png"))
    plt.close()
    print("Generated Fig 4: Pareto Recall")

# =============================================================================
# FIGURE 5: LATENCY BREAKDOWN PROFILE
# =============================================================================
def generate_fig5_latency_breakdown():
    fig, ax = plt.subplots(figsize=(3.5, 2.7))

    stages = ["Vectorization\n(WASM SIMD)", "Routing\n(K=100 L2)", "CDN Fetch\n(HTTP/2)", "Local Cosine\nRanking"]
    f32_ms = [18.2, 1.1, 68.0, 12.0]
    int8_ms = [18.2, 1.1, 23.8, 3.2]

    x = np.arange(len(stages))
    width = 0.35

    rects1 = ax.bar(x - width/2, f32_ms, width, label="Float32 (~425 KB)", color="#94a3b8", edgecolor="black", lw=0.8)
    rects2 = ax.bar(x + width/2, int8_ms, width, label="Int8 (~180 KB)", color="#10b981", edgecolor="black", lw=0.8)

    ax.set_ylabel("Execution Latency (ms)")
    ax.set_title("Query Pipeline Stage Latency Breakdown", fontsize=9.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=7.5)
    ax.legend(fontsize=7.5)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Annotate total
    ax.text(0.95, 0.90, f"Total Float32: {sum(f32_ms):.1f} ms\nTotal Int8: {sum(int8_ms):.1f} ms\nSpeedup: 2.1x",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8fafc", edgecolor="#cbd5e1"))

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig5_breakdown.pdf"))
    plt.savefig(os.path.join(FIG_DIR, "fig5_breakdown.png"))
    plt.close()
    print("Generated Fig 5: Latency Breakdown")

# =============================================================================
# FIGURE 6: K-ANONYMITY CHAFFING PRIVACY BOUNDS
# =============================================================================
def generate_fig6_privacy():
    fig, ax1 = plt.subplots(figsize=(3.5, 2.7))

    k_anon = np.arange(1, 11)
    guess_prob = (1.0 / k_anon) * 100.0
    entropy = np.log2(k_anon)

    color_prob = "#dc2626"
    color_ent = "#0284c7"

    ax1.set_xlabel("Anonymity Degree ($K_{\mathrm{anon}}$)")
    ax1.set_ylabel("Adversary Guess Probability (%)", color=color_prob)
    line1 = ax1.plot(k_anon, guess_prob, color=color_prob, marker="o", lw=1.8, label="Guess Probability ($1/K$)")
    ax1.tick_params(axis="y", labelcolor=color_prob)
    ax1.set_ylim(0, 105)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Shannon Entropy $H(X)$ (Bits)", color=color_ent)
    line2 = ax2.plot(k_anon, entropy, color=color_ent, marker="s", lw=1.8, linestyle="--", label="Entropy $H(X)$")
    ax2.tick_params(axis="y", labelcolor=color_ent)
    ax2.set_ylim(0, 3.8)

    # Highlight K=5 (EdgeVec default)
    ax1.axvline(x=5, color="#10b981", linestyle=":", lw=1.5)
    ax1.text(5.2, 50, "EdgeVec Default ($K=5$)\n$P=20\%$, $H=2.32$ bits", fontsize=7.5, color="#065f46", fontweight="bold")

    ax1.set_title("Information-Theoretic Privacy Proof", fontsize=9.5, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig6_privacy.pdf"))
    plt.savefig(os.path.join(FIG_DIR, "fig6_privacy.png"))
    plt.close()
    print("Generated Fig 6: Privacy Chaffing")

if __name__ == "__main__":
    generate_fig1_architecture()
    generate_fig2_voronoi()
    generate_fig3_concurrency()
    generate_fig4_pareto()
    generate_fig5_latency_breakdown()
    generate_fig6_privacy()
    print("All 6 publication figures generated successfully in paper/figures/")
