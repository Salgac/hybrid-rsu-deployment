import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# distance helper
# =========================================================


def _distance_m(lat1, lon1, lat2, lon2):

    meter_per_deg_lat = 111320
    meter_per_deg_lon = 111320 * np.cos(np.radians((lat1 + lat2) / 2))

    dx = (lon1 - lon2) * meter_per_deg_lon
    dy = (lat1 - lat2) * meter_per_deg_lat

    return np.sqrt(dx**2 + dy**2)


# =========================================================
# nodes covered
# =========================================================


def nodes_covered(G, rsu_nodes, radius):

    covered = set()

    for rsu in rsu_nodes:

        lat1 = G.nodes[rsu]["y"]
        lon1 = G.nodes[rsu]["x"]

        for node in G.nodes():

            lat2 = G.nodes[node]["y"]
            lon2 = G.nodes[node]["x"]

            if _distance_m(lat1, lon1, lat2, lon2) <= radius:
                covered.add(node)

    return covered


# =========================================================
# graph coverage
# =========================================================


def graph_coverage(G, rsu_nodes, radius):

    covered = nodes_covered(G, rsu_nodes, radius)

    return len(covered) / len(G.nodes())


# =========================================================
# traffic coverage
# =========================================================


def traffic_coverage(G, rsu_nodes, radius):

    covered = nodes_covered(G, rsu_nodes, radius)

    total_demand = sum(G.nodes[n].get("demand", 0) for n in G.nodes())

    if total_demand == 0:
        return 0

    covered_demand = sum(G.nodes[n].get("demand", 0) for n in covered)

    return covered_demand / total_demand


# =========================================================
# hybrid coverage
# =========================================================


def hybrid_coverage(G, S_static, M_mobile, r_static, r_mobile):

    covered_static = nodes_covered(G, S_static, r_static)
    covered_mobile = nodes_covered(G, M_mobile, r_mobile)

    covered = covered_static | covered_mobile

    total_demand = sum(G.nodes[n].get("demand", 0) for n in G.nodes())

    if total_demand == 0:
        traffic_cov = 0
    else:
        covered_demand = sum(G.nodes[n].get("demand", 0) for n in covered)
        traffic_cov = covered_demand / total_demand

    graph_cov = len(covered) / len(G.nodes())

    return graph_cov, traffic_cov


def hybrid_coverage_metrics(G, S_static, M_mobile, r_static, r_mobile):

    covered_static = nodes_covered(G, S_static, r_static)
    covered_mobile = nodes_covered(G, M_mobile, r_mobile)

    covered_total = covered_static | covered_mobile

    total_nodes = len(G.nodes())

    graph_static = len(covered_static) / total_nodes
    graph_mobile = len(covered_mobile) / total_nodes
    graph_total = len(covered_total) / total_nodes

    if len(covered_total) == 0:
        guarantee_ratio = 0
    else:
        guarantee_ratio = len(covered_static) / len(covered_total)

    return {
        "graph_static_coverage": graph_static,
        "graph_mobile_coverage": graph_mobile,
        "graph_total_coverage": graph_total,
        "guaranteed_ratio": guarantee_ratio,
    }


# =========================================================
# run experiment
# =========================================================


def run_experiment(
    G,
    vehicles,
    strategies,
    budgets,
    r_static=300,
    r_mobile=500,
    save_csv="csv/rsu_experiment_results.csv",
):

    rows = []

    total_runs = len(strategies) * len(budgets)
    run_counter = 1

    for name, strategy_fn in strategies.items():

        print(f"\n=== Running strategy: {name} ===")

        for b in budgets:

            print(f"[{run_counter}/{total_runs}] Budget={b}")

            if name == "hybrid":

                S_static, M_mobile = strategy_fn(
                    G,
                    vehicles,
                    total_rsu_budget=b,
                    srsu_radius=r_static,
                    mrsu_radius=r_mobile,
                )

                graph_cov, traffic_cov = hybrid_coverage(
                    G, S_static, M_mobile, r_static, r_mobile
                )

                metrics = hybrid_coverage_metrics(
                    G, S_static, M_mobile, r_static, r_mobile
                )

                static_count = len(S_static)
                mobile_count = len(M_mobile)

            else:

                rsu_nodes = strategy_fn(G, b)

                graph_cov = graph_coverage(G, rsu_nodes, r_static)
                traffic_cov = traffic_coverage(G, rsu_nodes, r_static)

                metrics = {
                    "graph_static_coverage": graph_cov,
                    "graph_mobile_coverage": 0,
                    "graph_total_coverage": graph_cov,
                    "guaranteed_ratio": 1,
                }

                static_count = len(rsu_nodes)
                mobile_count = 0

            print(
                f"   graph_cov={graph_cov:.3f}, "
                f"traffic_cov={traffic_cov:.3f}, "
                f"static={static_count}, mobile={mobile_count}"
            )

            rows.append(
                {
                    "strategy": name,
                    "budget": b,
                    "static_rsus": static_count,
                    "mobile_rsus": mobile_count,
                    "graph_coverage": graph_cov,
                    "traffic_coverage": traffic_cov,
                    **metrics,
                }
            )

            run_counter += 1

    df = pd.DataFrame(rows)

    df.to_csv(save_csv, index=False)

    print(f"\nResults saved to {save_csv}")

    return df


def build_comparison_table(df):

    graph_table = (
        df.pivot_table(index="budget", columns="strategy", values="graph_coverage")
        .reset_index()
        .rename(columns={"budget": "num_rsus"})
    )

    traffic_table = (
        df.pivot_table(index="budget", columns="strategy", values="traffic_coverage")
        .reset_index()
        .rename(columns={"budget": "num_rsus"})
    )

    return graph_table, traffic_table


def plot_coverage(table, title):

    plt.figure(figsize=(6, 4))

    for col in table.columns[1:]:
        plt.plot(table["num_rsus"], table[col], marker="o", label=col)

    plt.xlabel("Total RSU budget")
    plt.ylabel("Coverage")
    plt.title(title)

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


import seaborn as sns
import matplotlib.pyplot as plt


def plot_coverage_curves_seaborn(df):

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(7, 5))

    sns.lineplot(data=df, x="budget", y="traffic_coverage", hue="strategy", marker="o")

    plt.xlabel("Total RSU budget")
    plt.ylabel("Traffic coverage")

    plt.tight_layout()
    plt.show()


def plot_graph_coverage_seaborn(df):

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(7, 5))

    sns.lineplot(data=df, x="budget", y="graph_coverage", hue="strategy", marker="o")

    plt.xlabel("Total RSU budget")
    plt.ylabel("Graph coverage")

    plt.tight_layout()
    plt.show()


def plot_hybrid_allocation(df):

    sns.set_theme(style="whitegrid")

    hybrid = df[df.strategy == "hybrid"]

    plt.figure(figsize=(7, 5))

    sns.lineplot(data=hybrid, x="budget", y="static_rsus", label="Static RSUs")

    sns.lineplot(data=hybrid, x="budget", y="mobile_rsus", label="Mobile RSUs")

    plt.xlabel("Total RSU budget")
    plt.ylabel("Number of RSUs")

    plt.tight_layout()
    plt.show()


def plot_hybrid_allocation(df):

    sns.set_theme(style="whitegrid")

    hybrid = df[df.strategy == "hybrid"]

    plt.figure(figsize=(7, 5))

    sns.lineplot(data=hybrid, x="budget", y="static_rsus", label="Static RSUs")

    sns.lineplot(data=hybrid, x="budget", y="mobile_rsus", label="Mobile RSUs")

    plt.xlabel("Total RSU budget")
    plt.ylabel("Number of RSUs")

    plt.tight_layout()
    plt.show()


def plot_guaranteed_ratio(df):

    sns.set_theme(style="whitegrid")

    hybrid = df[df.strategy == "hybrid"]

    plt.figure(figsize=(7, 5))

    sns.lineplot(data=hybrid, x="budget", y="guaranteed_ratio", marker="o")

    plt.xlabel("Total RSU budget")
    plt.ylabel("Guaranteed coverage ratio")

    plt.tight_layout()
    plt.show()


def plot_combined_coverage(df, save_path=None):

    # Publication style
    sns.set_theme(context="paper", style="ticks", font_scale=1.2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # -----------------------------------------------------
    # Traffic coverage
    # -----------------------------------------------------

    sns.lineplot(
        data=df,
        x="budget",
        y="traffic_coverage",
        hue="strategy",
        linewidth=2,
        ax=axes[0],
    )

    axes[0].set_xlabel("Total RSU Budget")
    axes[0].set_ylabel("Traffic Coverage")

    axes[0].set_title("(a) Traffic Coverage")

    axes[0].set_ylim(0, 1.05)

    axes[0].grid(True, linestyle="--", alpha=0.4)

    # -----------------------------------------------------
    # Graph coverage
    # -----------------------------------------------------

    sns.lineplot(
        data=df,
        x="budget",
        y="graph_coverage",
        hue="strategy",
        linewidth=2,
        ax=axes[1],
        legend=False,
    )

    axes[1].set_xlabel("Total RSU Budget")
    axes[1].set_ylabel("Graph Coverage")

    axes[1].set_title("(b) Graph Coverage")

    axes[1].set_ylim(0, 1.05)

    axes[1].grid(True, linestyle="--", alpha=0.4)

    # -----------------------------------------------------
    # Legend
    # -----------------------------------------------------

    axes[1].legend(title="Deployment Strategy", loc="lower right", frameon=True)

    sns.despine()

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
