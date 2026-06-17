from scipy.spatial import cKDTree
import numpy as np
import pandas as pd


# =========================================================
# build KD-tree for graph nodes
# =========================================================


def build_node_kdtree(G):

    nodes = list(G.nodes())

    coords = np.array([(G.nodes[n]["y"], G.nodes[n]["x"]) for n in nodes])

    tree = cKDTree(coords)

    return nodes, coords, tree


# =========================================================
# nearest node using KD-tree
# =========================================================


def nearest_node_kdtree(nodes, tree, lat, lon):

    dist, idx = tree.query([lat, lon])

    return nodes[idx]


# =========================================================
# node coverage using KD-tree
# =========================================================


def node_coverage_kdtree(nodes, coords, tree, node_idx, radius_deg):

    center = coords[node_idx]

    idxs = tree.query_ball_point(center, radius_deg)

    return {nodes[i] for i in idxs}


# =========================================================
# convert meters → degrees (approx)
# =========================================================


def meters_to_degrees(radius_m):

    return radius_m / 111320.0


# =========================================================
# compute node demand from congestion
# =========================================================


def compute_node_demand(G):

    demand = {}

    for node in G.nodes():

        weights = [data.get("congestion", 0) for _, _, data in G.edges(node, data=True)]

        demand[node] = float(np.mean(weights)) if weights else 0

    return demand


# =========================================================
# vehicle density → mobile RSU candidates
# =========================================================


def vehicle_node_density(G, vehicle_df, nodes, tree):

    visits = {}

    for _, row in vehicle_df.iterrows():

        lat = row["Latitude"]
        lon = row["Longitude"]

        node = nearest_node_kdtree(nodes, tree, lat, lon)

        visits[node] = visits.get(node, 0) + 1

    ranked = sorted(visits.items(), key=lambda x: x[1], reverse=True)

    return [n for n, _ in ranked]


# =========================================================
# Hybrid RSU deployment (KD-tree optimized)
# =========================================================


def hybrid_rsu_deployment(
    G, vehicle_df, total_rsu_budget=20, srsu_radius=300, mrsu_radius=500, theta=0.2
):
    """
    Mobile-first hybrid RSU deployment.

    The method first selects mobile RSU candidate nodes derived from public
    transport trajectories. The remaining budget is then used to place static
    RSUs in locations that complement the mobile coverage.

    Returns:
        S_static: list of graph nodes selected for static RSUs
        M_mobile: list of graph nodes selected as mobile RSU opportunities
    """

    # -----------------------------------------------------
    # internal configuration
    # -----------------------------------------------------

    mobile_budget_share = 0.70  # budget share reserved for mobile opportunities
    mobile_alpha = 0.45  # lower value gives more graph-coverage weight
    static_alpha = 0.75  # higher value gives more traffic-demand weight
    mobility_bias = 0.30  # boosts frequently visited PT locations

    # -----------------------------------------------------
    # graph preparation
    # -----------------------------------------------------

    nodes, coords, tree = build_node_kdtree(G)
    node_index = {n: i for i, n in enumerate(nodes)}

    # Prefer already stored node demand if available.
    has_node_demand = any(float(G.nodes[n].get("demand", 0) or 0) > 0 for n in nodes)

    if has_node_demand:
        demand = {n: float(G.nodes[n].get("demand", 0) or 0) for n in nodes}
    else:
        demand = compute_node_demand(G)

        # Keep compatibility with your existing evaluation code, which reads
        # G.nodes[n]["demand"].
        for n in nodes:
            G.nodes[n]["demand"] = demand.get(n, 0.0)

    total_demand = sum(demand.get(n, 0.0) for n in nodes)

    # Candidate static RSU locations: high-demand nodes.
    H = [n for n in nodes if demand.get(n, 0.0) >= theta]

    # Fallback if theta is too strict.
    if not H:
        nonzero = [demand.get(n, 0.0) for n in nodes if demand.get(n, 0.0) > 0]

        if nonzero:
            q = np.quantile(nonzero, 0.75)
            H = [n for n in nodes if demand.get(n, 0.0) >= q]
        else:
            H = list(nodes)

    # -----------------------------------------------------
    # derive mobile candidate nodes from trajectories
    # -----------------------------------------------------

    visits = {}

    for _, row in vehicle_df.iterrows():
        if "Latitude" not in row or "Longitude" not in row:
            continue

        lat = row["Latitude"]
        lon = row["Longitude"]

        if pd.isna(lat) or pd.isna(lon):
            continue

        node = nearest_node_kdtree(nodes, tree, lat, lon)
        visits[node] = visits.get(node, 0) + 1

    ranked_mobile_nodes = [
        n for n, _ in sorted(visits.items(), key=lambda x: x[1], reverse=True)
    ]

    # No mobile data available: fall back to static-only greedy placement.
    if not ranked_mobile_nodes:
        ranked_mobile_nodes = []

    radius_s = meters_to_degrees(srsu_radius)
    radius_m = meters_to_degrees(mrsu_radius)

    # -----------------------------------------------------
    # scoring function
    # -----------------------------------------------------

    def weighted_gain(new_nodes, alpha):
        """
        alpha controls the demand-vs-graph tradeoff.

        alpha = 1.0 -> traffic demand only
        alpha = 0.0 -> graph coverage only
        """

        if not new_nodes:
            return 0.0

        if total_demand > 0:
            traffic_gain = sum(demand.get(n, 0.0) for n in new_nodes) / total_demand
        else:
            traffic_gain = 0.0

        graph_gain = len(new_nodes) / max(1, len(nodes))

        return alpha * traffic_gain + (1.0 - alpha) * graph_gain

    # -----------------------------------------------------
    # coverage caches
    # -----------------------------------------------------

    mobile_cover_cache = {}
    static_cover_cache = {}

    def get_mobile_cover(node):
        if node not in mobile_cover_cache:
            mobile_cover_cache[node] = node_coverage_kdtree(
                nodes, coords, tree, node_index[node], radius_m
            )
        return mobile_cover_cache[node]

    def get_static_cover(node):
        if node not in static_cover_cache:
            static_cover_cache[node] = node_coverage_kdtree(
                nodes, coords, tree, node_index[node], radius_s
            )
        return static_cover_cache[node]

    # -----------------------------------------------------
    # Phase 1: select mobile RSU opportunities first
    # -----------------------------------------------------

    mobile_budget = int(round(total_rsu_budget * mobile_budget_share))
    mobile_budget = max(0, min(mobile_budget, total_rsu_budget))
    mobile_budget = min(mobile_budget, len(ranked_mobile_nodes))

    S_static = []
    M_mobile = []
    coverage = set()

    max_visits = max(visits.values()) if visits else 1

    while len(M_mobile) < mobile_budget:
        best_node = None
        best_score = 0.0
        best_cover = set()

        for node in ranked_mobile_nodes:
            if node in M_mobile:
                continue

            cover = get_mobile_cover(node)
            new_cover = cover - coverage

            base_score = weighted_gain(new_cover, mobile_alpha)
            visit_score = visits.get(node, 0) / max_visits

            # Coverage is dominant; visit frequency only boosts frequently
            # traversed public transport locations.
            score = base_score * (1.0 + mobility_bias * visit_score)

            if score > best_score:
                best_score = score
                best_node = node
                best_cover = cover

        if best_node is None or best_score <= 0:
            break

        M_mobile.append(best_node)
        coverage |= best_cover

    # -----------------------------------------------------
    # Phase 2: place static RSUs into remaining blind spots
    # -----------------------------------------------------

    static_budget = total_rsu_budget - len(M_mobile)

    fallback_static_alpha = 0.25
    # Lower alpha in the fallback stage means that once traffic hotspots
    # are already covered, static RSUs are used mainly to improve graph coverage.

    while len(S_static) < static_budget:

        excluded = set(S_static) | set(M_mobile)

        best_node = None
        best_score = -np.inf
        best_cover = set()

        # -------------------------------------------------
        # Stage 1: try high-demand candidates first
        # -------------------------------------------------

        for node in H:

            if node in excluded:
                continue

            cover = get_static_cover(node)
            new_cover = cover - coverage

            if not new_cover:
                continue

            score = weighted_gain(new_cover, static_alpha)

            if score > best_score:
                best_score = score
                best_node = node
                best_cover = cover

        # -------------------------------------------------
        # Stage 2: if high-demand candidates add no coverage,
        # fall back to all graph nodes to cover remaining blind spots
        # -------------------------------------------------

        if best_node is None:

            for node in nodes:

                if node in excluded:
                    continue

                cover = get_static_cover(node)
                new_cover = cover - coverage

                if not new_cover:
                    continue

                score = weighted_gain(new_cover, fallback_static_alpha)

                if score > best_score:
                    best_score = score
                    best_node = node
                    best_cover = cover

        # -------------------------------------------------
        # Stage 3: if coverage is already saturated, still fill
        # the remaining deployment budget with the highest-demand
        # unused node so that unit counts match the budget.
        # -------------------------------------------------

        if best_node is None:

            remaining = [node for node in nodes if node not in excluded]

            if not remaining:
                break

            best_node = max(remaining, key=lambda n: demand.get(n, 0.0))

            best_cover = get_static_cover(best_node)

        S_static.append(best_node)
        coverage |= best_cover

    # safeguard: keep output compatible with your evaluation code
    S_static = [n for n in S_static if n in G.nodes]
    M_mobile = [n for n in M_mobile if n in G.nodes]

    return S_static, M_mobile
