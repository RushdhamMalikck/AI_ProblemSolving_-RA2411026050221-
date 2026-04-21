import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Map Coloring CSP Solver",
    page_icon="🗺️",
    layout="wide"
)

# ─────────────────────────────────────────────
#  Color palette
# ─────────────────────────────────────────────
ALL_COLORS = {
    "Crimson":  "#E8576A",
    "Indigo":   "#6C7FD8",
    "Emerald":  "#3DBF8C",
    "Marigold": "#F5A623",
    "Violet":   "#9B6DD6",
    "Sky":      "#38AEDC",
    "Coral":    "#F07050",
    "Mint":     "#5DCAA5",
    "Rose":     "#E8709A",
    "Gold":     "#D4AF37",
    "Slate":    "#7B9EB8",
    "Peach":    "#F5A58C",
}

# ─────────────────────────────────────────────
#  CSP Backtracking Solver
# ─────────────────────────────────────────────
def is_consistent(region, color, assignment, adjacency):
    for neighbor in adjacency.get(region, []):
        if assignment.get(neighbor) == color:
            return False
    return True

def backtrack(assignment, regions, adjacency, colors):
    if len(assignment) == len(regions):
        return assignment
    unassigned = [r for r in regions if r not in assignment]
    # Pick the region with most assigned neighbors first (degree heuristic)
    region = max(unassigned,
                 key=lambda r: sum(1 for nb in adjacency.get(r, []) if nb in assignment))
    for color in colors:
        if is_consistent(region, color, assignment, adjacency):
            assignment[region] = color
            result = backtrack(assignment, regions, adjacency, colors)
            if result is not None:
                return result
            del assignment[region]
    return None

def compute_chromatic_lower(regions, adjacency):
    if not regions:
        return 0
    max_clique = 1
    for r in regions:
        nbrs = set(adjacency.get(r, []))
        clique = 1
        for r2 in regions:
            if r2 != r and r2 in nbrs:
                shared = len([x for x in adjacency.get(r2, []) if x in nbrs])
                if shared >= clique - 1:
                    clique += 1
        if clique > max_clique:
            max_clique = clique
    return max_clique

def compute_max_degree(regions, adjacency):
    if not regions:
        return 0
    return max(len(adjacency.get(r, [])) for r in regions)

# ─────────────────────────────────────────────
#  Session state init
# ─────────────────────────────────────────────
if "regions" not in st.session_state:
    st.session_state.regions = []
if "adjacency" not in st.session_state:
    st.session_state.adjacency = {}
if "solution" not in st.session_state:
    st.session_state.solution = None
if "selected_colors" not in st.session_state:
    st.session_state.selected_colors = ["Crimson", "Indigo", "Emerald", "Marigold"]

# ─────────────────────────────────────────────
#  Title
# ─────────────────────────────────────────────
st.title("🗺️ Map Coloring — CSP Solver")
st.markdown("**Problem 5 · Constraint Satisfaction Problem (CSP)**")
st.divider()

# ─────────────────────────────────────────────
#  Layout: left panel | right panel
# ─────────────────────────────────────────────
left, right = st.columns([1, 1.6], gap="large")

# ══════════════════════════════════════════════
#  LEFT PANEL
# ══════════════════════════════════════════════
with left:

    # ── Regions ──────────────────────────────
    st.subheader("1. Add Regions")
    region_input = st.text_input(
        "Enter region names (comma separated)",
        placeholder="e.g. A, B, C, D"
    )
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("➕ Add Regions", use_container_width=True):
            new_regions = [r.strip().upper() for r in region_input.split(",") if r.strip()]
            added = 0
            for r in new_regions:
                if r and r not in st.session_state.regions:
                    st.session_state.regions.append(r)
                    st.session_state.adjacency[r] = []
                    added += 1
            if added:
                st.session_state.solution = None
                st.success(f"Added {added} region(s)!")
    with col2:
        if st.button("🔄 Load Example", use_container_width=True):
            st.session_state.regions = ["A", "B", "C", "D", "E", "F"]
            st.session_state.adjacency = {
                "A": ["B", "C"],
                "B": ["A", "C", "D"],
                "C": ["A", "B", "D", "E"],
                "D": ["B", "C", "E", "F"],
                "E": ["C", "D", "F"],
                "F": ["D", "E"],
            }
            st.session_state.solution = None
            st.success("Example loaded!")

    if st.session_state.regions:
        st.markdown("**Current regions:** " +
                    " · ".join([f"`{r}`" for r in st.session_state.regions]))
        # Remove region
        remove_r = st.selectbox("Remove a region", ["— select —"] + st.session_state.regions, key="remR")
        if st.button("🗑️ Remove Region"):
            if remove_r != "— select —":
                st.session_state.regions.remove(remove_r)
                del st.session_state.adjacency[remove_r]
                for k in st.session_state.adjacency:
                    st.session_state.adjacency[k] = [
                        x for x in st.session_state.adjacency[k] if x != remove_r
                    ]
                st.session_state.solution = None
                st.rerun()

    st.divider()

    # ── Adjacency ────────────────────────────
    st.subheader("2. Define Adjacency")
    if len(st.session_state.regions) >= 2:
        c1, c2 = st.columns(2)
        with c1:
            from_r = st.selectbox("Region A", st.session_state.regions, key="fromR")
        with c2:
            to_options = [r for r in st.session_state.regions if r != from_r]
            to_r = st.selectbox("Region B", to_options, key="toR")

        if st.button("🔗 Link as Neighbors", use_container_width=True):
            if to_r not in st.session_state.adjacency[from_r]:
                st.session_state.adjacency[from_r].append(to_r)
                st.session_state.adjacency[to_r].append(from_r)
                st.session_state.solution = None
                st.success(f"{from_r} ↔ {to_r} linked!")
            else:
                st.warning("Already linked!")

        # Show current adjacency
        seen = set()
        links = []
        for a in st.session_state.regions:
            for b in st.session_state.adjacency.get(a, []):
                key = tuple(sorted([a, b]))
                if key not in seen:
                    seen.add(key)
                    links.append(f"{a} — {b}")
        if links:
            st.markdown("**Links:** " + " · ".join([f"`{l}`" for l in links]))
    else:
        st.info("Add at least 2 regions first.")

    st.divider()

    # ── Color selection ───────────────────────
    st.subheader("3. Select Colors")
    selected = st.multiselect(
        "Choose colors for the solver",
        options=list(ALL_COLORS.keys()),
        default=st.session_state.selected_colors
    )
    st.session_state.selected_colors = selected

    # Color swatches
    if selected:
        swatch_html = "".join([
            f'<span style="display:inline-block;width:20px;height:20px;'
            f'border-radius:50%;background:{ALL_COLORS[c]};'
            f'margin-right:4px;vertical-align:middle;" title="{c}"></span>'
            f'<span style="font-size:12px;margin-right:10px;">{c}</span>'
            for c in selected
        ])
        st.markdown(swatch_html, unsafe_allow_html=True)

    # Smart hint
    if st.session_state.regions:
        lower = compute_chromatic_lower(
            st.session_state.regions, st.session_state.adjacency)
        upper = compute_max_degree(
            st.session_state.regions, st.session_state.adjacency) + 1
        st.markdown("---")
        st.markdown("**💡 Smart Hint**")
        st.markdown(f"- Minimum colors needed: **at least {lower}**")
        st.markdown(f"- Safe upper bound: **at most {upper}** (max degree + 1)")
        if len(selected) < lower:
            st.error(f"⚠️ You selected {len(selected)} color(s) — need at least {lower}. Please add more!")
        else:
            st.success(f"✅ {len(selected)} colors selected — enough to solve!")

    st.divider()

    # ── Solve button ──────────────────────────
    st.subheader("4. Solve")
    col_s, col_r = st.columns(2)
    with col_s:
        solve_clicked = st.button("🧠 Solve CSP", use_container_width=True, type="primary")
    with col_r:
        if st.button("🗑️ Reset All", use_container_width=True):
            st.session_state.regions = []
            st.session_state.adjacency = {}
            st.session_state.solution = None
            st.session_state.selected_colors = ["Crimson", "Indigo", "Emerald", "Marigold"]
            st.rerun()

    if solve_clicked:
        if not st.session_state.regions:
            st.error("Add regions first!")
        elif not selected:
            st.error("Select at least one color!")
        else:
            result = backtrack(
                {}, st.session_state.regions,
                st.session_state.adjacency, selected
            )
            if result:
                st.session_state.solution = result
                st.success("✅ Solution found!")
            else:
                st.session_state.solution = None
                st.error("❌ No solution — try adding more colors!")

# ══════════════════════════════════════════════
#  RIGHT PANEL — Visualization
# ══════════════════════════════════════════════
with right:
    st.subheader("Map Visualization")

    regions = st.session_state.regions
    adjacency = st.session_state.adjacency
    solution = st.session_state.solution

    if not regions:
        st.info("Add regions on the left to see the graph here.")
    else:
        # Build graph
        G = nx.Graph()
        G.add_nodes_from(regions)
        seen = set()
        for a in regions:
            for b in adjacency.get(a, []):
                key = tuple(sorted([a, b]))
                if key not in seen:
                    seen.add(key)
                    G.add_edge(a, b)

        # Node colors
        if solution:
            node_colors = [ALL_COLORS.get(solution.get(r, "Slate"), "#7B9EB8") for r in G.nodes()]
        else:
            node_colors = ["#e8e8e8"] * len(G.nodes())

        # Draw
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor("#f7f6f2")
        ax.set_facecolor("#f7f6f2")

        pos = nx.spring_layout(G, seed=42, k=1.5)

        nx.draw_networkx_edges(G, pos, ax=ax,
                               edge_color="#cccccc", width=2)
        nx.draw_networkx_nodes(G, pos, ax=ax,
                               node_color=node_colors,
                               node_size=1200,
                               edgecolors="#555555",
                               linewidths=1.5)
        nx.draw_networkx_labels(G, pos, ax=ax,
                                font_size=11,
                                font_weight="bold",
                                font_color="#1a1a1a")
        ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Solution table
        if solution:
            st.markdown("**Solution:**")
            cols = st.columns(min(len(solution), 4))
            for i, (region, color) in enumerate(solution.items()):
                hex_color = ALL_COLORS.get(color, "#ccc")
                with cols[i % len(cols)]:
                    st.markdown(
                        f'<div style="background:{hex_color};padding:8px 12px;'
                        f'border-radius:20px;text-align:center;'
                        f'font-weight:500;font-size:13px;color:#1a1a1a;'
                        f'margin-bottom:6px;">'
                        f'{region} → {color}</div>',
                        unsafe_allow_html=True
                    )
            used = len(set(solution.values()))
            st.markdown(f"**Colors used:** {used} out of {len(selected)} selected")

            # Legend
            used_colors = list(set(solution.values()))
            patches = [mpatches.Patch(color=ALL_COLORS[c], label=c) for c in used_colors]
            fig2, ax2 = plt.subplots(figsize=(5, 0.5))
            fig2.patch.set_facecolor("#f7f6f2")
            ax2.legend(handles=patches, loc="center", ncol=len(patches),
                       frameon=False, fontsize=10)
            ax2.axis("off")
            st.pyplot(fig2)
            plt.close()

# ─────────────────────────────────────────────
#  How it works section
# ─────────────────────────────────────────────
st.divider()
with st.expander("📖 How does this work? (CSP Backtracking Algorithm)"):
    st.markdown("""
### What is Map Coloring?
Map coloring is the problem of assigning colors to regions on a map such that **no two adjacent regions share the same color**.

### What is CSP?
A **Constraint Satisfaction Problem (CSP)** consists of:
- **Variables** → Each region is a variable
- **Domain** → The set of available colors
- **Constraints** → Adjacent regions must have different colors

### Algorithm: Backtracking Search
1. Pick an unassigned region (using degree heuristic — most constrained first)
2. Try assigning each available color
3. Check if the assignment is consistent (no neighbor has the same color)
4. If consistent → move to next region
5. If no color works → backtrack to previous region and try a different color
6. Repeat until all regions are assigned

### Four Color Theorem
The famous **Four Color Theorem** states that any map can be colored using at most **4 colors** such that no adjacent regions share the same color.
    """)

