# AI Problem Solving — Map Coloring (CSP)

## Problem Statement
In a map coloring scenario, different regions on a map must be colored such that **no two adjacent regions share the same color**. The user can input regions and their neighboring relationships through an interactive GUI, and the system assigns colors to each region while satisfying all constraints.

---

## Algorithm Used
**Constraint Satisfaction Problem (CSP) — Backtracking Search**

### How it works:
1. Each **region** is treated as a variable
2. The **available colors** form the domain
3. The **constraint** is: no two adjacent regions can share the same color
4. The solver uses **backtracking** with a **degree heuristic** (most constrained region first)
5. If no color works for a region, it backtracks to the previous region and tries a different color

### Smart Hint System:
- Computes the **chromatic lower bound** (minimum colors needed)
- Computes the **safe upper bound** (max degree + 1)
- Warns the user if they have selected too few colors before solving

### Four Color Theorem:
Any map can be colored using at most **4 colors** such that no two adjacent regions share the same color.

---

## Technologies Used
- **Python 3**
- **Streamlit** — interactive web-based GUI
- **NetworkX** — graph construction
- **Matplotlib** — graph visualization

---

## Folder Structure
```
Problem5_MapColoring/
│
├── app.py          ← Main Streamlit application
├── README.md       ← Project documentation
```

---

## Execution Steps

### 1. Install dependencies
```bash
python -m pip install streamlit networkx matplotlib
```

### 2. Run the app
```bash
python -m streamlit run app.py
```

### 3. Open in browser
```
http://localhost:8501
```

---

## How to Use
1. **Add Regions** — Type region names (e.g. A, B, C, D) and click Add
2. **Define Adjacency** — Select two regions and click Link as Neighbors
3. **Select Colors** — Choose colors from the multiselect dropdown
4. **Check Smart Hint** — The app tells you the minimum colors needed
5. **Solve CSP** — Click Solve and see the colored graph!
6. **Load Example** — Click Load Example for a quick demo

---

## Sample Input
```
Regions: A, B, C, D
Adjacency:
  A → B, C
  B → A, C, D
  C → A, B, D
  D → B, C
Colors available: Crimson, Indigo, Emerald, Marigold
```

## Sample Output
```
A → Crimson
B → Indigo
C → Emerald
D → Crimson
Colors used: 3 out of 4 selected
```

---

## Constraints Satisfied
- ✅ No two adjacent regions have the same color
- ✅ Minimum number of colors is used
- ✅ User can select from 12 different colors dynamically
- ✅ Smart hint tells user the minimum colors needed before solving

---

## Team
| Name | Register Number |
|------|----------------|
| RUSHDHAM MALIK CK | RA2411026050221|
| SARVESH HS | RA2411026050213|

---

## Live Website
[Click here to view the interactive website](#)
