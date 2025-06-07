# AI-Nuruomino-Solver

An AI-powered solver for the **Nuruomino** puzzle. This system uses a custom-designed **Backjumping Depth-First Search (DFS)** algorithm enhanced with **constraint propagation** and an **adjacency graph** to efficiently prune invalid paths and reduce redundant computation. It is optimized for high-performance solving of complex puzzle instances.

---

## Features

- ✅ Solves Nuruomino puzzles with arbitrary grid configurations  
- 🧠 Employs Backjumping DFS to jump back to relevant decision points on constraint violations  
- ⚡ Fast and memory-efficient through intelligent pruning strategies  
- 🧩 Supports standard tetromino shapes: L, I, T, S  
- 📐 Checks adjacency, connectivity, and 2×2 invalid formations  
- 🛠️ Modular design for ease of experimentation and extension  

---

## How It Works

The solver models the puzzle as a **Constraint Satisfaction Problem (CSP)** and operates as follows:

1. **Region-based decomposition** of the board.
2. **Precomputation** of valid tetromino placements per region.
3. **MRV (Minimum Remaining Values)** heuristic to select the next region.
4. **Dynamic validation** of each move:
   - Enforces connectivity of painted regions.
   - Avoids 2x2 tetromino blocks.
   - Prevents adjacent same-type pieces.
5. **Backjumping** to skip irrelevant branches upon constraint failures.

---

## Project Structure

```
.
├── nuruomino.py   # Core puzzle logic and constraint handling
├── search.py      # Generic search framework (Node, Problem, DFS, etc.)
└── utils.py       # Helper functions and algorithms
```

---

## Usage

```bash
python nuruomino.py < input.txt
```

Where `input.txt` is a space-separated grid representing region IDs.

Example:
```
1 1 2 2
1 1 2 2
3 3 4 4
3 3 4 4
```

---

## Dependencies

- Python 3.8+
- Standard library only (no external packages required)

---

## Authors

- Hugo Oliveira Vicente (1109389)  
- Vladislav Nagornii (1110647)  

---
