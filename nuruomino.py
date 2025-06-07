# Grupo 85:
# 1109389  Hugo Oliveira Vicente
# 1110647  Vladislav Nagornii

from collections import deque
import sys
from search import Node, Problem


def painted_regions_connected(board: "Board") -> bool:
    """Return ``True`` if all painted regions remain potentially connected.

    Empty regions may act as bridges in the future. Only consider orthogonal
    adjacency between painted pieces when deciding connectivity.
    """

    painted = set(board.painted.keys())
    if len(painted) <= 1:
        return True

    coordinates_by_region = {
        region: set(board.coordinates_from_piece(*board.painted[region])) for region in painted
    }

    def regions_touch(region_a: int, region_b: int) -> bool:
        for row, col in coordinates_by_region[region_a]:
            for delta_row, delta_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (row + delta_row, col + delta_col) in coordinates_by_region[region_b]:
                    return True
        return False

    adjacency_graph = {region: set() for region in board.region_to_adjacent_regions_map}
    for region, neighbors in board.region_to_adjacent_regions_map.items():
        for neighbor in neighbors:
            if neighbor < region:
                continue
            if region in painted and neighbor in painted:
                if regions_touch(region, neighbor):
                    adjacency_graph[region].add(neighbor)
                    adjacency_graph[neighbor].add(region)
            else:
                adjacency_graph[region].add(neighbor)
                adjacency_graph[neighbor].add(region)

    start_region = next(iter(painted))
    visited = {start_region}
    queue = deque([start_region])

    while queue:
        current_region = queue.popleft()
        for next_region in adjacency_graph[current_region]:
            if next_region not in visited:
                visited.add(next_region)
                queue.append(next_region)

    return painted.issubset(visited)

class NuruominoState:
    """Lightweight wrapper holding a Board instance and unique id."""

    state_id = 0

    def __init__(self, board: "Board") -> None:
        self.board = board
        self.id = NuruominoState.state_id
        NuruominoState.state_id += 1

    def __lt__(self, other: "NuruominoState") -> bool:
        return self.id < other.id


class Board:
    """Internal representation of a *Nuruomino* puzzle board."""

    _PIECES_SET = frozenset({"L", "I", "T", "S"})

    def __init__(
        self,
        data,
        region_map=None,
        adjacent_cell_map=None,
        adjacent_region_map=None,
    ) -> None:
        self.grid = data
        self.size = len(self.grid)
        self.painted = {}
        self.values = set()

        if region_map and adjacent_cell_map and adjacent_region_map:
            self.region_to_positions_map = region_map
            self.cell_to_adjacent_positions_map = adjacent_cell_map
            self.region_to_adjacent_regions_map = adjacent_region_map
        else:
            self._precompute_region_positions()
            self._precompute_adjacency_maps()

    def _precompute_region_positions(self) -> None:
        """Pre-compute and store internal (0-based) positions for each region."""

        self.region_to_positions_map = {}
        for row_index, row_data in enumerate(self.grid):
            for col_index, region_id in enumerate(row_data):
                self.region_to_positions_map.setdefault(region_id, set()).add((row_index, col_index))

    def _precompute_adjacency_maps(self) -> None:
        """Pre-compute adjacency maps for cells and regions."""

        self.cell_to_adjacent_positions_map = {}
        self.region_to_adjacent_regions_map = {}

        diagonal_adjacencies = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
        orthogonal_adjacencies = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        all_region_ids = {region for row in self.grid for region in row}
        for region_id in all_region_ids:
            self.region_to_adjacent_regions_map[region_id] = set()
            self.region_to_positions_map.setdefault(region_id, set())

        for row_index in range(self.size):
            for col_index in range(self.size):
                current_position = (row_index, col_index)
                self.cell_to_adjacent_positions_map[current_position] = []

                for delta_row, delta_col in diagonal_adjacencies:
                    adjacent_row, adjacent_col = row_index + delta_row, col_index + delta_col
                    if 0 <= adjacent_row < self.size and 0 <= adjacent_col < self.size:
                        self.cell_to_adjacent_positions_map[current_position].append((adjacent_row, adjacent_col))

                current_region = self.grid[row_index][col_index]
                for delta_row, delta_col in orthogonal_adjacencies:
                    adjacent_row, adjacent_col = row_index + delta_row, col_index + delta_col
                    if 0 <= adjacent_row < self.size and 0 <= adjacent_col < self.size:
                        neighbor_region = self.grid[adjacent_row][adjacent_col]
                        if neighbor_region != current_region:
                            self.region_to_adjacent_regions_map[current_region].add(neighbor_region)

        # Convert region adjacency sets to lists for deterministic order
        for region_id, neighbors in self.region_to_adjacent_regions_map.items():
            self.region_to_adjacent_regions_map[region_id] = list(neighbors)

    def get_positions_from_region(self, region: int) -> set:
        """Return the coordinates belonging to *region* (internal indexing)."""
        return self.region_to_positions_map.get(region, set()).copy()

    def get_region_from_position(self, row: int, col: int) -> int:
        """Return the region id at position *(row, col)* (internal indexing)."""

        if not self.is_inside_board(row, col):
            raise ValueError("Position outside the board.")
        return self.grid[row][col]

    def get_adjacent_regions(self, region: int) -> list[int]:
        """Return regions neighboring *region* (cached)."""
        return self.region_to_adjacent_regions_map.get(region, []).copy()

    def get_adjacent_positions(self, row: int, col: int) -> list[tuple[int, int]]:
        """Return all adjacent positions (including diagonals) to *(row, col)*."""
        return self.cell_to_adjacent_positions_map.get((row, col), []).copy()

    def is_inside_board(self, row: int, col: int) -> bool:
        return 0 <= row < self.size and 0 <= col < self.size

    def get_value(self, row: int, col: int):
        """Return value at *(row, col)*.

        If painted, return piece type (``str``); otherwise return region id (``int``).
        ``0`` denotes an invalid coordinate.
        """
        if not self.is_inside_board(row, col):
            return 0
        coordinate = (row, col)
        original_region = self.grid[row][col]

        if coordinate in self.values and original_region in self.painted:
            return self.painted[original_region][0]
        return original_region

    def place_piece(
        self,
        region: int,
        piece_type: str,
        transformation,
        row: int,
        col: int,
    ) -> None:
        """Paint *region* with the given piece at (row, col) as anchor."""

        self.painted[region] = (piece_type, transformation, row, col)
        for coordinate in self.coordinates_from_piece(piece_type, transformation, row, col):
            self.values.add(coordinate)

    def coordinates_from_piece(
        self,
        piece_type: str,
        transformation,
        row: int,
        col: int,
    ) -> tuple:
        """Return tuple of coordinates occupied by *piece_type* with *transformation*."""

        result = []
        for row_offset, transformation_row in enumerate(transformation):
            for col_offset, cell_value in enumerate(transformation_row):
                if cell_value == 1:
                    result.append((row + row_offset, col + col_offset))
        return tuple(result)

    def get_adjacent_values(self, row: int, col: int) -> list:
        """Return values of all cells adjacent (incl. diagonals) to *(row, col)*."""

        return [self.get_value(r, c) for r, c in self.get_adjacent_positions(row, col)]

    def has_2x2_tetromino(self) -> bool:
        """Return ``True`` if a 2×2 block of painted cells exists."""

        for row in range(self.size - 1):
            for col in range(self.size - 1):
                value_00 = self.get_value(row, col)
                if not (isinstance(value_00, str) and value_00 in self._PIECES_SET):
                    continue
                value_01 = self.get_value(row, col + 1)
                if not (isinstance(value_01, str) and value_01 in self._PIECES_SET):
                    continue
                value_10 = self.get_value(row + 1, col)
                if not (isinstance(value_10, str) and value_10 in self._PIECES_SET):
                    continue
                value_11 = self.get_value(row + 1, col + 1)
                if isinstance(value_11, str) and value_11 in self._PIECES_SET:
                    return True
        return False

    def print(self):
        """Return board as text in the specified output format."""

        lines = []
        for row in range(self.size):
            line = []
            for col in range(self.size):
                line.append(str(self.get_value(row, col)))
            lines.append("\t".join(line))
        return "\n".join(lines) + "\n"

    @staticmethod
    def parse_instance():
        """Read instance from *stdin* and return a :class:`Board`."""

        grid_data = []
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            grid_data.append([int(number) for number in line.split()])

        if not grid_data:
            print("No data read from standard input.", file=sys.stderr)
            return None
        return Board(grid_data)


class Nuruomino(Problem):
    PIECES = ["L", "I", "T", "S"]
    _PIECES_SET_STATIC = frozenset(PIECES)

    TRANSFORMATIONS = {
        "L": tuple(
            map(
                lambda matrix: tuple(map(tuple, matrix)),
                [
                    [[1, 0], [1, 0], [1, 1]],
                    [[1, 1, 1], [1, 0, 0]],
                    [[1, 1], [0, 1], [0, 1]],
                    [[0, 0, 1], [1, 1, 1]],
                    [[0, 1], [0, 1], [1, 1]],
                    [[1, 1, 1], [0, 0, 1]],
                    [[1, 1], [1, 0], [1, 0]],
                    [[1, 0, 0], [1, 1, 1]],
                ],
            )
        ),
        "I": tuple(
            map(
                lambda matrix: tuple(map(tuple, matrix)),
                [
                    [[1], [1], [1], [1]],
                    [[1, 1, 1, 1]],
                ],
            )
        ),
        "T": tuple(
            map(
                lambda matrix: tuple(map(tuple, matrix)),
                [
                    [[1, 0], [1, 1], [1, 0]],
                    [[1, 1, 1], [0, 1, 0]],
                    [[0, 1], [1, 1], [0, 1]],
                    [[0, 1, 0], [1, 1, 1]],
                ],
            )
        ),
        "S": tuple(
            map(
                lambda matrix: tuple(map(tuple, matrix)),
                [
                    [[0, 1], [1, 1], [1, 0]],
                    [[1, 1, 0], [0, 1, 1]],
                    [[1, 0], [1, 1], [0, 1]],
                    [[0, 1, 1], [1, 1, 0]],
                ],
            )
        ),
    }

    def __init__(self, board: Board):
        initial_state = NuruominoState(board)
        super().__init__(initial_state)

        self.regions = {region_id for row in board.grid for region_id in row}
        self.sorted_regions = sorted(self.regions)
        self._precompute_all_actions()

    def _paint_size_4_regions(self):
        """Automatically paint regions containing exactly four cells."""

        for region_id in self.regions:
            region_positions = self.initial.board.get_positions_from_region(region_id)
            if len(region_positions) == 4:
                pass

    def _precompute_all_actions(self):
        """Pre-compute and organize actions per region."""

        self.all_possible_actions = set()
        self.actions_by_region = {region_id: [] for region_id in self.regions}

        for region_id in self.regions:
            if region_id in self.initial.board.painted:
                continue
            region_positions = self.initial.board.get_positions_from_region(region_id)
            if not region_positions:
                continue

            for piece_type in self.PIECES:
                for transformation_tuple in self.TRANSFORMATIONS[piece_type]:
                    relative_cells = [
                        (row_index, col_index)
                        for row_index, transformation_row in enumerate(transformation_tuple)
                        for col_index, cell_value in enumerate(transformation_row)
                        if cell_value == 1
                    ]
                    if not relative_cells:
                        continue

                    for piece_row, piece_col in relative_cells:
                        for region_row, region_col in region_positions:
                            anchor_row = region_row - piece_row
                            anchor_col = region_col - piece_col
                            if self._is_basic_placement_valid(
                                region_id,
                                piece_type,
                                transformation_tuple,
                                anchor_row,
                                anchor_col,
                            ):
                                action = (
                                    region_id,
                                    piece_type,
                                    transformation_tuple,
                                    anchor_row,
                                    anchor_col,
                                )
                                self.all_possible_actions.add(action)
                                self.actions_by_region[region_id].append(action)

    def _is_basic_placement_valid(
        self,
        region: int,
        piece_type: str,
        transformation,
        anchor_row: int,
        anchor_col: int,
    ) -> bool:
        """State-independent placement validations."""

        coordinates = self.initial.board.coordinates_from_piece(
            piece_type, transformation, anchor_row, anchor_col
        )
        if len(coordinates) != 4:
            return False

        for row_coordinate, col_coordinate in coordinates:
            if not self.initial.board.is_inside_board(row_coordinate, col_coordinate):
                return False
            try:
                current_region = self.initial.board.get_region_from_position(row_coordinate, col_coordinate)
            except ValueError:
                return False
            if current_region != region:
                return False
        return True

    def actions(self, state: NuruominoState):
        """Return actions restricted to the next MRV region."""

        next_region = self.get_next_unassigned_region_mrv(state)
        if next_region is None:
            return []

        valid_actions = []
        for action in self.all_possible_actions:
            region_id, piece_type, transformation, anchor_row, anchor_col = action
            if region_id == next_region and self._is_dynamic_placement_valid(
                state, region_id, piece_type, transformation, anchor_row, anchor_col
            ):
                valid_actions.append(action)
        return valid_actions

    def _is_dynamic_placement_valid(
        self,
        state: NuruominoState,
        region: int,
        piece_type: str,
        transformation,
        anchor_row: int,
        anchor_col: int,
    ) -> bool:
        """Placement checks depending on *state*."""

        coordinates_tuple = state.board.coordinates_from_piece(
            piece_type, transformation, anchor_row, anchor_col
        )
        coordinates_set = set(coordinates_tuple)

        # Adjacent same-type piece check (orthogonal only)
        for row_cell, col_cell in coordinates_tuple:
            for delta_row, delta_col in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                adjacent_row, adjacent_col = row_cell + delta_row, col_cell + delta_col
                if state.board.is_inside_board(adjacent_row, adjacent_col) and (adjacent_row, adjacent_col) not in coordinates_set:
                    neighbor = state.board.get_value(adjacent_row, adjacent_col)
                    if isinstance(neighbor, str) and neighbor == piece_type:
                        return False

        # 2×2 tetromino check
        if self._would_create_2x2_tetromino(state, coordinates_set, piece_type):
            return False

        temporary_board = self._copy_board(state.board)
        temporary_board.place_piece(region, piece_type, transformation, anchor_row, anchor_col)
        if not painted_regions_connected(temporary_board):
            return False
        return True

    def _would_create_2x2_tetromino(
        self,
        state: NuruominoState,
        new_coordinates: set,
        new_piece_type: str,
    ) -> bool:
        """Check if placing the piece would create a solid 2×2 block."""

        board_size = state.board.size
        squares_to_check = set()
        for row, col in new_coordinates:
            for delta_row_square, delta_col_square in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
                start_row, start_col = row + delta_row_square, col + delta_col_square
                if 0 <= start_row <= board_size - 2 and 0 <= start_col <= board_size - 2:
                    squares_to_check.add((start_row, start_col))

        for start_row, start_col in squares_to_check:
            if all(
                isinstance(
                    (
                        new_piece_type
                        if (position_row := start_row + delta_row, position_col := start_col + delta_col) in new_coordinates
                        else state.board.get_value(position_row, position_col)
                    ),
                    str,
                )
                and (
                    (
                        new_piece_type
                        if (position_row, position_col) in new_coordinates
                        else state.board.get_value(position_row, position_col)
                    )
                    in self._PIECES_SET_STATIC
                )
                for delta_row, delta_col in ((0, 0), (0, 1), (1, 0), (1, 1))
            ):
                return True
        return False

    @staticmethod
    def _copy_board(board: Board) -> Board:
        """Return a shallow copy of *board* (shared geometry maps)."""

        new_board = Board(
            board.grid,
            region_map=board.region_to_positions_map,
            adjacent_cell_map=board.cell_to_adjacent_positions_map,
            adjacent_region_map=board.region_to_adjacent_regions_map,
        )
        new_board.painted = board.painted.copy()
        new_board.values = board.values.copy()
        return new_board

    def result(self, state: NuruominoState, action): 
        """Return state resulting from *action* (with forward checking)."""

        new_board = self._copy_board(state.board)
        new_state = NuruominoState(new_board)
        region, piece_type, transformation, row, col = action
        new_state.board.place_piece(region, piece_type, transformation, row, col)
        self._apply_forward_checking(new_state, region, piece_type)
        return new_state

    def _apply_forward_checking(self, state, placed_region, placed_piece_type):
        """Prune inconsistent actions after placing a piece."""

        for adjacent_region in state.board.get_adjacent_regions(placed_region):
            if adjacent_region in state.board.painted:
                continue
            valid_actions = []
            for action in self.all_possible_actions:
                if action[0] != adjacent_region or action[1] == placed_piece_type:
                    continue
                if self._is_dynamic_placement_valid(state, *action[:5]):
                    valid_actions.append(action)
            self.actions_by_region[adjacent_region] = valid_actions

    def goal_test(self, state: NuruominoState): 
        """Return ``True`` if *state* is a goal."""

        if any(region not in state.board.painted for region in self.regions):
            return False
        if state.board.has_2x2_tetromino():
            return False

        for region_id, (piece_type, transformation, anchor_row, anchor_col) in state.board.painted.items():
            piece_coordinates = state.board.coordinates_from_piece(piece_type, transformation, anchor_row, anchor_col)
            piece_set = set(piece_coordinates)
            for row_cell, col_cell in piece_coordinates:
                for delta_row, delta_col in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    adjacent_row, adjacent_col = row_cell + delta_row, col_cell + delta_col
                    if state.board.is_inside_board(adjacent_row, adjacent_col) and (adjacent_row, adjacent_col) not in piece_set:
                        neighbor = state.board.get_value(adjacent_row, adjacent_col)
                        if isinstance(neighbor, str) and neighbor == piece_type:
                            return False

        return self._is_single_polyomino(state)

    def _is_single_polyomino(self, state: NuruominoState):
        """Return ``True`` if painted cells form a single polyomino."""

        painted_cells = state.board.values
        if not painted_cells:
            return not self.regions
        visited = set()
        queue = [next(iter(painted_cells))]
        visited.add(queue[0])

        while queue:
            row, col = queue.pop(0)
            for delta_row, delta_col in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                adjacent_position = (row + delta_row, col + delta_col)
                if adjacent_position in painted_cells and adjacent_position not in visited:
                    visited.add(adjacent_position)
                    queue.append(adjacent_position)
        return len(visited) == len(painted_cells)

    def get_next_unassigned_region_mrv(self, state):
        """Return unpainted region with minimum remaining values (MRV)."""

        minimum_actions = float("inf")
        best_region = None
        candidate_regions = []

        for region_id in self.sorted_regions:
            if region_id in state.board.painted:
                continue
            valid_actions = [
                action
                for action in self.all_possible_actions
                if action[0] == region_id
                and self._is_dynamic_placement_valid(state, *action[:5])
            ]
            action_count = len(valid_actions)
            if action_count < minimum_actions:
                minimum_actions = action_count
                best_region = region_id
                candidate_regions = [(region_id, action_count)]
            elif action_count == minimum_actions:
                candidate_regions.append((region_id, action_count))

        if len(candidate_regions) > 1:
            return self._tiebreak_by_degree(state, candidate_regions)
        return best_region

    def _tiebreak_by_degree(self, state, candidate_regions):
        """Tie-break using degree heuristic (most unpainted neighbors)."""

        maximum_degree = -1
        best_region = None
        for region_id, _ in candidate_regions:
            unpainted_neighbor_count = sum(
                1
                for adjacent_region in self.initial.board.get_adjacent_regions(region_id)
                if adjacent_region not in state.board.painted
            )
            if unpainted_neighbor_count > maximum_degree:
                maximum_degree = unpainted_neighbor_count
                best_region = region_id
        return best_region

def check_conflicts(state, region, action):
    """Check for conflicts with placement.

    Return ``(has_conflict: bool, conflict_regions: set)``.
    """

    region_id, piece_type, transformation, anchor_row, anchor_col = action
    coordinates = state.board.coordinates_from_piece(piece_type, transformation, anchor_row, anchor_col)
    coordinates_set = set(coordinates)
    conflict_regions = set()
    board = state.board

    # Neighboring same-type pieces
    for row_cell, col_cell in coordinates:
        for adjacent_row, adjacent_col in (
            (row_cell, col_cell + 1),
            (row_cell, col_cell - 1),
            (row_cell + 1, col_cell),
            (row_cell - 1, col_cell),
        ):
            if board.is_inside_board(adjacent_row, adjacent_col) and (adjacent_row, adjacent_col) not in coordinates_set:
                neighbor_value = board.get_value(adjacent_row, adjacent_col)
                if neighbor_value == piece_type:
                    try:
                        conflict_regions.add(board.get_region_from_position(adjacent_row, adjacent_col))
                    except ValueError:
                        pass

    # 2×2 tetromino conflicts
    if not conflict_regions:
        board_size = board.size
        squares_to_check = set()
        for row, col in coordinates:
            for start_row, start_col in (
                (row, col),
                (row, col - 1),
                (row - 1, col),
                (row - 1, col - 1),
            ):
                if 0 <= start_row <= board_size - 2 and 0 <= start_col <= board_size - 2:
                    squares_to_check.add((start_row, start_col))

        for start_row, start_col in squares_to_check:
            square_regions = set()
            all_tetrominos = True
            for position_row, position_col in (
                (start_row, start_col),
                (start_row, start_col + 1),
                (start_row + 1, start_col),
                (start_row + 1, start_col + 1),
            ):
                if (position_row, position_col) in coordinates_set:
                    square_regions.add(region_id)
                else:
                    current_value = board.get_value(position_row, position_col)
                    if current_value in Nuruomino._PIECES_SET_STATIC:
                        try:
                            square_regions.add(board.get_region_from_position(position_row, position_col))
                        except ValueError:
                            all_tetrominos = False
                            break
                    else:
                        all_tetrominos = False
                        break
            if all_tetrominos and len(square_regions) > 1:
                conflict_regions.update(square_regions)
                conflict_regions.discard(region_id)
                break

    return bool(conflict_regions), conflict_regions


def conflict_stack_backjumping_search(problem):
    """Back-jumping search with MRV + Forward Checking."""

    initial_state = problem.initial
    conflict_sets = {region: set() for region in problem.regions}

    def depth_first_search(state, assignment_order):
        if problem.goal_test(state):
            return Node(state)
        next_region = problem.get_next_unassigned_region_mrv(state)
        if next_region is None:
            return None
        for action in problem.actions(state):
            new_state = problem.result(state, action)
            assignment_order.append(next_region)
            result = depth_first_search(new_state, assignment_order)
            if result is not None:
                return result
            assignment_order.pop()
        return None

    return depth_first_search(initial_state, [])

board = Board.parse_instance()
problem = Nuruomino(board)
solution_node = conflict_stack_backjumping_search(problem)
print(solution_node.state.board.print(), end="")
