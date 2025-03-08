import pdb
from collections import deque
from typing import Deque, Dict, List, Set, Tuple

import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

NUM_PIECES = 8


class Piece:
    def __init__(self, array: NDArray[np.float64]):
        """
        array (np): Array of object
        index (tuple[int, int])
        """
        self.array = array
        self.index = (0, 0)
        self.rotation = 0
        self.flip = False

    def set_index(self, index: tuple):
        self.index = index

    def get_cur_index_internal(
        self, cur_index_rel: tuple, array: NDArray[np.float64], index: tuple
    ):
        """Internal use as cur_index is different for different size arrays"""
        # Rotates counter-clockwise
        a, b = array.shape
        if self.rotation == 0 and not self.flip or self.rotation == 3 and self.flip:
            tl = (0, 0)
        elif self.rotation == 1 and not self.flip or self.rotation == 2 and self.flip:
            tl = (a - 1, 0)
        elif self.rotation == 2 and not self.flip or self.rotation == 1 and self.flip:
            tl = (a - 1, b - 1)
        else:
            tl = (0, b - 1)

        # Equivalent to setting top-left to 0, and then adding relative distance to it
        # (cur_index_rel is the distance from top-left)
        cur_index = (
            index[0] + cur_index_rel[0] - tl[0],
            index[1] + cur_index_rel[1] - tl[1],
        )
        return cur_index

    def get_cur_index(self, cur_index_rel: tuple):
        """
        cur_index_rel (tuple) is the index we are trying to find relative to top left index before rotation
        cur_index_rel is the number associated with the index in self.array
        We are trying to get the index of the puzzle
        """
        return self.get_cur_index_internal(cur_index_rel, self.array, self.index)

    def get_subarray(self, board: NDArray[np.float64]) -> NDArray[np.float64]:
        array_shape = self.array.shape
        top_side = self.index[0] - (array_shape[0] - 1)
        left_side = self.index[1] - (array_shape[1] - 1)
        bottom_side = self.index[0] + array_shape[0]
        right_side = self.index[1] + array_shape[1]
        # Get it to match top-left point before rotation
        if self.rotation == 0 and not self.flip or self.rotation == 3 and self.flip:
            subarray = board[self.index[0] : bottom_side, self.index[1] : right_side]
        elif self.rotation == 1 and not self.flip or self.rotation == 2 and self.flip:
            subarray = board[top_side : self.index[0] + 1, self.index[1] : right_side]
        elif self.rotation == 2 and not self.flip or self.rotation == 1 and self.flip:
            subarray = board[
                top_side : self.index[0] + 1, left_side : self.index[1] + 1
            ]
        else:
            subarray = board[self.index[0] : bottom_side, left_side : self.index[1] + 1]
        return subarray

    def get_mega_array(
        self, board: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], tuple]:
        """
        Subarry plus surrounding values
        Return new subarray plus index of new top-left piece rotated
        """
        array_shape = self.array.shape
        board_shape = board.shape

        top_side = self.index[0] - (array_shape[0] - 1)
        left_side = self.index[1] - (array_shape[1] - 1)
        bottom_side = self.index[0] + array_shape[0]
        right_side = self.index[1] + array_shape[1]

        if top_side > 0:
            top_side = self.index[0] - (array_shape[0] - 1) - 1
        if left_side > 0:
            left_side = self.index[1] - (array_shape[1] - 1) - 1
        # Since it is exclusive on the right hand side
        if bottom_side < board_shape[0] + 1:
            bottom_side = self.index[0] + array_shape[0] + 1
        if right_side < board_shape[1] + 1:
            right_side = self.index[1] + array_shape[1] + 1

        # Get it to match top-left point before rotation
        if self.rotation == 0 and not self.flip or self.rotation == 3 and self.flip:
            if self.index[0] > 0:
                top_side = self.index[0] - 1
            else:
                top_side = self.index[0]
            if self.index[1] > 0:
                left_side = self.index[1] - 1
            else:
                left_side = self.index[1]
            subarray = board[top_side:bottom_side, left_side:right_side]
            top_left_now = (top_side, left_side)
        elif self.rotation == 1 and not self.flip or self.rotation == 2 and self.flip:
            if self.index[0] + 1 < board_shape[0]:
                bottom_side = self.index[0] + 2
            else:
                bottom_side = self.index[0] + 1
            if self.index[1] > 0:
                left_side = self.index[1] - 1
            else:
                left_side = self.index[1]
            subarray = board[top_side:bottom_side, left_side:right_side]
            # top_left_now is inclusive
            top_left_now = (bottom_side - 1, left_side)
        elif self.rotation == 2 and not self.flip or self.rotation == 1 and self.flip:
            if self.index[0] + 1 < board_shape[0]:
                bottom_side = self.index[0] + 2
            else:
                bottom_side = self.index[0] + 1
            if self.index[1] + 1 < board_shape[1]:
                right_side = self.index[1] + 2
            else:
                right_side = self.index[1] + 1

            subarray = board[top_side:bottom_side, left_side:right_side]
            top_left_now = (bottom_side - 1, right_side - 1)
        else:
            if self.index[0] > 0:
                top_side = self.index[0] - 1
            else:
                top_side = self.index[0]
            if self.index[1] + 1 < board_shape[1]:
                right_side = self.index[1] + 2
            else:
                right_side = self.index[1] + 1
            subarray = board[top_side:bottom_side, left_side:bottom_side]
            top_left_now = (top_side, right_side - 1)

        return subarray, top_left_now

    def place(self, board: NDArray[np.float64], indexes: Set) -> bool:
        """
        Attempt to place piece into the puzzle, based off top-left index
        If the puzzle will not work with this placement return false
        board (np)
        Return if it was successful or not
        """
        if not self.inBoard(board):
            return False

        # We can now assume all points won't be out of bounds
        # Assume this always represents the top-left before rotation
        for (i, j), value in np.ndenumerate(self.array):
            # cur_index = (i + self.index[0], j + self.index[1])
            cur_index = self.get_cur_index((i, j))
            if value == 0:
                continue
            # np.nan will return false for the first one
            if board[cur_index] > 0 or np.isnan(board[cur_index]):
                return False

        subarray = self.get_subarray(board)
        subarray += self.array

        # Find the amount of 0 in a group (accessible) remove group from hash table, and keep going stop if group is too small for smallest piece
        # subarray plus 1 in each direction
        # Check if solution is not possilbe
        mega_array, top_left_new = self.get_mega_array(board)
        # Adds the index in board of zeros in the mega_array
        zero_set = set()
        for (i, j), value in np.ndenumerate(mega_array):
            if value == 0:
                zero_set.add(
                    self.get_cur_index_internal((i, j), mega_array, top_left_new)
                )

        while len(zero_set):
            point = zero_set.pop()
            result = self.zero_recursion(board, zero_set, point)
            if result == False:
                self.remove(board, indexes)
                return False

        # Current top-left index after rotation
        # Base index at (0,0)
        tlr_index = self.get_cur_index((0, 0))
        for index, _ in np.ndenumerate(subarray):
            # If we didn't change it
            if self.array[index] == 0:
                continue
            print(board)
            print(subarray)
            indexes.remove((index[0] + tlr_index[0], index[1] + tlr_index[1]))

        return True

    def zero_recursion_helper(
        self, board: NDArray[np.float64], zero_set: Set, point: tuple, size: Set
    ):
        """Used to check if there are any empty values near by"""
        board_shape = board.shape
        if board[point] == 0 and len(size) < 5 and point not in size:
            size.add(point)
            left_point = (point[0] - 1, point[1])
            right_point = (point[0] + 1, point[1])
            up_point = (point[0], point[1] - 1)
            down_point = (point[0], point[1] + 1)
            if left_point[0] >= 0:
                self.zero_recursion_helper(board, zero_set, left_point, size)
            if right_point[0] < board_shape[0]:
                self.zero_recursion_helper(board, zero_set, right_point, size)
            if up_point[1] >= 0:
                self.zero_recursion_helper(board, zero_set, up_point, size)
            if down_point[1] < board_shape[1]:
                self.zero_recursion_helper(board, zero_set, down_point, size)
        elif len(size) >= 5 and point in zero_set:
            zero_set.remove(point)

    def zero_recursion(
        self, board: NDArray[np.float64], zero_set: Set, point: tuple
    ) -> bool:
        """
        Check the continuous 0s near insertions
        Return whether a solution with current board is possilbe
        Point is a point in board
        """
        size = set()
        self.zero_recursion_helper(board, zero_set, point, size)
        if len(size) >= 5:
            return True
        else:
            return False

    def remove(self, board: NDArray[np.float64], indexes: Set):
        """
        Remove piece from the puzzle, based off top-left index
        Must call place before remove
        board (np)
        """
        subarray = self.get_subarray(board)
        subarray -= self.array
        tlr_index = self.get_cur_index((0, 0))
        for index, _ in np.ndenumerate(subarray):
            if self.array[index] == 0:
                continue
            indexes.add((index[0] + tlr_index[0], index[1] + tlr_index[1]))

    def rotate(self):
        """Must remove before rotating"""
        self.array = np.rot90(self.array)
        # Can be 0, 1, 2, 3
        self.rotation = (self.rotation + 1) % 4

    def do_flip(self):
        self.array = np.fliplr(self.array)
        self.flip = not self.flip

    def inBoard(self, board: NDArray[np.float64]) -> bool:
        """
        Return true if the piece would remain on the board
        If top-left of board or bottom-right of the board is out-of-bounds returns false
        """
        # cur_index is the index at (0,0)
        cur_index = self.get_cur_index((0, 0))
        if cur_index[0] < 0 or cur_index[1] < 0:
            return False
        board_shape = board.shape
        piece_shape = self.array.shape
        # Since we are checking the top-left index it is only possible to be
        # out of bounds if it is greater than max index, nan won't matter for this
        if (
            board_shape[0] - 1 < cur_index[0] + piece_shape[0] - 1
            or board_shape[1] - 1 < cur_index[1] + piece_shape[1] - 1
        ):
            return False
        return True


# Create board
board = np.zeros((7, 7), dtype="float")
board[6, 3:] = np.nan
board[0:2, 6] = np.nan


num_to_month = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}

while True:
    try:
        user_input = input("Input month/day ex. 3/3: \n")
        month_str, day_str = user_input.split("/")
        month_int = int(month_str)
        day_int = int(day_str)
        day = str(day_int)
        month = num_to_month[month_int]
        month_subset = board[0:3, 0:6]
        # Original array is not continuous in memory so ravel returns a copy
        month_loco = np.unravel_index(month_int - 1, month_subset.shape)
        month_subset[month_loco] = np.nan
        day_subset = board[2:7, 0:7]
        day_subset_flat = day_subset.ravel()
        day_subset_flat[day_int - 1] = np.nan
        day_loco_unravel = np.unravel_index(day_int - 1, day_subset.shape)
        day_loco = (day_loco_unravel[0] + 2, day_loco_unravel[1])

        break
    except ValueError:
        print("Please enter a valid day: \n")
info = {month: month_loco, day: day_loco}

# Every piece has a value on the top left
pieces: List[Piece] = []
pieces.append(Piece(np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype="float")))
pieces.append(Piece(2 * np.array([[1, 1, 1], [1, 1, 1]], dtype="float")))
pieces.append(Piece(3 * np.array([[1, 1, 1], [0, 0, 1], [0, 0, 1]], dtype="float")))
pieces.append(Piece(4 * np.array([[1, 1, 1], [1, 1, 0]], dtype="float")))
pieces.append(Piece(5 * np.array([[1, 1, 1, 1], [1, 0, 0, 0]], dtype="float")))
pieces.append(Piece(6 * np.array([[1, 1, 1], [1, 0, 1]], dtype="float")))
pieces.append(Piece(7 * np.array([[1, 1, 1, 0], [0, 0, 1, 1]], dtype="float")))
pieces.append(Piece(8 * np.array([[1, 1, 1, 1], [0, 0, 1, 0]], dtype="float")))


def piece_recursion(pieces: List[Piece], indexes: Set, i: int = 0, done=[False]):
    """Recursively add and remmove pieces from the board, until solution is found"""
    piece = pieces[i]
    cur_indexes = indexes.copy()
    for index in cur_indexes:
        # It was already removed
        assert index in indexes
        piece.set_index(index)
        for _ in range(4):
            piece.rotate()
            for _ in range(2):
                piece.do_flip()
                if not piece.place(board, indexes):
                    continue
                # This means that it works
                if i == NUM_PIECES - 1:
                    done[0] = True
                    # print("Success")
                    return
                if i < NUM_PIECES - 1:
                    piece_recursion(pieces, indexes, i + 1, done)
                    if done[0]:
                        return
                # This means that it failed
                piece.remove(board, indexes)


# Run algorithm
# tuple -> bool, if true then it is available
# Indexes holds all the indicies if they are left, while
indexes = set()


for (i, j), value in np.ndenumerate(board):
    if np.isnan(value):
        continue
    indexes.add((i, j))

piece_recursion(pieces, indexes)

fig, ax = plt.subplots()

print(board)

linewidth = 4
shrink_factor = linewidth / 30  # Adjust to shrink the rectangle size slightly

for (i, j), value in np.ndenumerate(board):
    if np.isnan(value):
        color = "black"
    elif value == 1:
        color = "red"
    elif value == 2:
        color = "blue"
    elif value == 3:
        color = "#849CE4"
    elif value == 4:
        color = "#85A296"
    elif value == 5:
        color = "#B47C61"
    elif value == 6:
        color = "#D2DDE3"
    elif value == 7:
        color = "#B1060F"
    elif value == 8:
        color = "#C0D684"
    else:
        continue
    bottom_left = (j - 0.5 + shrink_factor / 2, i - 0.5 + shrink_factor / 2)
    if np.isnan(value):
        rect = patches.Rectangle(
            bottom_left,
            1 - shrink_factor,
            1 - shrink_factor,
            linewidth=linewidth,
            edgecolor=color,
            facecolor="black",
        )
    else:
        rect = patches.Rectangle(
            bottom_left,
            1 - shrink_factor,
            1 - shrink_factor,
            linewidth=linewidth,
            edgecolor=color,
            facecolor="tan",
        )
    ax.add_patch(rect)
    if info[month] == (i, j) or info[day] == (i, j):
        if info[month] == (i, j):
            text = month
        else:
            text = day
        ax.text(
            bottom_left[0] + (1 - shrink_factor) / 2,
            bottom_left[1] + (1 - shrink_factor) / 2,
            text,
            fontsize=12,
            ha="center",
            va="center",
            color="white",
        )

ax.set_xlim(-0.5, board.shape[1] - 0.5)
ax.set_ylim(-0.5, board.shape[0] - 0.5)
ax.set_aspect("equal", adjustable="box")
plt.gca().invert_yaxis()

plt.title("Puzzle")
# plt.axis("off")  # Hide the axis
plt.show()
