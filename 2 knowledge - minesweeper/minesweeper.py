import itertools
import random

# https://elo.beta4all.nl/mod/assign/view.php?id=4599
# considering their knowledge base, and making inferences based on that knowledge


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        if len(self.cells) == self.count:
            return self.cells
        return None

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        if self.count == 0:
            return self.cells
        return None

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        sentence.mark_mine(cell)
        """
        newCells = set()
        for item in self.cells:
            if item != cell:
                newCells.add(item)
            else:
                self.count -= 1
        self.cells = newCells

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        newCells = set()
        for item in self.cells:
            if item != cell:
                newCells.add(item)
        self.cells = newCells


class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)


    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """
        if (cell in self.moves_made):
            return

        print("============================")
        print(f"check cell {cell} has count {count}")
        # 1) mark the cell as a move that has been made
        self.moves_made.add(cell)

        # 2) mark the cell as safe
        self.mark_safe(cell)

        # 3) add a new sentence to the AI's knowledge base
        #    based on the value of `cell` and `count`
        neighbors, count = self.get_neighbors(cell, count)

        # 5) add any new sentences to the AI's knowledge base
        #    if they can be inferred from existing knowledge
        sentence = Sentence(neighbors, count)
        self.knowledge.append(sentence)

        # see if we can infer new knowledge
        # any time we have two sentences set1 = count1 and set2 = count2
        # where set1 is a subset of set2,
        # then we can construct the new sentence set2 - set1 = count2 - count1
        new_inferences = []

        if not(len(sentence.cells)):
            return
        print("===============")
        print(f"s: cells {sorted(sentence.cells)}")
        print(f"s: count {sentence.count}")

        for k in self.knowledge:
            print("----------------")
            print(f"knowledge cells: {sorted(k.cells)}")
            print(f"knowledge count: {k.count}")

            if sentence.cells.issubset(k.cells):
                sDiff = k.cells - sentence.cells
                print(f"knowledge diff (sk): {sDiff}")

                # if total count of the knowledge is equal to sentence
                # then the remaining cells are safe
                if k.count == sentence.count:
                    for cellsafe in sDiff:
                        self.mark_safe(cellsafe)
                        print(f"safe: {cellsafe}")

                # if total number of remaining cells is equal to
                # the knowledge count minus the sentence count
                # then the remaining cells are mines
                elif len(sDiff) == k.count - sentence.count:
                    for cellmine in sDiff:
                        self.mark_mine(cellmine)
                        print(f"mine: {cellmine}")
                # Known inference
                else:
                    new_inferences.append(
                        Sentence(sDiff, k.count - sentence.count)
                    )

            elif k.cells.issubset(sentence.cells):
                # the reverse of the other subset if statement
                sDiff = sentence.cells - k.cells
                print(f"knowledge diff (ks): {sDiff}")
                print(f"Sentence.count - k.count diff (ks): {sentence.count - k.count}")

                if sentence.count == k.count:
                    for cellsafe in sDiff:
                        self.mark_safe(cellsafe)
                        print(f"safe: {cellsafe}")

                elif len(sDiff) == sentence.count - k.count:
                    for cellmine in sDiff:
                        self.mark_mine(cellmine)
                        print(f"mine: {cellmine}")

                # Known inference
                else:
                    new_inferences.append(
                        Sentence(sDiff, sentence.count - k.count)
                    )

        # EXTEND new inferences to knowledge
        # append would add an object to it
        if len(new_inferences):
            self.knowledge.extend(new_inferences)

        # make knowledge unique
        u = []
        for s in self.knowledge:
            if s not in u:
                u.append(s)
        self.knowledge = u
        print("===============")


    def get_neighbors(self, cell, count):
        """
        get the neighboring cells for the cell we want to check
        """
        # print(f"get_neighbors: cell: {cell} count {count}")
        i = cell[0]
        j = cell[1]
        r = [i - 1, i, i + 1]
        c = [j - 1, j, j + 1]
        neighbors = []
        list_safes = []
        for checked_row in r:
            if (checked_row < 0 or checked_row >= self.height):
                # print(f"row out of bounds {checked_row} > rows: 0 - {self.height}")
                continue

            for checked_col in c:
                # if statements to make sure we stay inside the bounds
                # and to skip the cell that originated the function
                # or if we know the cell we are checking is in the list of safe cells
                if (checked_col < 0 or checked_col >= self.width) \
                or (checked_row == i and checked_col == j) \
                or (checked_row, checked_col) in self.safes \
                or (checked_row, checked_col) in self.moves_made:
                    continue
                elif (checked_row, checked_col) in self.mines:
                    # found a known mine, lower the count
                    count -= 1
                    continue

                if count == 0  \
                and (checked_row, checked_col) not in self.mines:
                    # print(f"get_neighbors: cell: {cell}: r{checked_row},c{checked_col} count 0 so automatically marked safe")
                    self.mark_safe((checked_row, checked_col))
                    list_safes.append((checked_row, checked_col))
                    continue

                neighbors.append((checked_row, checked_col))


        print(f"added to safe cells: {len(list_safes)} {sorted(list_safes)}")
        print(f"safes {len(self.safes - self.moves_made)} : {sorted(self.safes - self.moves_made)}")
        print(f"mines {len(self.mines - self.moves_made)} : {sorted(self.mines - self.moves_made)}")
        return neighbors, count

    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        # subtract the moves already made
        # get the cells that are marked as safe and "click" one of them
        safes = self.safes
        sCells = self.safes - self.moves_made
        # print(f"safe moves left: {len(sCells)} : {sCells}")

        if len(sCells) == 0:
            # if we return None, the runner will run make_random_move()
            return None

        safecell = sCells.pop()
        print(f"making safe move: {safecell}")

        return safecell

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        allmoves = set()
        for i in range(self.height):
            # if (i,j) not in self.mines and (i,j) not in self.moves_made:
            #     # only pick from non-mines and not already clicked cell_size
            for j in range(self.width):
                if (i,j) not in self.mines and (i,j) not in self.moves_made:
                    allmoves.add((i,j))

        # it is possible we have no (more) moves, so account for that here
        if len(allmoves) == 0:
            return None

        randchoice = random.choice(tuple(allmoves))
        print(f"making random move: {randchoice}")
        # there is a possible move, so pick one at random
        # tuple() is needed to avoid 'set' object is not subscriptable
        return randchoice

if __name__ == "__main__":
    cells = {(1,2), (4, 4)}
    sentence = Sentence(cells, len(cells))
    assert sentence.known_mines() == cells, "known_mines werkt niet naar behoren"
    # assert sentence.known_safes() == cells, "known_safes werkt niet naar behoren"
