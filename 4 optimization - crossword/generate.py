import sys

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        # v is an object created from the words file
        # it has starting row,starting col, direction, length
        for v in self.crossword.variables:
            # make a copy so we can remove values in the domain we are checking
            domains = self.domains[v].copy()
            for x in domains:
                if len(x) != v.length:
                    self.domains[v].remove(x)


    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        # arc consistency: when all values in a variable's domain satisfy its constraints
        
        # get the overlap between the two domains using self.crossword.overlaps
        overlap = self.crossword.overlaps[x,y]
        if overlap == None:
            return False
        
        # loop over values in self.domains[x]
        # remove each found value if there is no corresponding value for y in self.domains[y]
        
        # make a copy so we can remove values in the domain we are checking
        domains = self.domains[x].copy()
        
        # use a variable to keep track if there has been a revision
        # if we used a return, it would stop after the first vx
        revised = False
        
        for vx in domains:
            # if we don't find the vx in vy, we delete
            delete = True
            for vy in self.domains[y]:
                if vx[overlap[0]] == vy[overlap[1]]:
                    delete = False
            if delete:
                self.domains[x].remove(vx)
                # if we do a return here, we would miss subsequent words, so use a variable
                revised = True
            
        return revised


    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        # pseudo function AC-3(csp):
        # 1. queue = all arcs in csp
        if arcs == None:
            # arcs is None, so start with all the arcs in the csp
            queue = []
            for v in self.crossword.variables:
                for neighbor in self.crossword.neighbors(v):
                    queue.append((v, neighbor))
            # make a copy so we can add/remove values in the for-loop
            arcs = queue.copy()
        else:
            # otherwise use the supplied arcs as the queue
            queue = arcs.copy()
            
        # 2. while queue is not empty:
        for arc in queue:
            
            # each arc is a tuple: (x, y)
            x = arc[0]
            y = arc[1]
            
            # 2.a (x,y) = dequeue(queue)
            arcs.remove(arc)
            
            # 2.b if revise(csp,x,y):
            if self.revise(x, y):
                
                # 2.b.1 if size of x.domain == 0:
                # this means that there is no solution possible, so return False
                if len(self.domains[x]) == 0:
                    # 2.b.1 return false
                    return False
                
                # 2.b.2 for each Z in x.neighbors - {y}:
                for neighbor in self.crossword.neighbors(x):
                    
                    # 2.b.2.a enqueue(queue,(z,x))
                    arcs.append((neighbor, x))
                    
        # 3. return true
        return True


    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        # check if the word (v) is in the keys of dictionary assignment
        for v in self.crossword.variables:
            # we have to use "not in" because otherwise it would stop
            # after we find one v in keys
            # and we only want to return False if it can't find one
            if v not in assignment.keys():
                return False
        return True


    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        # if any if statement below is met, return False
        # if none are met, return True
        
        # are all values unique? can't re-use words!
        if len(assignment) != len(set(assignment.values())):
            return False

        # check constraints for each assigned key and its value
        for key, value in assignment.items():
            
            # check unary constraints (in itself, in this code here, just the length of the word)
            if len(value) != key.length:
                return False
            
            # check binary constraints (with its neighbor)
            for neighbor in self.crossword.neighbors(key):
                
                if neighbor in assignment.keys():
                    overlap = self.crossword.overlaps[key, neighbor]
                    if value[overlap[0]] != assignment[neighbor][overlap[1]]:
                        return False

        return True


    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        # first: make a dict with each value (val) set to 0
        valdict = {val: 0 for val in self.domains[var]}

        # make list of neighbors that are not already assigned
        neighbors = [n for n in self.crossword.neighbors(var) if n not in assignment.keys()]
        
        for val in self.domains[var]:
            for neighbor in neighbors:            
                # check overlap 
                overlap = self.crossword.overlaps[var, neighbor]
                
                # check if val rules out the val of the neighbor
                for word in self.domains[neighbor]:
                    if val[overlap[0]] != word[overlap[1]]:
                        # it does, so change the dict for this value
                        valdict[val] += 1
                    
        # return a list of all of the values in the domain of var,
        # ordered according to the least-constraining values heuristic.
        # print("sorted\n",sorted([v for v in valdict], key = lambda v: valdict[v]))
        # print(valdict)
        return sorted([v for v in valdict], key = lambda v: valdict[v])

    
    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        
        # unassigned variables are keys of the domains minus the assigned keys
        unassigned = set(self.domains.keys()) - set(assignment.keys())

        # list of variables
        unlisted = [var for var in unassigned]
        # sort variables ascending by minimum remaining values (mrv) and then by highest degree
        # mrv is len(self.domains[x])
        # highest degree is the number of connections (neighbors) (we want to ascend, so make them neg)
        unlisted.sort(key = lambda v: (len(self.domains[v]), -len(self.crossword.neighbors(v))))
        
        # use [0] because we only need to return the "best" unassigned variable 
        return unlisted[0]


    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.
        `assignment` is a mapping from variables (keys) to words (values).
        If no assignment is possible, return None.
        """
        
        # is the assignment is complete?
        if len(assignment) == len(self.crossword.variables):
            return assignment
        
        # not complete, so try a new variable (least mrv and highest degree)
        var = self.select_unassigned_variable(assignment)
        
        # check for each val in domains for the selected unassigned variable
        for val in self.domains[var]:
            new_assignment = assignment.copy()
            new_assignment[var] = val
            
            # interleave search with consistency check
            if self.consistent(new_assignment):
                # using recursion
                result = self.backtrack(new_assignment)
                if result is not None:
                    return result

        return None



def main():
    test = True
    
    if test:
        complexity = 2
        from datetime import timezone
        import datetime
        
        dt = datetime.datetime.now(timezone.utc)  
        utc_time = dt.replace(tzinfo=timezone.utc)
        utc_timestamp = utc_time.timestamp()
        
        structure = ".//data//structure" + str(complexity) + ".txt"
        words = ".//data//words" + str(complexity) + ".txt"
        output = "output_" + str(complexity) + "_" + str(int(utc_timestamp)) + ".png"
    else:
        # Check usage
        if len(sys.argv) not in [3, 4]:
            sys.exit("Usage: python generate.py structure words [output]")

        # Parse command-line arguments
        structure = sys.argv[1]
        words = sys.argv[2]
        output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
