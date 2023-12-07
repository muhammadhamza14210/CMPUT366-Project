import matplotlib.pyplot as plt
import numpy as np
import time 


class PlotResults:
    """
    Class to plot the results. 
    """
    def plot_results(self, data1, data2, label1, label2, filename):
        """
        This method receives two lists of data point (data1 and data2) and plots
        a scatter plot with the information. The lists store statistics about individual search 
        problems such as the number of nodes a search algorithm needs to expand to solve the problem.

        The function assumes that data1 and data2 have the same size. 

        label1 and label2 are the labels of the axes of the scatter plot. 
        
        filename is the name of the file in which the plot will be saved.
        """
        _, ax = plt.subplots()
        ax.scatter(data1, data2, s=100, c="g", alpha=0.5, cmap=plt.cm.coolwarm, zorder=10)
    
        lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
    
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        plt.xlabel(label1)
        plt.ylabel(label2)
        plt.grid()
        plt.savefig(filename)

class Grid:
    """
    Class to represent an assignment of values to the 81 variables defining a Sudoku puzzle. 

    Variable _cells stores a matrix with 81 entries, one for each variable in the puzzle. 
    Each entry of the matrix stores the domain of a variable. Initially, the domains of variables
    that need to have their values assigned are 123456789; the other domains are limited to the value
    initially assigned on the grid. Backtracking search and AC3 reduce the the domain of the variables 
    as they proceed with search and inference.
    """
    def __init__(self):
        self._cells = []
        self._complete_domain = "123456789"
        self._width = 9

    def copy(self):
        """
        Returns a copy of the grid. 
        """
        copy_grid = Grid()
        copy_grid._cells = [row.copy() for row in self._cells]
        return copy_grid

    def get_cells(self):
        """
        Returns the matrix with the domains of all variables in the puzzle.
        """
        return self._cells

    def get_width(self):
        """
        Returns the width of the grid.
        """
        return self._width

    def read_file(self, string_puzzle):
        """
        Reads a Sudoku puzzle from string and initializes the matrix _cells. 

        This is a valid input string:

        4.....8.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......

        This is translated into the following Sudoku grid:

        - - - - - - - - - - - - - 
        | 4 . . | . . . | 8 . 5 | 
        | . 3 . | . . . | . . . | 
        | . . . | 7 . . | . . . | 
        - - - - - - - - - - - - - 
        | . 2 . | . . . | . 6 . | 
        | . . . | . 8 . | 4 . . | 
        | . . . | . 1 . | . . . | 
        - - - - - - - - - - - - - 
        | . . . | 6 . 3 | . 7 . | 
        | 5 . . | 2 . . | . . . | 
        | 1 . 4 | . . . | . . . | 
        - - - - - - - - - - - - - 
        """
        i = 0
        row = []
        for p in string_puzzle:
            if p == '.':
                row.append(self._complete_domain)
            else:
                row.append(p)

            i += 1

            if i % self._width == 0:
                self._cells.append(row)
                row = []
            
    def print(self):
        """
        Prints the grid on the screen. Example:

        - - - - - - - - - - - - - 
        | 4 . . | . . . | 8 . 5 | 
        | . 3 . | . . . | . . . | 
        | . . . | 7 . . | . . . | 
        - - - - - - - - - - - - - 
        | . 2 . | . . . | . 6 . | 
        | . . . | . 8 . | 4 . . | 
        | . . . | . 1 . | . . . | 
        - - - - - - - - - - - - - 
        | . . . | 6 . 3 | . 7 . | 
        | 5 . . | 2 . . | . . . | 
        | 1 . 4 | . . . | . . . | 
        - - - - - - - - - - - - - 
        """
        for _ in range(self._width + 4):
            print('-', end=" ")
        print()

        for i in range(self._width):

            print('|', end=" ")

            for j in range(self._width):
                if len(self._cells[i][j]) == 1:
                    print(self._cells[i][j], end=" ")
                elif len(self._cells[i][j]) > 1:
                    print('.', end=" ")
                else:
                    print(';', end=" ")

                if (j + 1) % 3 == 0:
                    print('|', end=" ")
            print()

            if (i + 1) % 3 == 0:
                for _ in range(self._width + 4):
                    print('-', end=" ")
                print()
        print()

    def print_domains(self):
        """
        Print the domain of each variable for a given grid of the puzzle.
        """
        for row in self._cells:
            print(row)

    def is_solved(self):
        """
        Returns True if the puzzle is solved and False otherwise. 
        """
        for i in range(self._width):
            for j in range(self._width):
                if len(self._cells[i][j]) > 1 or not self.is_value_consistent(self._cells[i][j], i, j):
                    return False
        return True
    
    def is_value_consistent(self, value, row, column):
        for i in range(self.get_width()):
            if i == column: continue
            if self.get_cells()[row][i] == value:
                return False
        
        for i in range(self.get_width()):
            if i == row: continue
            if self.get_cells()[i][column] == value:
                return False

        row_init = (row // 3) * 3
        column_init = (column // 3) * 3

        for i in range(row_init, row_init + 3):
            for j in range(column_init, column_init + 3):
                if i == row and j == column:
                    continue
                if self.get_cells()[i][j] == value:
                    return False
        return True

class VarSelector:
    """
    Interface for selecting variables in a partial assignment. 

    Extend this class when implementing a new heuristic for variable selection.
    """
    def select_variable(self, grid):
        pass

class FirstAvailable(VarSelector):
    """
    Naïve method for selecting variables; simply returns the first variable encountered whose domain is larger than one.
    """
    def select_variable(self, grid):
        # Implement here the first available heuristic
        for i in range(grid.get_width()):
            for j in range(grid.get_width()):
                if len(grid.get_cells()[i][j]) > 1:
                    return (i, j)
        return None

class MRV(VarSelector):
    """
    Implements the MRV heuristic, which returns one of the variables with smallest domain. 
    """
    def select_variable(self, grid):
        # Implement here the mrv heuristic
        min_domain_size = float('inf')
        selected_variable = None

        for i in range(grid.get_width()):
            for j in range(grid.get_width()):
                current_domain_size = len(grid.get_cells()[i][j])

                if 1 < current_domain_size < min_domain_size:
                    min_domain_size = current_domain_size
                    selected_variable = (i, j)

        return selected_variable


class DegreeHeuristic(VarSelector):
    def select_variable(self, grid):
        max_degree = -1
        selected_variable = None

        for i in range(grid.get_width()):
            for j in range(grid.get_width()):
                if len(grid.get_cells()[i][j]) > 1:
                    current_degree = self.calculate_degree(grid, i, j)

                    if current_degree > max_degree:
                        max_degree = current_degree
                        selected_variable = (i, j)

        return selected_variable
    
    
    def calculate_degree(self, grid, row, column):
        degree = 0

        # Check conflicts in the same row, column, and unit
        for i in range(grid.get_width()):
            if i != column and len(grid.get_cells()[row][i]) > 1:
                degree += self.has_conflict(grid, grid.get_cells()[row][i], row, i)

            if i != row and len(grid.get_cells()[i][column]) > 1:
                degree += self.has_conflict(grid, grid.get_cells()[i][column], i, column)

        unit_row = (row // 3) * 3
        unit_column = (column // 3) * 3

        for i in range(unit_row, unit_row + 3):
            for j in range(unit_column, unit_column + 3):
                if i != row and j != column and len(grid.get_cells()[i][j]) > 1:
                    degree += self.has_conflict(grid, grid.get_cells()[i][j], i, j)

        return degree

    def has_conflict(self, grid, value, row, column):
        return not grid.is_value_consistent(value, row, column)

class AC3:
    """
    This class implements the methods needed to run AC3 on Sudoku. 
    """
    def remove_domain_row(self, grid, row, column):
        """
        Given a matrix (grid) and a cell on the grid (row and column) whose domain is of size 1 (i.e., the variable has its
        value assigned), this method removes the value of (row, column) from all variables in the same row. 
        """
        variables_assigned = []

        for j in range(grid.get_width()):
            if j != column:
                new_domain = grid.get_cells()[row][j].replace(grid.get_cells()[row][column], '')

                if len(new_domain) == 0:
                    return None, True

                if len(new_domain) == 1 and len(grid.get_cells()[row][j]) > 1:
                    variables_assigned.append((row, j))

                grid.get_cells()[row][j] = new_domain
        
        return variables_assigned, False

    def remove_domain_column(self, grid, row, column):
        """
        Given a matrix (grid) and a cell on the grid (row and column) whose domain is of size 1 (i.e., the variable has its
        value assigned), this method removes the value of (row, column) from all variables in the same column. 
        """
        variables_assigned = []

        for j in range(grid.get_width()):
            if j != row:
                new_domain = grid.get_cells()[j][column].replace(grid.get_cells()[row][column], '')
                
                if len(new_domain) == 0:
                    return None, True

                if len(new_domain) == 1 and len(grid.get_cells()[j][column]) > 1:
                    variables_assigned.append((j, column))

                grid.get_cells()[j][column] = new_domain

        return variables_assigned, False

    def remove_domain_unit(self, grid, row, column):
        """
        Given a matrix (grid) and a cell on the grid (row and column) whose domain is of size 1 (i.e., the variable has its
        value assigned), this method removes the value of (row, column) from all variables in the same unit. 
        """
        variables_assigned = []

        row_init = (row // 3) * 3
        column_init = (column // 3) * 3

        for i in range(row_init, row_init + 3):
            for j in range(column_init, column_init + 3):
                if i == row and j == column:
                    continue

                new_domain = grid.get_cells()[i][j].replace(grid.get_cells()[row][column], '')

                if len(new_domain) == 0:
                    return None, True

                if len(new_domain) == 1 and len(grid.get_cells()[i][j]) > 1:
                    variables_assigned.append((i, j))

                grid.get_cells()[i][j] = new_domain
        return variables_assigned, False

    def pre_process_consistency(self, grid):
        """
        This method enforces arc consistency for the initial grid of the puzzle.

        The method runs AC3 for the arcs involving the variables whose values are 
        already assigned in the initial grid. 
        """
        # Implement here the code for making the CSP arc consistent as a pre-processing step; this method should be called once before search
        Q = [(i, j) for i in range(grid.get_width()) for j in range(grid.get_width()) if len(grid.get_cells()[i][j]) == 1]

        # Call the consistency method to enforce arc consistency
        return self.consistency(grid, Q)

    def consistency(self, grid, Q):
        """
        This is a domain-specific implementation of AC3 for Sudoku. 

        It keeps a set of variables to be processed (Q) which is provided as input to the method. 
        Since this is a domain-specific implementation, we don't need to maintain a graph and a set 
        of arcs in memory. We can store in Q the cells of the grid and, when processing a cell, we
        ensure arc consistency of all variables related to this cell by removing the value of
        cell from all variables in its column, row, and unit. 

        For example, if the method is used as a preprocessing step, then Q is initialized with 
        all cells that start with a number on the grid. This method ensures arc consistency by
        removing from the domain of all variables in the row, column, and unit the values of 
        the cells given as input. Like the general implementation of AC3, the method adds to 
        Q all variables that have their values assigned during the propagation of the contraints. 

        The method returns True if AC3 detected that the problem can't be solved with the current
        partial assignment; the method returns False otherwise. 
        """
        # Implement here the domain-dependent version of AC3.
        while Q:
            var = Q.pop()

            # Remove value from the same row, column, and unit
            rows_assigned, failure_row = self.remove_domain_row(grid, var[0], var[1])
            columns_assigned, failure_column = self.remove_domain_column(grid, var[0], var[1])
            units_assigned, failure_unit = self.remove_domain_unit(grid, var[0], var[1])

            # If any removal returned failure, return failure
            if failure_row or failure_column or failure_unit:
                return True

            # Add to Q all variables that had their domains reduced to size 1
            Q.extend(rows_assigned + columns_assigned + units_assigned)

        return False

class Backtracking:
    """
    Class that implements backtracking search for solving CSPs. 
    """

    def search(self, grid, var_selector):
        """
        Implements backtracking search with inference. 
        """
        # Implemente here the Backtracking search.

        ac3 = AC3()
        failure_pre_process = ac3.pre_process_consistency(grid)
        
        # If pre-processing fails, return None
        if failure_pre_process:
            return None

        # Base case: if the grid is complete, return it
        if grid.is_solved():
            return grid

        # Select an unassigned variable using the given variable selector
        var = var_selector.select_variable(grid)

        # If no unassigned variable is found, the current assignment is complete
        if var is None:
            return grid

        row, column = var  # Extract row and column from the variable

        # Iterate through the domain values arbitrarily
        for value in grid.get_cells()[row][column]:
            # Check if the value is consistent with the current assignment
            if grid.is_value_consistent(value, row, column):
                # Create a copy of the grid
                copy_grid = grid.copy()

                # Assign the value to the selected variable in the copy
                copy_grid.get_cells()[row][column] = value

                # Run consistency for the assigned value
                failure_consistency = ac3.consistency(copy_grid, [(row, column)])

                # If consistency fails, skip this value assignment and backtrack
                if not failure_consistency:
                    # Recursively call the search method for the next variable
                    result = self.search(copy_grid, var_selector)

                    # If a solution is found, return it
                    if result is not None and result.is_solved():
                        return result

        # If no consistent value is found, return failure (None)
        return None
    
class BacktrackingLearnNoGoods(Backtracking):
    """
    Extends the Backtracking class to incorporate learning no-goods.
    """

    def __init__(self):
        super().__init__()
        self.learned_no_goods = set()

    def search(self, grid, var_selector):
        """
        Implements backtracking search with inference and learning no-goods using an iterative approach.
        """
        stack = [(grid, set())]  # Stack to manage state: (current_grid, visited_nodes)
        ac3 = AC3()

        while stack:
            current_grid, visited_nodes = stack.pop()

            failure_pre_process = ac3.pre_process_consistency(current_grid)

            if failure_pre_process:
                continue

            if current_grid.is_solved():
                return current_grid

            var = var_selector.select_variable(current_grid)

            if var is None:
                continue

            row, column = var

            value_found = False
            for value in current_grid.get_cells()[row][column]:
                if current_grid.is_value_consistent(value, row, column):
                    copy_grid = current_grid.copy()
                    copy_grid.get_cells()[row][column] = value
                    failure_consistency = ac3.consistency(copy_grid, [(row, column)])

                    if not failure_consistency:
                        # Make sure to check if the current grid is already in learned_no_goods
                        if copy_grid not in visited_nodes and copy_grid not in self.learned_no_goods:
                            stack.append((copy_grid, visited_nodes | {copy_grid}))
                            value_found = True

                            if copy_grid.is_solved():
                                return copy_grid

            # If no consistent value is found, record the conflict as a learned no-good
            if not value_found:
                visited_nodes.add(current_grid)
                self.learned_no_goods.add(current_grid)

        return None


if __name__ == '__main__':
    plotter = PlotResults()
    file = open('top95.txt', 'r')
    problems = file.readlines()
    counter_first_available = 0 
    counter_mrv = 0
    counter_degree_heurisitic = 0
    running_time_mrv = []
    running_time_first_available = []
    running_time_degree_heuristic = []

    for p in problems:
        # Read problem from string
        g = Grid()
        g.read_file(p)

        # Print the grid on the screen
        print('Puzzle')
        g.print()

        copy_g = g.copy()

        # # Removing 2 from the domain of the variable in the first row and second column
        copy_g.get_cells()[0][1] = copy_g.get_cells()[0][1].replace('2', '')

        # Instace of Backtracking Object
        a = time.time()
        backtracking = Backtracking()
        backtracking_learn_no_goods = BacktrackingLearnNoGoods()
        # Instance of AC3 Object
        ac3 = AC3()
        
        
        print('Solving the puzzle using Backtracking with FirstAvailable:')
        start_time = time.time()
        solution_first_available = backtracking.search(g, FirstAvailable())
        end_time = time.time()
        running_time_first_available.append(end_time-start_time)
        
        if solution_first_available and solution_first_available.is_solved():
            print('Is the current grid a solution? ', solution_first_available.is_solved())
            solution_first_available.print()
            counter_first_available += 1
        else:
            print('Is the current grid a solution? ', solution_first_available.is_solved())
        
        # Solve the puzzle using backtracking with the MRV variable selector
        print('Solving the puzzle using Backtracking with MRV:')
        start_time2 = time.time()
        solution_mrv = backtracking.search(g, MRV())
        end_time2 = time.time()
        running_time_mrv.append(end_time2-start_time2)
        
        if solution_mrv and solution_mrv.is_solved():
            print('Is the current grid a solution? ', solution_mrv.is_solved())
            solution_mrv.print()
            counter_mrv += 1
        else:
            print('Is the current grid a solution? ', solution_mrv.is_solved())

        # Solve the puzzle using backtracking with the MRV variable selector
        print('Solving the puzzle using Backtracking with Degree Heuristic:')
        start_time3 = time.time()
        solution_degree_heuristic = backtracking.search(g, DegreeHeuristic())
        end_time3 = time.time()
        running_time_degree_heuristic.append(end_time3 - start_time3)

        if solution_degree_heuristic and solution_degree_heuristic.is_solved():
            print('Is the current grid a solution? ', solution_degree_heuristic.is_solved())
            solution_degree_heuristic.print()
            counter_degree_heurisitic += 1
        else:
            print('Is the current grid a solution? ', solution_degree_heuristic.is_solved())
            print('Partial solution:')
            solution_degree_heuristic.print()  # Add this line to print the partial solution for debugging

            
        #print('-' * 40)  # Add a separator between problems
        
        
        """
        
        print('Solving the puzzle using Backtracking with Learning No-Goods FA:')
        start_time = time.time()
        solution_learn_no_goods_fa = backtracking_learn_no_goods.search(g, FirstAvailable())
        end_time = time.time()
        running_time_first_available.append(end_time-start_time)
        
        if solution_learn_no_goods_fa and solution_learn_no_goods_fa.is_solved():
            print('Is the current grid a solution? ', solution_learn_no_goods_fa.is_solved())
            solution_learn_no_goods_fa.print()
            counter_first_available += 1
        else:
            print('Is the current grid a solution? ', solution_learn_no_goods_fa.is_solved())

        print("Running Time of Backtracking with Learning No-Goods: ", end_time - start_time)
        print("Learned No-Goods: ", len(backtracking_learn_no_goods.learned_no_goods))
        
        print('-' * 40)
        
        print('Solving the puzzle using Backtracking with Learning No-Goods MRV:')
        start_time2 = time.time()
        solution_learn_no_goods_mrv = backtracking_learn_no_goods.search(g, MRV())
        end_time2 = time.time()
        running_time_mrv.append(end_time2-start_time2)
        
        if solution_learn_no_goods_mrv and solution_learn_no_goods_mrv.is_solved():
            print('Is the current grid a solution? ', solution_learn_no_goods_mrv.is_solved())
            solution_learn_no_goods_mrv.print()
            counter_mrv += 1
        else:
            print('Is the current grid a solution? ', solution_learn_no_goods_fa.is_solved())

        print("Running Time of Backtracking with Learning No-Goods: ", end_time2 - start_time2)
        print("Learned No-Goods: ", len(backtracking_learn_no_goods.learned_no_goods))
        
        print('-' * 40)
        
        print('Solving the puzzle using Backtracking with Learning No-Goods Degree Heuristic:')
        start_time3 = time.time()
        solution_learn_no_goods_degree_heuristic = backtracking_learn_no_goods.search(g, DegreeHeuristic())
        end_time3 = time.time()
        running_time_degree_heuristic.append(end_time3-start_time3)
        
        if solution_learn_no_goods_degree_heuristic and solution_learn_no_goods_degree_heuristic.is_solved():
            print('Is the current grid a solution? ', solution_learn_no_goods_degree_heuristic.is_solved())
            solution_learn_no_goods_degree_heuristic.print()
            counter_degree_heurisitic += 1
        else:
            print('Is the current grid a solution? ', solution_learn_no_goods_degree_heuristic.is_solved())

        print("Running Time of Backtracking with Learning No-Goods: ", end_time3 - start_time3)
        print("Learned No-Goods: ", len(backtracking_learn_no_goods.learned_no_goods))
        
        print('-' * 40)
        
        """


    print("The number of solutions solved from top95.txt with first available method is: ", counter_first_available)
    print("The number of solutions solved from top95.txt with mrv method is: ", counter_mrv)
    print("The number of solutions solved from top95.txt with degree heuristic method is: ", counter_degree_heurisitic)
    print("Running Time of Degree Heuristic", running_time_degree_heuristic)
    print("Running Time of mrv", running_time_mrv)
    print("Running Time of first available", running_time_first_available)