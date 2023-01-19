import numpy as np
import copy
import heapq

def print_succ(state):
    states = successors(state)
    for s in states: 
            print(s, "h= " + str(heuristic(s)))
            
def successors(state):
    """
    Returns: a list of successors to the current state
    """
    board = np.array(state).reshape(3,3)
    states = []
    
    space_coordinate = ()
    for row in range(3):
        for col in range(3):
            if board[row][col] == 0:
                space_coordinate = (row, col)
        else:
            continue
        break
    
    row = space_coordinate[0]
    col = space_coordinate[1]
    
    if row == 0:
        states.append(list(copy_board(board, row, col, 1, 0).flatten()))
    elif row == 2: 
        states.append(list(copy_board(board, row, col, -1, 0).flatten()))
    else:
        states.append(list(copy_board(board, row, col, 1, 0).flatten()))
        states.append(list(copy_board(board, row, col, -1, 0).flatten()))
    
    if col == 0:
        states.append(list(copy_board(board, row, col, 0, 1).flatten()))
    elif col == 2:
        states.append(list(copy_board(board, row, col, 0, -1).flatten()))
    else:
        states.append(list(copy_board(board, row, col, 0, 1).flatten()))
        states.append(list(copy_board(board, row, col, 0, -1).flatten()))
    return sorted(states)

def copy_board(board, row, col, move_row, move_col):
    """
    Copies the board reference and makes changes to find a successor
    """
    new_board = copy.deepcopy(board)
    new_board[row][col] = new_board[row+move_row][col+move_col]
    new_board[row+move_row][col+move_col] = 0
    return new_board

def heuristic(state):
    """
    Helper function for print_succ
    Returns: value of heuristic function of current state 
    """
    h = 0
    goal = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1)]
    board = np.array(state).reshape(3,3)
    for row in range(3):
        for col in range(3):
            label = board[row][col]
            if label == 0:
                continue
            h += abs(goal[label-1][0] - row) + abs(goal[label-1][1] - col)
    return h

def solve(state):
    
    goal = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    pq = []
    path = []
    list_parents = []
    heapq.heappush(pq, (heuristic(state), state, (0, heuristic(state), -1)))
    while(len(pq) != 0):
        n = heapq.heappop(pq)
        path.append(n)
        if n[1] == goal:
            break
        for successor in successors(n[1]):
            list_parents.append((successor, n[1]))
            parent_index = n[2][2] + 1
            g =  n[2][0] + 1
            h = heuristic(successor)
            n_next = (g + h, successor, (g, h, parent_index))
            in_path = False
            in_pq = False            
                    
            for s in pq:
                if s[1] == successor:
                    in_pq = True
                    break

            for s in path:
                if s[1] == successor:
                    in_path = True
                    break
            
            if (not in_path) and (not in_pq):
                heapq.heappush(pq, n_next)
    
    next_node = path[-1]
    p = []
    for i, curr in reversed(list(enumerate(path))):
        if next_node != curr:
            continue
        if curr[2][2] == -1:
            p.append(curr)
            break
            
        parent_found = False
        
        curr_parent = []
        for pair in reversed(list_parents):
            if pair[0] == curr[1]:
                if pair[1] == path[i-1][1]:
                    next_node = path[i - 1]
                    parent_found = True
                    break
                else:
                    curr_parent.append(pair[1])
                    
        if not parent_found:
            n = 2
            while(path[i-n][1] not in curr_parent):
                n = n + 1
            next_node = path[i-n]
        p.append(curr)
            
    for board in reversed(p):
        print("{} h={} moves: {}".format(board[1], board[2][1], board[2][0]))
        
if __name__ == "__main__":
    solve([4,3,8,5,1,6,7,2,0])