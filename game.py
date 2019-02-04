import numpy as np

BOARD_SIZE = 4


def initialState():
    board = np.zeros((4,4), dtype=np.uint8)
    score = 0
    return (randomSpawn(board), score)

def actions(state):
    board, score = state
    actions = []
    for d in range(0,4):
        new, same, total = swipeBoard(board, d)
        if not same:
            actions.append(d)

    return actions

def successor(state, action):
    board, score = state
    b, _, s = swipeBoard(board, action)
    b = randomSpawn(b)
    return (b, score + s)


def isEnd(state):
    return len(actions(state)) == 0

def score(state):
    return state[1]
    #return hyperbolic(state)
def hyperbolic(state):
    board, score = state
    tot = 0
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            tot += 2**(board[i,j])/((i+1)*(j+1) + 1)

    return tot


def randomSpawn(b):
    if np.count_nonzero(b) == 16:
        return None
    else:
        #b = board.copy()
        randGen = 1 if np.random.random() < 0.9 else 2
        zeros = np.where(b==0)
        randIndex = np.random.choice(zeros[0].shape[0], 1)
        b[zeros[0][randIndex], zeros[1][randIndex]] = randGen
        return b

def swipeBoard(board, direction):
    # LEFT
    if direction == 0:
        b, total = swipeLeft(board)
        
    # RIGHT
    elif direction == 1:
        b, total = swipeLeft(np.rot90(board, 2))
        b = np.rot90(b, 2)
    # DOWN
    elif direction == 2:
        b, total = swipeLeft(np.rot90(board, 1))
        b = np.rot90(b, 3)
    
    # UP
    elif direction == 3:
        b, total = swipeLeft(np.rot90(board, 3))
        b = np.rot90(b, 1)

    return b, np.array_equal(b,board), total

def swipeLeft(board):
    board = board.copy()
    total = 0
    for i in range(board.shape[0]):
        total += swipeRowLeft(board, i)
    return board, total

def swipeRowLeft(board, row):
    total = 0
    for col in range(4):
        if board[row, col] != 0:
            for col2 in range(col+1, 4):
                if (board[row, col2]):
                    if (board[row, col] == board[row, col2]):
                        board[row, col] += 1
                        total += 2**board[row, col]
                        col += 1
                        board[row, col2] = 0
                    break
    # compaction steps
    nonzero = np.where(board[row, :] !=0)[0].tolist()
    for i in range(4):
        if i >= len(nonzero):
            board[row, i] = 0
        else:
            board[row, i] = board[row, nonzero[i]]

    
    return total

def featureExtractor(state, action):
    board, score = state

    result, _, _ = swipeBoard(board, action)
    
    return result.reshape(1, 1, 4, 4)



def pretty(state):
    board, score = state

    print("\n SCORE: ", score)
    print('\n'+'______'*4)
    for i in range(board.shape[0]):

        for j in range(board.shape[1]):

            print('|  ' + str(board[i, j]) + '  ', end='' )
        
        print('|\n'+'______'*4)
