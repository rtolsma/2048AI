
from player import ModelTest


#Number of times we update QModels
NUM_ITERATIONS = 10**3

# Number of Games a fixed QModel plays
NUM_GAMES = 1

if __name__ == "__main__":
    m = ModelTest(verbose=False)
    for j in range(NUM_ITERATIONS):
        print("\n\n\n\n\n STARTING ITERATION: ", j, "\n\n\n\n\n ")
        for i in range(NUM_GAMES):
            print("GAME:", i, "ITERATION:" , j)
            m.playGame()
            
        m.refresh()
        
        