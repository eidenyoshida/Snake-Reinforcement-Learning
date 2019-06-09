'''
Basic implementation of Q Learning without neural networks
Sometimes the machine will get stuck in an infinite loop of back and forth moves. If so just rerun the script
'''
import random
from Snake import SnakeGame
import numpy as np
import time
import os


# Q learning equation from
# https://towardsdatascience.com/simple-reinforcement-learning-q-learning-fcddc4b6fe56

def clear():
    #to clear the console between moves
    if os.name == 'posix':
        os.system('clear')
    else:
        os.system('cls')

def evaluateScore(Q, numRuns, displayGame):
    # Run the game for a specified number of runs given a specific Q matrix
    cutoff = 250 # X moves without increasing score will cut off this game run
    scores = []
    for i in range(numRuns):
        game = SnakeGame(16, 16)
        state = game.calcStateNum()
        score = 0
        oldScore = 0
        gameOver = False
        moveCounter = 0
        while not gameOver:
            possibleQs = Q[state, :]
            action = np.argmax(possibleQs)
            state, reward, gameOver, score = game.makeMove(action)
            if displayGame:
                game.display()
                print("Snake Length:", score)
                # sleep so we can see the machine play
                time.sleep(0.05)
                clear()
            if score==oldScore:
                moveCounter += 1
            else:
                oldScore = score
                moveCounter = 0
            if moveCounter >= cutoff:
                #stuck going back and forth
                scores.append(score)
                break
            
    return np.average(score)

#%%
# state is as follows.
# Is direction blocked by wall or snake?
# Is food in this direction? can either be 1 or two directions eg left and up
# (top blocked, right blocked, down blocked, left blocked, up food, right food, down food, left food)
numStates = 2**8  # 8 boolean values. Not all states are reachable (eg states with 0 or 3 or 4 food directions)
numActions = 4  # 4 directions
Q = np.zeros((numStates, numActions))

lr = 0.9 #learning rate
gamma = 0.9 #discount rate
epsilon = 0.2 #exploration rate in training games
numEpochs = 2500 #number of games

Qs = []
print("Training for", numEpochs, "games...")
for epoch in range(numEpochs):
    #    print("New Game")
    game = SnakeGame(16, 16)
    state = game.calcStateNum()
    gameOver = False
    score = 0
    while not gameOver:
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, 3)
        else:
            possibleQs = Q[state, :]
            action = np.argmax(possibleQs)
        new_state, reward, gameOver, score = game.makeMove(action)
        Q[state, action] = Q[state, action] + lr * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])
        state = new_state
    if epoch % 100 == 0:
        print("Epoch", epoch, "Average snake length:", evaluateScore(Q, 25, False))

# %%

print("Testing with last train Q matrix...")
print("Average snake length:", evaluateScore(Q, 5, True))