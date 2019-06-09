'''
Basic implementation of Q Learning without neural networks
'''
import random
from Snake import SnakeGame
import numpy as np
import time
import os
from subprocess import call
# sometimes the machine will get stuck in an infinite loop of back and forth moves.
# if so just rerun the script

# Q learning equation from
# https://towardsdatascience.com/simple-reinforcement-learning-q-learning-fcddc4b6fe56

def clear():
    #to clear the console between moves
    if os.name == 'posix':
        os.system('clear')
    else:
        os.system('cls')
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
numEpochs = 10000 #number of games

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
        print("Epoch", epoch)

# %%

print("Testing with last train Q matrix...")
scores = []
for testRun in range(5):
    game = SnakeGame(16, 16)
    state = game.calcStateNum()
    gameOver = False
    score = 0
    print("New Game")
    while not gameOver:
        possibleQs = Q[state, :]
        action = np.argmax(possibleQs)
        game.display()
        print("Snake Length:", score)
        # sleep so we can see the machine play
        time.sleep(0.05)
        clear()
        state, reward, gameOver, score = game.makeMove(action)
    game.display()
    print("Final Snake Length:", score)
    scores.append(score)
print("Final snake lengths: ", scores, "Average:", np.average(scores))
