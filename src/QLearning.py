'''
Basic implementation of Q Learning without neural networks
Sometimes the machine will get stuck in an infinite loop of non-scoring moves. If so just rerun the script
'''
import random
from Snake import SnakeGame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#%%

def evaluateScore(Q, boardDim, numRuns, displayGame=False):
    # Run the game for a specified number of runs given a specific Q matrix
    cutoff = 100  # X moves without increasing score will cut off this game run
    scores = []
    for i in range(numRuns):
        game = SnakeGame(boardDim, boardDim)
        state = game.calcStateNum()
        score = 0
        oldScore = 0
        gameOver = False
        moveCounter = 0
        while not gameOver:
            possibleQs = Q[state, :]
            action = np.argmax(possibleQs)
            state, reward, gameOver, score = game.makeMove(action)
            if score == oldScore:
                moveCounter += 1
            else:
                oldScore = score
                moveCounter = 0
            if moveCounter >= cutoff:
                # stuck going back and forth
                break
        scores.append(score)
    return np.average(scores), scores


# %%
boardDim = 16  # size of the baord

# state is as follows.
# Is direction blocked by wall or snake?
# Is food in this direction? can either be one or two directions eg (food is left) or (food is left and up)
# state =  (top blocked, right blocked, down blocked, left blocked, up food, right food, down food, left food)
# 8 boolean values. Not all states are reachable (eg states with food directions that don't make sense)
numStates = 2**8
numActions = 4  # 4 directions that the snake can move
Q = np.zeros((numStates, numActions))

# lr = 0.9 #learning rate. not used in this Q learning equation
gamma = 0.8  # discount rate
epsilon = 0.2  # exploration rate in training games
numEpochs = 2600  # number of games to train for

Qs = dict()
bestLength = 0
print("Training for", numEpochs, "games...")
for epoch in range(numEpochs):
    #    print("New Game")
    game = SnakeGame(boardDim, boardDim)
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

        # http: // mnemstudio.org/path-finding-q-learning-tutorial.htm
        Q[state, action] = reward + gamma * np.max(Q[new_state, :])

        # https://towardsdatascience.com/simple-reinforcement-learning-q-learning-fcddc4b6fe56
        # Q[state, action] = Q[state, action] + lr * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])
        state = new_state
    if epoch % 100 == 0:
        Qs[epoch] = np.copy(Q)
        averageLength, lengths = evaluateScore(Q, boardDim, 25)
        if averageLength > bestLength:
            bestLength = averageLength
            bestQ = np.copy(Q)
        print("Epoch", epoch, "Average snake length without exploration:", averageLength)
        
#%%
print("Generating data for animation...")
maxFrames = 1000
plotEpochs = [0, 200, 400, 600, 800, 1000, 1500, 2000, 2500]
fig, axes = plt.subplots(3, 3, figsize=(8,8))
axList = []
ims = []
dataArrays = []
games = []
states = []
scores = []
images = []
labels = []
for i, row in enumerate(axes):
    for j, ax in enumerate(row):
        ax.set_title("Epoch " + str(plotEpochs[i*len(row) + j]))
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        axList.append(ax)
        ims.append(ax.imshow(np.zeros([boardDim, boardDim]), vmin=-1, vmax=1, cmap='RdGy'))
        labels.append(ax.text(0,15, "Length: 0"))
        dataArrays.append(list())
        scores.append(list())
        game = SnakeGame(boardDim, boardDim)
        games.append(game)
        states.append(game.calcStateNum())
for j in range(maxFrames):
    for i in range(len(plotEpochs)):
        possibleQs = Qs[plotEpochs[i]][states[i], :]
        action = np.argmax(possibleQs)
        states[i], reward, gameOver, score = games[i].makeMove(action)
        dataArrays[i].append(games[i].plottableBoard())
        scores[i].append(score)

def animate(frameNum):
    for i, im in enumerate(ims):
        labels[i].set_text("Length: " + str(scores[i][frameNum]))
        ims[i].set_data(dataArrays[i][frameNum])
    return ims+labels
print("Animating snakes at different epochs...")
ani = animation.FuncAnimation(fig, func=animate, frames=maxFrames,blit=True, interval=75, repeat=False, )

##uncomment below if you want to output to a video file
#print("Saving to file")
#ani.save('output.mp4', fps=15, extra_args=['-vcodec', 'libx264'])
#print("Done")
plt.show()