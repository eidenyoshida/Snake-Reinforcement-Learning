##Generates convergence graph after QLearning script has run
#%%
usedQs = []
for index, val in np.ndenumerate(Qs[0,:,:]):
    qLine = Qs[:, index[0], index[1]][:5000]
    if any([item != 0 for item in qLine]) and qLine[-1] != 0:
        usedQs.append(np.copy(qLine))
        
usedQs = np.abs(usedQs)
y = []
finalVals = []
for i in range(len(usedQs)):
    #y.append((fullQs[j][hasSomethingList[i][0],hasSomethingList[i][1]] - maxVals[i])/maxVals[i])
    finalVal = np.mean(usedQs[i][-100:])
    finalVals.append(finalVal)
    y = (usedQs[i]-finalVal) / finalVal
    plt.plot(y,color="black",alpha=0.025)
plt.ylim(-1,0.5)
plt.title("Convergence of Q Table")
plt.ylabel("Normalized Q Value")
plt.xlabel("Episode Number")
plt.savefig("ConvergenceGraph.png", dpi=300)
plt.show()