# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """


  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.
    getAction chooses among the best options according to the evaluation function.
    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.
    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    foodListPacman = newFood.asList()

    # To find the closest food and closest ghost, I sort the respective lists by manhattandistance from the ghost

    foodListPacman.sort(lambda x,y: util.manhattanDistance(newPos, x)-util.manhattanDistance(newPos, y))
    nearestFoodScore=float(util.manhattanDistance(newPos, foodListPacman[0]))

    ghostPositions=[(int(Ghost.getPosition()[0]),int(Ghost.getPosition()[1])) for Ghost in newGhostStates]
    ghostPositions.sort(lambda x,y: util.manhattanDistance(newPos, x)-util.manhattanDistance(newPos, y))
    nearestGhostScore=float(util.manhattanDistance(newPos, ghostPositions[0]))

    # if we are currently at a Ghost or Food return the respective score
    if nearestGhostScore == 0:
      return -200

    if nearestFoodScore == 0:
      return 1


    #this guides to the closest food, with subtracting the closest ghost at a higher cost value
    finalScore =  (1.0/nearestFoodScore) - (2 * 1.0/nearestGhostScore)
    return finalScore
        #return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.
      Here are some method calls that might be useful when implementing minimax.
      
      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1
      
      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action
      
      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    retValue = self.miniMax(gameState,0,0)
    return retValue[1]


  def miniMax(self, gameState, currentIndex, currentDepth, alpha=None, beta=None):
    if currentIndex >= gameState.getNumAgents():
      currentIndex = 0
      currentDepth += 1

    if currentDepth == self.depth:
        return (self.evaluationFunction(gameState), 'none')

    if currentIndex == 0:
        return self.maximumValue(gameState, currentIndex, currentDepth)
    else:
        return self.minimumValue(gameState, currentIndex, currentDepth)

  def minimumValue(self,gameState,index, currentDepth, alpha=None, beta = None):
    moves = gameState.getLegalActions(index)
    bestScore = float('inf')
    bestMove = None
    if not moves: return (self.evaluationFunction(gameState), 'none')

    for move in moves:
      if move == Directions.STOP: 
        continue 
      nextState = gameState.generateSuccessor(index,move)
      score, newMove = self.miniMax(nextState,index+1,currentDepth)
      if score < bestScore:
        bestMove = move
        bestScore = score

    return (bestScore,bestMove)

  def maximumValue(self,gameState,index, currentDepth, alpha=None, beta=None):
    moves = gameState.getLegalActions(index)
    bestScore = float('-inf')
    bestMove = None
    if not moves: return (self.evaluationFunction(gameState), 'none')

    for move in moves:
      if move == Directions.STOP: 
        continue 
      nextState = gameState.generateSuccessor(index,move)
      score, newMove = self.miniMax(nextState,index+1,currentDepth)
      if score > bestScore:
        bestMove = move
        bestScore = score

    return (bestScore,bestMove)



class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    retValue = self.miniMax(gameState, 0, 0, float('-inf'), float('inf'))
    return retValue[1]

  #This follows the same logic as before, but now we will prune
  def miniMax(self, gameState, currentIndex, currentDepth, alpha=None, beta=None):
    if currentIndex >= gameState.getNumAgents():
      currentIndex = 0
      currentDepth += 1

    if currentDepth == self.depth:
        return (self.evaluationFunction(gameState), 'none')

    if currentIndex == 0:
        return self.maximumValue(gameState, currentIndex, currentDepth, alpha, beta)
    else:
        return self.minimumValue(gameState, currentIndex, currentDepth, alpha, beta)

  def minimumValue(self,gameState,index, currentDepth, alpha=None, beta = None):
    moves = gameState.getLegalActions(index)
    bestScore = float('inf')
    bestMove = None
    if not moves: return (self.evaluationFunction(gameState),'none')

    for move in moves:
      if move == Directions.STOP: continue 
      nextState = gameState.generateSuccessor(index,move)
      score,newMove = self.miniMax(nextState,index+1,currentDepth, alpha, beta)
      if score < bestScore:
        bestMove = move
        bestScore = score
      if bestScore < alpha: 
        return (bestScore, bestMove)
      beta = min(beta, bestScore)
    return (bestScore,bestMove)

  def maximumValue(self,gameState,index, currentDepth, alpha=None, beta=None):
    moves = gameState.getLegalActions(index)
    bestScore = float('-inf')
    bestMove = None
    if not moves: return (self.evaluationFunction(gameState),'none')

    for move in moves:
      if move == Directions.STOP: 
        continue 
      nextState = gameState.generateSuccessor(index,move)
      score,newMove = self.miniMax(nextState,index+1,currentDepth,alpha, beta)
      if score > bestScore:
        bestMove = move
        bestScore = score
      if bestScore > beta:
        return (bestScore, bestMove)
      alpha = max(alpha, bestScore)
    return (bestScore,bestMove)

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.treeDepth and self.evaluationFunction
      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"

    retVal = self.miniMax(gameState, 0, 0)
    return retVal[1]

  # the logic will be same as before but we will now use an expected value instead of a minimizer function
  def miniMax(self, gameState, currentIndex, currentDepth):
    # if the index has reached the total number of agents
    # we have finished our cylcle, so we increase the depth

    if currentIndex >= gameState.getNumAgents():
      currentIndex = 0
      currentDepth += 1

    if currentDepth == self.depth:
        return (self.evaluationFunction(gameState), 'none')

    if currentIndex == 0:
        return self.maximumValue(gameState, currentIndex, currentDepth)
    else:
        return self.expectimaxValue(gameState, currentIndex, currentDepth)

  def expectimaxValue(self,gameState,index, currentDepth):
    moves = gameState.getLegalActions(index)
    if not moves: return (self.evaluationFunction(gameState),'none')
    bestScore = 0
    moveProbability = 1.0/len(moves)
    bestMove = None

    for move in moves:
      if move == Directions.STOP: continue 
      nextState = gameState.generateSuccessor(index,move)
      score,newMove = self.miniMax(nextState,index+1,currentDepth)
      bestScore += score * moveProbability
      bestMove = newMove 

    return (bestScore,bestMove)


  def maximumValue(self,gameState,index, currentDepth):
    moves = gameState.getLegalActions(index)
    if not moves: return (self.evaluationFunction(gameState),'none')

    bestScore = float('-inf')

    bestMove = None
    for move in moves:
      if move == Directions.STOP: 
        continue 
      nextState = gameState.generateSuccessor(index,move)
      score,newMove = self.miniMax(nextState,index+1,currentDepth)
      if score > bestScore:
        bestMove = move
        bestScore = score
    
    return (bestScore,bestMove)



def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    DESCRIPTION: This function takes the number of food and subtracts frome the current score
                 in order to guide the pacman into getting the the very last food. (The score increases 
                 as he eats more). If we encouter a ghost in our our nextState, we will return -200, to
                 stay as far away as possible. Otherwise, we will subtract the reciprical of the closest
                 ghost, so that as the ghost gets farther the score goes up. Of course there is a higher
                 weight attached to the ghosts, as we do not want the pacman to die.
  """
  "*** YOUR CODE HERE ***"


  numOfFood = currentGameState.getNumFood()
  currentPositon = currentGameState.getPacmanPosition()
  newFood = currentGameState.getFood()
  newCapsule = currentGameState.getCapsules()

  numCapsuleLeft = len(newCapsule)
  scaredGhost = []
  activeGhost = []
  closestActiveGhostScore = float("inf")
  closestScaredGhostScore = 0
  foodListPacman = newFood.asList()

  foodListPacman.sort(lambda x,y: util.manhattanDistance(currentPositon, x)-util.manhattanDistance(currentPositon, y))
  if foodListPacman:
    nearestFoodScore= util.manhattanDistance(currentPositon, foodListPacman[0])
  else:
    nearestFoodScore = 0

  newGhostStates = currentGameState.getGhostStates()
  ghostPositions=[(int(Ghost.getPosition()[0]),int(Ghost.getPosition()[1])) for Ghost in newGhostStates]
  for Ghost in newGhostStates:
    if(Ghost.scaredTimer):
      scaredGhost.append(Ghost.getPosition())
    else:
      activeGhost.append(Ghost.getPosition())

  activeGhost.sort(lambda x,y: util.manhattanDistance(currentPositon, x)-util.manhattanDistance(currentPositon, y))
  scaredGhost.sort(lambda x,y: util.manhattanDistance(currentPositon, x)-util.manhattanDistance(currentPositon, y))
  if scaredGhost:
    closestScaredGhostScore = util.manhattanDistance(currentPositon, scaredGhost[0])
  else:
    closestScaredGhostScore = 0
  if activeGhost:
    closestActiveGhostScore=util.manhattanDistance(currentPositon, activeGhost[0])
  else:
    closestActiveGhostScore = float("inf")
  
  if closestActiveGhostScore == 0:
    return -200

  return currentGameState.getScore() - (25 * numCapsuleLeft) - (1.5 * nearestFoodScore) - (4 * numOfFood) - (3 * closestScaredGhostScore) - (2 * 1.0/closestActiveGhostScore)
  
# Abbreviation
better = betterEvaluationFunction


