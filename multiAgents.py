# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

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
      
      Notes:
        - A capable reflex agent will have to consider both food locations and ghost locations to perform well.
        - The reciprocal of important values (distance to food) are used as features rather than just the values themselves.
        - The evaluation function evaluates state-action pairs only.
        - Default ghosts used are random.
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
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        "*** YOUR CODE HERE ***"
        evaluateScore=0.0
        ## lists to record distance between food & Pacman, ghost & Pacman
        foodDistance=[]
        ghostDistance=[]
        ## current food positions
        currentFood=currentGameState.getFood()
        currentFoodList=currentFood.asList()
        for food in currentFoodList:
            foodDistance.append(manhattanDistance(newPos, food))
        foodScore=min(foodDistance)# closest food 
      
        # ghost
        newGhostPosition=[]
        for newGhost in newGhostStates:
            newGhostPosition.append(newGhost.getPosition())
        if len(newGhostPosition)==0:
            ghostScore=0           
        else:
            for ghost in newGhostPosition:
                ghostDistance.append(manhattanDistance(newPos, ghost))         
            if min(ghostDistance)==0:
                return float("-inf")   ## game over
            else:
                ghostScore=min(ghostDistance)
        if foodScore==0:
            evaluateScore=-1.0/(ghostScore)+1
        else:
            evaluateScore=-1.0/(ghostScore)*2+1.0/foodScore   
        
        return evaluateScore
    
      
        
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
        
        ## Pacman go first
        ## return argmax 
        self.countAgents = gameState.getNumAgents()
        pacmanAction=self.Max_Value(gameState, 0,0)
        return pacmanAction[1]  
        
  ## def getValue(gamestate) function:     
  ##     if the state is a terminal state: return the state's utility
  ##     if the next agent is a MAX: return Max_Value(gameState)
  ##     if the next agent is a MIN: return Min_Value(gameState)
    def getValue(self,gameState,agentIndex,currentDepth):
        "Terminal Test"
        if currentDepth>=self.countAgents*self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if agentIndex==0:    ## agentindex==0 is pacman
            return self.Max_Value(gameState, agentIndex, currentDepth)[0]   ## [0] is bestValue,[1] is bestAction
        else:
            return self.Min_Value(gameState, agentIndex, currentDepth)
            
  ## def Max_Value(gamestate) function:     
  ##     initialize v=-inf
  ##     for each successor of gamestate:
  ##        v=max(v,getValue(successor))
  ##     return v
  
    def Max_Value(self,gameState,agentIndex,currentDepth):  
        legalActions=gameState.getLegalActions(agentIndex)
        bestMaxValue=float('-inf')
        bestAction='Stop'
        for action in legalActions:
            successorState=gameState.generateSuccessor(agentIndex,action)
            successorDepth=currentDepth+1
            value=self.getValue(successorState,successorDepth % self.countAgents, successorDepth)
            if value>bestMaxValue:
                bestMaxValue=value
                bestAction=action
        return [bestMaxValue,bestAction]    # bestMaxValue-> getValue(), bestAction-> getAction
        
  ## def Min_Value(gamestate) function:     
  ##     initialize v=inf
  ##     for each successor of gamestate:
  ##        v=min(v,getValue(successor))
  ##     return v
  
    def Min_Value(self,gameState,agentIndex,currentDepth):  
        legalActions=gameState.getLegalActions(agentIndex)
        bestMinValue=float("inf")
        bestAction='Stop'
     #   value=(bestMinValue,bestAction)
        for action in legalActions:
            successorState=gameState.generateSuccessor(agentIndex,action)
            successorDepth=currentDepth+1
            value=self.getValue(successorState,successorDepth % self.countAgents, successorDepth)
            if value<bestMinValue:
                bestMinValue=value
                bestAction=action

        return bestMinValue
    
  
 
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        ## Pacman go first
        ## initilize alpha, beta
        alpha=float('-inf')
        beta=float('inf')
        self.countAgents = gameState.getNumAgents()
        pacmanAction=self.Max_Value(gameState, 0,0,alpha,beta)
        return pacmanAction[1]  
        
        
  ## def getValue(gamestate,..,alpha,beta) function:     
  ##     if the state is a terminal state: return the state's utility
  ##     if the next agent is a MAX: return Max_Value(gameState,..,alpha,beta)
  ##     if the next agent is a MIN: return Min_Value(gameState,..,alpha,beta)
    def getValue(self,gameState,agentIndex,currentDepth,alpha,beta):
        self.countAgents = gameState.getNumAgents()
        "Terminal Test"
        if currentDepth>=self.countAgents*self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if agentIndex==0:
            return self.Max_Value(gameState,agentIndex,currentDepth,alpha,beta)[0]   ## [0] is bestValue,[1] is bestAction
        else:
            return self.Min_Value(gameState,agentIndex,currentDepth,alpha,beta)[0] 


  ## def Max_Value(gamestate,..,alpha,beta) function:     
  ##     initialize v=-inf
  ##     for each successor of gamestate:
  ##        v=max(v,getValue(successor,alpha,beta))
  ##        if v>beta: return v
  ##        alpha=max(alpha,v) 
  ##     return v
  
    def Max_Value(self,gameState,agentIndex,currentDepth,alpha,beta):  
        legalActions=gameState.getLegalActions(agentIndex)
        bestMaxValue=float('-inf')
        bestAction='Stop'
        for action in legalActions:
            successorState=gameState.generateSuccessor(agentIndex,action)
            successorDepth=currentDepth+1
            value=self.getValue(successorState,successorDepth%self.countAgents, successorDepth,alpha,beta)
            if value>bestMaxValue:
                bestMaxValue=value
                bestAction=action
                if bestMaxValue>beta:
                    return [bestMaxValue,bestAction]
                alpha=max(alpha, bestMaxValue)
        return [bestMaxValue,bestAction]    # bestMaxValue-> getValue(), bestAction-> getAction


  ## def Min_Value(gamestate,...,alpha,beta) function:     
  ##     initialize v=inf
  ##     for each successor of gamestate:
  ##        v=min(v,getValue(successor,alpha,beta))
  ##        if v<alpha:  return v
  ##        beta=min(beta,v)
  ##     return v
  
    def Min_Value(self,gameState,agentIndex,currentDepth,alpha,beta):  
        legalActions=gameState.getLegalActions(agentIndex)
        bestMinValue=float("inf")
        bestAction='Stop'
        for action in legalActions:
            successorState=gameState.generateSuccessor(agentIndex,action)
            successorDepth=currentDepth+1
            value=self.getValue(successorState,successorDepth%self.countAgents,successorDepth,alpha,beta)
            if value<bestMinValue:
                bestMinValue=value
                bestAction=action
                if bestMinValue<alpha:
                    return [bestMinValue,bestAction]
                beta=min(beta,bestMinValue)

        return [bestMinValue,bestAction]

 
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        ## Pacman go first

        self.countAgents = gameState.getNumAgents()
        pacmanAction=self.Max_Value(gameState, 0,0)
        return pacmanAction[1]  


  ## def getValue(gamestate,..) function:     
  ##     if the state is a terminal state: return the state's utility
  ##     if the next agent is a MAX: return Max_Value(gameState,..,alpha,beta)
  ##     if the next agent is a EXPECTED: return Expcted_Value(gameState,..,alpha,beta)
    def getValue(self,gameState,agentIndex,currentDepth):
        self.countAgents = gameState.getNumAgents()
        "Terminal Test"
        if currentDepth>=self.countAgents*self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if agentIndex==0:
            return self.Max_Value(gameState,agentIndex,currentDepth)[0]   ## [0] is bestValue,[1] is bestAction
        else:
            return self.Expected_Value(gameState,agentIndex,currentDepth)  ## stochastic actions (only utility value needed)


  ## def Max_Value(gamestate,..,alpha,beta) function:     
  ##     initialize v=-inf
  ##     for each successor of gamestate:
  ##        v=max(v,getValue(successor))
  ##     return v
  
    def Max_Value(self,gameState,agentIndex,currentDepth):  
        legalActions=gameState.getLegalActions(agentIndex)
        bestMaxValue=float('-inf')
        bestAction='Stop'
        for action in legalActions:
            successorState=gameState.generateSuccessor(agentIndex,action)
            successorDepth=currentDepth+1
            value=self.getValue(successorState,successorDepth% self.countAgents, successorDepth)
            if value>bestMaxValue:
                bestMaxValue=value
                bestAction=action
        return [bestMaxValue,bestAction]    # bestMaxValue-> getValue(), bestAction-> getAction


  ## def Expected_Value(gamestate,...) function:     
  ##     initialize v=[]
  ##     for each successor of gamestate:
  ##        value=sum(probabilities*utilities)
  ##     return value
  
    def Expected_Value(self,gameState,agentIndex,currentDepth):  
        legalActions=gameState.getLegalActions(agentIndex)
        expectedUtility=[]
        for action in legalActions:
            successorState=gameState.generateSuccessor(agentIndex,action)
            successorDepth=currentDepth+1
            value=self.getValue(successorState,successorDepth% self.countAgents, successorDepth)
            expectedUtility.append(value)
        expectedValue=sum(expectedUtility)*1.0/len(expectedUtility)

        return expectedValue


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    currPacState=currentGameState.getPacmanState()
    currPacPos=currentGameState.getPacmanPosition()
    currScore=scoreEvaluationFunction(currentGameState)
    foodDistance=[]
    ghostDistance=[]
    countScared=0.0
    evaluateScore=0.0
    foodScore=0.0
    ghostScore=0.0
    
    
    ## current food positions
    currentFood=currentGameState.getFood()
    currentFoodList=currentFood.asList()
    for food in currentFoodList:
        foodDist=manhattanDistance(currPacPos, food)
        if foodDist==0:
            foodDistance.append(0)
            foodScore=0.0
        else:
            foodDistance.append(1.0/foodDist)
            foodScore=max(foodDistance)
        

        # ghost
    currGhostStates = currentGameState.getGhostStates()
  ##      newScaredTimes = [ghostState.scaredTimer for ghostState in currGhostStates]
    currGhostPosition=[]
    for ghostState in currGhostStates:
        if ghostState.scaredTimer ==0:
            countScared+=1
        currGhostPosition.append(ghostState.getPosition())
        for ghost in currGhostPosition:
            ghostDist=manhattanDistance(currPacPos, ghost)
            if ghostDist==0:
                ghostDistance.append(0)
            else:
                ghostDistance.append(ghostDist)
    ghostScore=(-1)*min(ghostDistance)
    evaluateScore= currScore+foodScore+2.0*ghostScore-3.0*countScared
    
    return evaluateScore

    
# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

