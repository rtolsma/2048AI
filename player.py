# NN stuffs
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.cuda as cuda
import torch

from tensorboardX import SummaryWriter
import math
import random
import os

#2048 Stuff
from game import *

TWriter = SummaryWriter()

class Player:

    def __init__(self, discount, model, verbose=False):
        self.verbose = verbose
        self.discount = discount
        self.model = model
        self.numiters = 0
        self.gameHistory = []
        self.gamesPlayed = 0
        self.replayIters = 0
        
    def playGame(self):
        state = initialState()
        self.playHistory = []
        self.nummoves = 0
        while(not isEnd(state)):

            possibleActions = actions(state)
            actionChoice = self.chooseAction(state, possibleActions)
            newState = successor(state, actionChoice)

            if isEnd(newState):
                reward = 0 #-2**20
            else:
                reward = score(newState) - score(state)
            
            ## needs this order...lol bad code
            self.playHistory.append((state, actionChoice, newState, reward))
            self.incorporateFeedback(newState, reward)
            if self.verbose:
                pretty(state)
            
            self.nummoves += 1
            state = newState


        print("FINAL STATE TOOK: ", self.nummoves, " MOVES")
        TWriter.add_scalar('data/score', newState[1], self.gamesPlayed)
        pretty(state)
        self.gamesPlayed += 1
        self.gameHistory.append(self.playHistory[:])
            
 

    def getQ(self, state, action):
        vector_features = featureExtractor(state, action)
        output = self.model.predict(vector_features)
        return output  

    def getStepSize(self):
        return 1.0 / self.numiters

    def incorporateFeedback(self, newState, reward):
        lastState, lastAction, _,_ = self.playHistory[-1]
        self.personalFeedback(lastState, lastAction, reward, newState)

    def personalFeedback(self, state, action, reward, newState, replay=False):
        if replay:
            self.replayIters +=1 
        else:
            self.numiters += 1
        vector_features = featureExtractor(state, action)
        # get best action for next state
        nextActions =  actions(newState)
        nextQs = [(self.getQ(newState, a) , a) for a in nextActions]
        nextBestQ = (max(nextQs))[0] if len(nextQs) > 0 else 0
        target = reward + self.discount * nextBestQ
        loss = self.model.update(vector_features, target)
        
        if replay:
            TWriter.add_scalar('replay/loss', loss, self.replayIters)
        else:
            TWriter.add_scalar('play/loss', loss, self.numiters)


    def chooseAction(self, state, actions):
        # naive UCB
        if random.random() < np.sqrt(float(1 / max(self.numiters, 1))):
            chosen = random.choice(actions)
        else:
            # tuple hax
            score, chosen = max([(self.getQ(state,action), action) for action in actions])
        return chosen







### PLAYER MODEL ###


class ModelTest(Player):

    QMODEL_OLD = None
    QMODEL_TRAIN = None
    QMODEL_CRITERION = None


    def __init__(self, verbose=False):

        self.testQModel = self.start()

        
        super().__init__(discount=1, model=self.testQModel, verbose=verbose,)



    def start(self):
        learning_rate = 1e-2 # usually a reasonable val
        weights = self.getNNStructure()
        oldweights = self.getNNStructure()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(weights.parameters(), lr=learning_rate, weight_decay=5e-4)

        if not (ModelTest.QMODEL_OLD == None and ModelTest.QMODEL_TRAIN == None):
            weights = ModelTest.QMODEL_TRAIN
            oldweights = ModelTest.QMODEL_TEST
            criterion = ModelTest.QMODEL_CRITERION
        else:

            # setup gpu computing
            if cuda.is_available():
                weights = weights.cuda()
                oldweights = oldweights.cuda()
                criterion = criterion.cuda()
                optimizer = optim.Adam(weights.parameters(), lr=learning_rate, weight_decay=0)
                #print("cuda'd optimizer")

            self.load(oldweights, optimizer)
            self.load(weights, optimizer)
            ModelTest.QMODEL_TRAIN = weights
            ModelTest.QMODEL_TEST = oldweights
            ModelTest.QMODEL_CRITERION = criterion


        pred, upd = getLambdas(criterion, optimizer, oldweights, weights)
        self.optimizer = optimizer
        self.weights = weights
        self.oldweights = oldweights
        return QModel(weights, pred, upd)

    
    def replayHistory(self):
        for game in self.gameHistory:
            for state, action, newState, reward in game[:-1]:
                
                print('replaying things')
                self.personalFeedback(state, action, newState, reward)


    
    def refresh(self):
        self.save()
        self.model = self.start()
        self.replayIters = 0
        print("REPLAYING HISTORY:", len(self.gameHistory))
        self.replayHistory()
        print("FINISHED REPLAYING HISTORY")


    def save(self, path="./qmodel"):
        
        state = {
            "model": self.weights.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, path)


    def load(self, model, optimizer, path="./qmodel"):

        if os.path.exists(path):
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
    

    ## TODO Readup again on CNN approaches
    def getNNStructure(self):
        '''
        return nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            Flatten(),
            nn.Linear(16*16, 1)
        )
        '''
        return nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5,  padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            Flatten(),
            nn.Linear(16*32, 1),
        )
        


#### QMODEL ####

class QModel:

    def __init__(self, model, predict_lambda, update_lambda):
        self.model = model
        self.predict_lambda = predict_lambda
        self.update_lambda = update_lambda
        self.num_iters = 0

    
    def predict(self, features):
        features = torch.tensor(np.flip(features, axis=0).copy()).float()
        return self.predict_lambda(self.model, features)
    
    def update(self, features, target):
        self.num_iters += 1
        features = torch.tensor(np.flip(features, axis=0).copy()).float()
        loss = self.update_lambda(self.model, features, target)
        return loss

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(-1)

class Structure(nn.Module):
    def __init__(self):
        super(Structure, self).__init__()
    def forward(self, x):
        return x.view(-1, 4, 4)

class Unflatten(nn.Module):
    def __init__(self):
        super(Unflatten, self).__init__()
    def forward(self, x):
        return x.view(1,-1, -1)

# generates update lambdas with correct optimizer+criterion+model for QModel class
def getLambdas(criterion, optimizer, oldweights, weights):
        def pred(weights, features):
            with torch.no_grad():
                if cuda.is_available():
                    features = features.cuda()
                score = oldweights(features)
            return score

        def upd(weights, features, target):
            t = torch.tensor(float(target))
            if cuda.is_available():
                features = features.cuda()
                t = t.cuda()
            
            current_estimate = weights(features)
            loss = criterion(current_estimate, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return loss
        return pred, upd