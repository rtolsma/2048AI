### 2048 AI

I was bored today after the superbowl, and decided to try my luck at an 2048 AI implementation. Currently, the only the game logic and a vanilla Q-Learning Algorithm is being used, and the scoring is relatively poor.

### Potential Improvements

1) Implement some form of Expectimax
2) Instead of Expectimax use a Monte Carlo Tree Search Algorithm
3) Finish implementing episode replays (stochastic games won't rlly learn as much tho)
4) Optimize computational efficiency using bit hacks instead of matrices to store the board and complete transformations
5) Experiment with different NN Architectures/ Hyperparameter tuning (once loss function actually looks reasonable lol)
6) Realize that Deep Learning is most definitely not the right choice for this game and just handcraft an evaluator function... lol
7) More Layers :)