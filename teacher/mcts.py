# -*- coding: utf-8 -*-
"""
A pure implementation of the Monte Carlo Tree Search (MCTS)
@author: Junxiao Song
"""

import numpy as np
import copy
from operator import itemgetter

class TreeNode(object):
    """A node in the MCTS tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p=0.25):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_dict):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for k, action in action_dict.items():
            if k not in self._children:
                self._children[k] = TreeNode(self)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

### Changed this to make a one-player game
    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(leaf_value)
            # self._parent.update_recursive(-leaf_value)

        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """A simple implementation of Monte Carlo Tree Search."""

    def __init__(self, reward_fn, c_puct=5, n_playout=10000):
        """
        reward_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._reward_fn = reward_fn
        self._c_puct = c_puct
        self._n_playout = n_playout


    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while(1):
            if node.is_leaf():

                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            state.do_move(action)

        # action_probs, _ = self._policy(state)
        # Check for end of game
        end = state.game_end()
        if not end:
            node.expand(state.action_dict)

##Change for our reward landscape
        # Evaluate the leaf node by random rollout
        # print(state.state)
        # if (0 <= state.state[0]) & (state.state[0] < 100):
        #     if (0 <= state.state[1]) & (state.state[1] < 100):  
        #         leaf_value = self._reward_fn[state.state[0]][state.state[1]]
        # else:
        #     leaf_value = -1
        leaf_value = state.get_reward()
        # print('leaf_value: ', leaf_value)

##Changed for one player
        # Update value and visit count of nodes in this traversal.
        # node.update_recursive(-leaf_value)
        node.update_recursive(leaf_value)


    def get_move(self, state):
        """Runs all playouts sequentially and returns the most visited action.
        state: the current game state
        Return: the selected action
        """
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        return max(self._root._children.items(),
                   key=lambda act_node: act_node[1]._n_visits)[0]

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """AI player based on MCTS"""
    def __init__(self, reward_fn, c_puct=6, n_playout=100):
        self.mcts = MCTS(reward_fn, c_puct, n_playout)


    def reset_player(self):
        self.mcts.update_with_move(-1)


    def get_action(self, board):
        move = self.mcts.get_move(board)
        self.mcts.update_with_move(move)
        return move


    def __str__(self):
        return "MCTS {}".format(self.player)