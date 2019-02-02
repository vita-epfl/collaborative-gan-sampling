
import numpy as np
import copy
from operator import itemgetter
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class TeacherMCTS(object):

    def __init__(self, scale=2.0, centers=None, xres=100, yres=100, look_ahead=20):
        '''Current Configuration sets the Reward Function expanse same as (-scale,scale)'''
        '''This function is analogous to environment configuration'''
        self.scale = scale
        self.centers = [
                        (1, 0),
                        (-1, 0),
                        (0, 1),
                        (0, -1),
                        (1. / np.sqrt(2), 1. / np.sqrt(2)),
                        (1. / np.sqrt(2), -1. / np.sqrt(2)),
                        (-1. / np.sqrt(2), 1. / np.sqrt(2)),
                        (-1. / np.sqrt(2), -1. / np.sqrt(2))
                        ]
        self.centers = [((scale / 1.414) * x, (scale/1.414) * y) for x, y in self.centers]
        if centers is not None:
            self.centers = centers
        self.xlim = (-scale, scale)
        self.ylim = (-scale, scale)
        self.xres = xres
        self.yres = yres
        self.x_start = self.xlim[0]
        self.x_end = self.xlim[1]
        self.y_start = self.ylim[0]
        self.y_end = self.ylim[1]
        self.look_ahead = look_ahead

    def set_env(self, sess, gan):
        self.sess = sess
        self.gan = gan

    def init_MCTS(self, reward_path=None, init_gs_state=[0.5,0.5], c_puct=6.0, n_playout=500, step_size=2, thresh=3):
        '''Initializes the parameters of MCTS'''
        ### MCTS Hyper Parameters
        # init_gs_state - sets the initial point on the board. A sample of the generator
        # c_puct - the rate of exploration 
        # n_playout - number of steps to look ahead before finalizing the next move
        # step_size - distance to advance at each move
        # thresh - stopping criterion when near a target state (mode of Discriminator)

        ## Necessary imports to init the MCTS
        from mcts import TreeNode, MCTS, MCTSPlayer
        from Game import Board
        from utils import print_board, reward_function

        ### MCTS Initializing Variables
        #Possible directions to move
        action_list = [[step_size, 0], [0, step_size], [-step_size, 0], [0,-step_size]]
        action_dict = {0 : [step_size,0], 1 : [0,step_size], 2 : [-step_size, 0], 3 : [0,-step_size]}
        #The modes of Discriminator
        target_states = [self.groundtruth_to_pixel(center) for center in self.centers]
        
        #Reward function at each position. Equivalent to score of D
        if reward_path is not None:
            import h5py
            h5f = h5py.File(reward_path, 'r')
            reward_fn = h5f['reward_array'][:]
            h5f.close()
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.imshow(reward_fn)
            plt.show()
        else:   
            reward_fn = reward_function(self.centers, self.scale, self.xlim, self.ylim, self.xres, self.yres)
        
        #Converts the coordinates from GAN to MCTS Framework
        init_state = np.array(self.groundtruth_to_pixel(init_gs_state))
        # import pdb; pdb.set_trace()

        ## Initializing the Board and MCTS Player
        #The baord deinfes the states, actions and rewards of the game
        #The MCTSPlayer handles only the Tree Search. It is agnostic to the task at hand
        self.board = Board(init_state, action_list, action_dict, reward_fn, target_states, thresh)
        self.mcts_player = MCTSPlayer(reward_fn, c_puct, n_playout)
        self.print = print_board

    ## Converting between the MCTS Envt and GAN environment
    def groundtruth_to_pixel(self, gxy):
        gx = gxy[0]
        gy = gxy[1]
        px = int(self.xres*((gx - self.x_start)/(self.x_end - self.x_start)))
        py = int(self.yres*((gy - self.y_start)/(self.y_end - self.y_start))) 
        return [px, self.yres - py]

    def pixel_to_groundtruth(self, pxy):
        px = pxy[0]
        py = self.yres - pxy[1]
        gx = (float(px)/self.xres)*(self.x_end - self.x_start) + self.x_start
        gy = (float(py)/self.yres)*(self.y_end - self.y_start) + self.y_start
        return [gx, gy]

    ## MAIN FUNCTION
    def manipulate_gradient(self, generated_samples, fake_logit, fake_grad):
        '''Takes in the samples and returns the look-ahead gradient'''
        #look_ahead - number of moves to be played before stopping the search

        ##Samples of GANs are states of the board
        state_list = generated_samples      ##The Input
        grad_list = []                      ##The Output
        states_traj = []                    ##For Visualization
        main_board = self.board
        mcts_player = self.mcts_player
        count = 0
        for s in state_list:
            # count += 1
            # print("Count: ", count)
            states_traj = []
            init_s = s
            # print("Init GT State: ", s)
            main_board.reset_state(np.array(self.groundtruth_to_pixel(s)))
            # print("Init Board State: ", np.array(self.groundtruth_to_pixel(s)))
            if not main_board.game_end():
                for t in range(self.look_ahead):
                    move = mcts_player.get_action(main_board)
                    mcts_player.mcts.update_with_move(move)
                    states_traj.append(main_board.state)
                    # import pdb; pdb.set_trace()
                    main_board.do_move(move)
                    # self.print(main_board.rewards, states_traj)
                    # print("Reward of new state: ", main_board.get_reward())
                    if main_board.game_end():
                        # print("Reached Optimal Node")
                        break
            # else:
                # print("Started from Optimal Node")


            ##For Debugging Purposes
            # print("States trajectory")
            # print(states_traj)

            # print("GAN Coordinates")
            final_s = self.pixel_to_groundtruth(main_board.state)
            # print(self.pixel_to_groundtruth(main_board.state))
            grad_list.append([j - i for i, j in zip(final_s, init_s)])

        # print(grad_list)
        return np.array(grad_list)

def main():
    look_ahead = 20
    ##Initializing Teacher Envt and its components: MCTSPlayer and the Board
    envt = TeacherMCTS(look_ahead=look_ahead)
    envt.init_MCTS()
    sample_list = [[1.8, -1.8],[0.3, 0.7],[1.2, 0.2],[-0.3, 0.3]]
    grad_list = envt.manipulate_gradient(0, 0, sample_list)
    print(grad_list)


if __name__ == '__main__':
    main()