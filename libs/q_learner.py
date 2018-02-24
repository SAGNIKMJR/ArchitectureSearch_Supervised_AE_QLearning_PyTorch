import math
import numpy as np
import pandas as pd
import os
from operator import itemgetter
import cnn
import state_enumerator as se
from state_string_utils import StateStringUtils
import train_

class QValues:
    def __init__(self):
        self.q = {}

    def save_to_csv(self, q_csv_path):
        start_layer_type = []
        start_layer_depth = []
        start_filter_depth = []
        start_filter_size = []
        start_stride = []
        start_image_size = []
        start_fc_size = []
        start_terminate = []
        end_layer_type = []
        end_layer_depth = []
        end_filter_depth = []
        end_filter_size = []
        end_stride = []
        end_image_size = []
        end_fc_size = []
        end_terminate = []
        utility = []
        for start_state_list in self.q.keys():
            start_state = se.State(state_list=start_state_list)
            for to_state_ix in range(len(self.q[start_state_list]['actions'])):
                to_state = se.State(state_list=self.q[start_state_list]['actions'][to_state_ix])
                utility.append(self.q[start_state_list]['utilities'][to_state_ix])
                start_layer_type.append(start_state.layer_type)
                start_layer_depth.append(start_state.layer_depth)
                start_filter_depth.append(start_state.filter_depth)
                start_filter_size.append(start_state.filter_size)
                start_stride.append(start_state.stride)
                start_image_size.append(start_state.image_size)
                start_fc_size.append(start_state.fc_size)
                start_terminate.append(start_state.terminate)
                end_layer_type.append(to_state.layer_type)
                end_layer_depth.append(to_state.layer_depth)
                end_filter_depth.append(to_state.filter_depth)
                end_filter_size.append(to_state.filter_size)
                end_stride.append(to_state.stride)
                end_image_size.append(to_state.image_size)
                end_fc_size.append(to_state.fc_size)
                end_terminate.append(to_state.terminate)

        q_csv = pd.DataFrame({'start_layer_type' : start_layer_type,
                              'start_layer_depth' : start_layer_depth,
                              'start_filter_depth' : start_filter_depth,
                              'start_filter_size' : start_filter_size,
                              'start_stride' : start_stride,
                              'start_image_size' : start_image_size,
                              'start_fc_size' : start_fc_size,
                              'start_terminate' : start_terminate,
                              'end_layer_type' : end_layer_type,
                              'end_layer_depth' : end_layer_depth,
                              'end_filter_depth' : end_filter_depth,
                              'end_filter_size' : end_filter_size,
                              'end_stride' : end_stride,
                              'end_image_size' : end_image_size,
                              'end_fc_size' : end_fc_size,
                              'end_terminate' : end_terminate,
                              'utility' : utility})
        q_csv.to_csv(q_csv_path, index=False)

class QLearner:
    def __init__(self,
                 state_space_parameters, 
                 epsilon,
                 data_path = './MNIST', 
                 state=None,
                 qstore=None,   


                 # replay_dictionary = pd.DataFrame(columns=['net',
                 #                                         'accuracy_best_val',
                 #                                         'accuracy_last_val',       # uncomment while actual training 
                 #                                         'epsilon',
                 #                                         'train_flag'])):

                 replay_dictionary = pd.DataFrame(columns=['net',
                                                         'loss_inverse',
                                                         'loss',
                                                         'epsilon',
                                                         'computeLoss_flag'])):

        self.state_list = []
        self.data_path = data_path
        # self.bucketed_state_list = []
        self.state_space_parameters = state_space_parameters

        self.enum = se.StateEnumerator(state_space_parameters)    
        self.stringutils = StateStringUtils(state_space_parameters)

        self.state = se.State('start', 0, 1, 0, 0, state_space_parameters.image_size, 0, 0) if not state else state
        # self.bucketed_state = self.enum.bucket_state(self.state)

        self.qstore = QValues() if not qstore else qstore
        self.replay_dictionary = replay_dictionary

        self.epsilon=epsilon 

    def generate_net(self, epsilon = None): 
        if epsilon != None:
          self.epsilon = epsilon 
        self._reset_for_new_walk()
        # bucketed_state_list, state_list = self._run_agent()
        state_list = self._run_agent()
        # bucketed_state_list = self.stringutils.add_drop_out_states(bucketed_state_list)

        ''' uncomment to include dropout states '''
        # state_list = self.stringutils.add_drop_out_states(state_list)             # uncomment to include dropout states
        ''' uncommenting ends here '''
        # net_string = self.stringutils.state_list_to_string(bucketed_state_list)
        # print('Before training:')
        # for state in self.state_list:
        #   print('{} {} {} {} {} {} {} {}'.format(state.layer_type,state.layer_depth, state.filter_depth,\
        #     state.filter_size, state.stride, state.image_size, state.fc_size, state.terminate))
        # print(state_list == self.state_list)
        net_string = self.stringutils.state_list_to_string(state_list)
        ''' uncomment while training '''
        # train_flag = True
        ''' uncommenting ends '''
        computeLoss_flag = True
        if net_string in self.replay_dictionary['net'].values:

          ''' Uncomment while training '''
          # acc_best_val = self.replay_dictionary[self.replay_dictionary['net']==net_string]['accuracy_best_val'].values[0]
          # acc_last_val = self.replay_dictionary[self.replay_dictionary['net']==net_string]['accuracy_last_val'].values[0]
          # train_flag = self.replay_dictionary[self.replay_dictionary['net']==net_string]['train_flag'].values[0]
          ''' Uncommenting ends '''

          loss_inverse = self.replay_dictionary[self.replay_dictionary['net']==net_string]['loss_inverse'].values[0]
          loss = self.replay_dictionary[self.replay_dictionary['net']==net_string]['loss'].values[0]
          computeLoss_flag = self.replay_dictionary[self.replay_dictionary['net']==net_string]['computeLoss_flag'].values[0]
        
        else:
          ''' Uncomment while training '''
          # acc_best_val, acc_last_val, train_flag = train_.train_val_net(state_list, \
          #                                                               self.state_space_parameters, \
          #                                                               self.data_path)
          ''' Uncommenting ends '''
          state_list2 = state_list
          loss_inverse, loss, computeLoss_flag = train_.train_val_net(state_list2, \
                                                                        self.state_space_parameters, \
                                                                        self.data_path)
          # loss_inverse, loss, computeLoss_flag = 1000, 0.1, True
          self.replay_dictionary = self.replay_dictionary.append(pd.DataFrame([[net_string, loss_inverse, loss, \
                                        self.epsilon, computeLoss_flag]], columns=['net', 'loss_inverse', \
                                        'loss', 'epsilon', 'computeLoss_flag']), ignore_index = True)
          if computeLoss_flag == True:
            print('Inverse loss:{} and loss:{}'\
              .format(loss_inverse, loss))

    def _reset_for_new_walk(self):

        self.state_list = []
        self.bucketed_state_list = []
        self.state = se.State('start', 0, 1, 0, 0, self.state_space_parameters.image_size, 0, 0)
        # self.bucketed_state = self.enum.bucket_state(self.state)

    def _run_agent(self):
        while self.state.terminate == 0:
            self._transition_q_learning()

        # return self.bucketed_state_list, self.state_list
        return self.state_list

    def _transition_q_learning(self):
        # if self.bucketed_state.as_tuple() not in self.qstore.q:
        #     self.enum.enumerate_state(self.bucketed_state, self.qstore.q)
        if self.state.as_tuple() not in self.qstore.q:
            self.enum.enumerate_state(self.state, self.qstore.q)        

        # action_values = self.qstore.q[self.bucketed_state.as_tuple()]
        action_values = self.qstore.q[self.state.as_tuple()]

        if np.random.random() < self.epsilon:
            action = se.State(state_list=action_values['actions'][np.random.randint(len(action_values['actions']))])
        else:
            max_q_value = max(action_values['utilities'])
            max_q_indexes = [i for i in range(len(action_values['actions'])) if action_values['utilities'][i]==max_q_value]
            max_actions = [action_values['actions'][i] for i in max_q_indexes]
            action = se.State(state_list=max_actions[np.random.randint(len(max_actions))])

        self.state = self.enum.state_action_transition(self.state, action)
        # self.bucketed_state = self.enum.bucket_state(self.state)
        self._post_transition_updates()

    def _post_transition_updates(self):
        # bucketed_state = self.bucketed_state.copy()
        non_bucketed_state = self.state.copy()
        # self.bucketed_state_list.append(bucketed_state)
        self.state_list.append(non_bucketed_state)

    def sample_replay_for_update(self):
        for i in range(self.state_space_parameters.replay_number):
            net = np.random.choice(self.replay_dictionary['net'])

            ''' Uncomment for training '''
            # accuracy_best_val = self.replay_dictionary[self.replay_dictionary['net'] == net]['accuracy_best_val'].values[0]
            # accuracy_last_val = self.replay_dictionary[self.replay_dictionary['net'] == net]['accuracy_last_val'].values[0]
            # train_flag = self.replay_dictionary[self.replay_dictionary['net'] == net]['train_flag'].values[0]
            ''' Uncommenting ends '''
            loss_inverse = self.replay_dictionary[self.replay_dictionary['net'] == net]['loss_inverse'].values[0]
            loss = self.replay_dictionary[self.replay_dictionary['net'] == net]['loss'].values[0]
            computeLoss_flag = self.replay_dictionary[self.replay_dictionary['net'] == net]['computeLoss_flag'].values[0]

            state_list = self.stringutils.convert_model_string_to_states(cnn.parse('net', net))
            # print('During update:')
            # for state in state_list:
            #   print('{} {} {} {} {} {} {} {}'.format(state.layer_type,state.layer_depth, state.filter_depth,\
            #     state.filter_size, state.stride, state.image_size, state.fc_size, state.terminate))
            # print(state_list)
            ''' Uncomment while training '''
            # state_list = self.stringutils.remove_drop_out_states(state_list)
            ''' Uncommenting ends '''

            # state_list = [self.enum.bucket_state(state) for state in state_list]

            ''' Uncomment while training '''
            # if train_flag == True:
            #   self.update_q_value_sequence(state_list, self.accuracy_to_reward(accuracy_best_val))
            ''' Uncommenting ends '''
            if computeLoss_flag == True:
              self.update_q_value_sequence(state_list, self.accuracy_to_reward(loss_inverse))


    def accuracy_to_reward(self, acc):
        return acc

    def update_q_value_sequence(self, states, termination_reward):
        self._update_q_value(states[-2], states[-1], termination_reward)
        for i in reversed(range(len(states) - 2)):
            # self._update_q_value(states[i], states[i+1], 0)
            self._update_q_value(states[i], states[i+1], termination_reward)

    def _update_q_value(self, start_state, to_state, reward):
        if start_state.as_tuple() not in self.qstore.q:
            self.enum.enumerate_state(start_state, self.qstore.q)
        if to_state.as_tuple() not in self.qstore.q:
            self.enum.enumerate_state(to_state, self.qstore.q)

        actions = self.qstore.q[start_state.as_tuple()]['actions']
        values = self.qstore.q[start_state.as_tuple()]['utilities']

        max_over_next_states = max(self.qstore.q[to_state.as_tuple()]['utilities']) if to_state.terminate != 1 else 0

        action_between_states = self.enum.transition_to_action(start_state, to_state).as_tuple()
        values[actions.index(action_between_states)] = values[actions.index(action_between_states)] + \
                                                self.state_space_parameters.learning_rate * (reward + \
                                                self.state_space_parameters.discount_factor * max_over_next_states\
                                                 - values[actions.index(action_between_states)])

        self.qstore.q[start_state.as_tuple()] = {'actions': actions, 'utilities': values}

    




