import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p = prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p



    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO 

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        
        
        # Step 1. Initialize variables
        
        obs_len = len(input_observation_states) # length of observed sequence
        num_hidden_states = len(self.hidden_states) # number of hidden states

        
        # Step 2. Calculate probabilities

        first_obs = input_observation_states[0] # get first observation
        first_obs_index = self.observation_states_dict[first_obs] # get index of first observation
        hidden_state_probs = self.prior_p * self.emission_p[:, first_obs_index] # calculate probability of first observation for each hidden state prior

        for i in range(1, obs_len):
            obs_state = input_observation_states[i] # get observed state at position i
            obs_state_index = self.observation_states_dict[obs_state] # get index of observed state 
            
            next_hidden_probs = np.zeros(num_hidden_states) # temporary variable to store probability for current observed state
            for j in range(num_hidden_states):
                # below, we find the total probability of all paths to hidden state j and multiply by emission probability for the current observed state at position i
                next_hidden_probs[j] = np.dot(hidden_state_probs, self.transition_p[:,j]) * self.emission_p[j][obs_state_index]
            hidden_state_probs = next_hidden_probs # update probabilities for current observed state


        # Step 3. Return final probability 
        return np.sum(hidden_state_probs) # return sum of probabilities for all paths to final observed state



    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """        
        
        # Step 1. Initialize variables
        
        obs_len = len(decode_observation_states) # length of observed sequence
        num_hidden_states = len(self.hidden_states) # number of hidden states

        #store probabilities of hidden state at each step 
        viterbi_table = np.zeros((obs_len, num_hidden_states))
        #store best path for traceback
        best_path = np.zeros(obs_len, dtype=int)
        # backpointer[time, state] stores WHICH previous state led to this max probability, from gemini
        backpointer = np.zeros((obs_len, num_hidden_states), dtype=int)         
       
        # Step 2. Calculate Probabilities
        first_obs = decode_observation_states[0] # get first observation
        first_obs_index = self.observation_states_dict[first_obs] # get index of first observation
        viterbi_table[0, :] = self.prior_p * self.emission_p[:, first_obs_index] # calculate probability of first observation for each hidden state prior

        for i in range(1, obs_len):
            obs_state = decode_observation_states[i] # get observed state at position i
            obs_state_index = self.observation_states_dict[obs_state] # get index of observed state 
            
            for j in range(num_hidden_states):
                # calculate probability of all paths to hidden state j at observation i
                probabilities = viterbi_table[i-1, :] * self.transition_p[:,j] * self.emission_p[j][obs_state_index] # suggested by gemini
                # below, find the max probability of all paths to hidden state j and multiply by emission probability for the observed state at position i
                viterbi_table[i, j] = np.max(probabilities)
                backpointer[i, j] = np.argmax(probabilities) # store index of previous state that led to max probability
        
        # Step 3. Traceback 
        best_path[obs_len - 1] = np.argmax(viterbi_table[obs_len - 1, :]) # initialize with end of path for traceback
        for t in range(obs_len - 2, -1, -1): # iterate backwards, suggested by gemini
            best_path[t] = backpointer[t + 1, best_path[t + 1]]
        
        # Step 4. Return best hidden state sequence 
        return np.array([self.hidden_states[state] for state in best_path]) # concise list comprehension, suggested by gemini