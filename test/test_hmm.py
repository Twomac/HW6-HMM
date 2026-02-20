import pytest
from hmm import HiddenMarkovModel
import numpy as np




def test_mini_weather():
    """
    TODO: 
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

    mini_hmm=np.load('./data/mini_weather_hmm.npz')
    mini_input=np.load('./data/mini_weather_sequences.npz')

    hmm = HiddenMarkovModel(
            observation_states=mini_hmm['observation_states'],
            hidden_states=mini_hmm['hidden_states'],
            prior_p=mini_hmm['prior_p'],
            transition_p=mini_hmm['transition_p'],
            emission_p=mini_hmm['emission_p']
        )
    
    assert hmm.transition_p.shape == (len(hmm.hidden_states), len(hmm.hidden_states)), 'Transition probability matrix has incorrect shape'
    assert hmm.emission_p.shape == (len(hmm.hidden_states), len(hmm.observation_states)), 'Emission probability matrix has incorrect shape'
    
    for i in range(len(hmm.hidden_states)):
        assert np.sum(hmm.transition_p[i,:]) == 1, "Total transition probability from any state must be 1."
        assert np.sum(hmm.emission_p[i,:]) == 1, "Total emission probability of each state must be 1."

    forward_prob = hmm.forward(mini_input['observation_state_sequence'])
    expected_prob = 0.035064411621093756
    # This will raise an AssertionError with a detailed message if they don't match, suggested by Gemini 
    # since github actions was calculating the float point value slightly differently than my local machine
    np.testing.assert_allclose(forward_prob, expected_prob, rtol=1e-5, atol=0)
    
    viterbi_sequence = hmm.viterbi(mini_input['observation_state_sequence'])
    for i in range(len(viterbi_sequence)):
        assert viterbi_sequence[i] == mini_input['best_hidden_state_sequence'][i], 'Viterbi sequence for mini weather sequence is incorrect'
        



def test_full_weather():

    """
    TODO: 
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """

    full_hmm=np.load('./data/full_weather_hmm.npz')
    full_input=np.load('./data/full_weather_sequences.npz')

    hmm = HiddenMarkovModel(
            observation_states=full_hmm['observation_states'],
            hidden_states=full_hmm['hidden_states'],
            prior_p=full_hmm['prior_p'],
            transition_p=full_hmm['transition_p'],
            emission_p=full_hmm['emission_p']
        )
    
    assert hmm.transition_p.shape == (len(hmm.hidden_states), len(hmm.hidden_states)), 'Transition probability matrix has incorrect shape'
    assert hmm.emission_p.shape == (len(hmm.hidden_states), len(hmm.observation_states)), 'Emission probability matrix has incorrect shape'
    
    for i in range(len(hmm.hidden_states)):
        assert np.sum(hmm.transition_p[i,:]) == 1, "Total transition probability from any state must be 1."
        assert np.sum(hmm.emission_p[i,:]) == 1, "Total emission probability of each state must be 1."

    forward_prob = hmm.forward(full_input['observation_state_sequence'])

    viterbi_sequence = hmm.viterbi(full_input['observation_state_sequence'])
    # the below loop effectively tests for correct size and correct order since it must match exactly the best sequence provided element for element
    for i in range(len(viterbi_sequence)):
        assert viterbi_sequence[i] == full_input['best_hidden_state_sequence'][i], 'Viterbi sequence for mini weather sequence is incorrect'














