# Import NeedlemanWunsch class and read_fasta function
import numpy as np
from hmm import HiddenMarkovModel


def main():
    mini_weather_states = np.load('./data/mini_weather_hmm.npz')

    for key, val in mini_weather_states.items():
        print(key,': \n', val)

        mini_weather_sequences = np.load('./data/mini_weather_sequences.npz')
    
    print(mini_weather_sequences)

    for key, val in mini_weather_sequences.items():
        print(key,': \n', val)

    hmm = HiddenMarkovModel(
        observation_states=mini_weather_states['observation_states'],
        hidden_states=mini_weather_states['hidden_states'],
        prior_p=mini_weather_states['prior_p'],
        transition_p=mini_weather_states['transition_p'],
        emission_p=mini_weather_states['emission_p']
    )

    for key, val in mini_weather_sequences.items():
        print(key,': \n', val)
    forward_prob = hmm.forward(mini_weather_sequences['observation_state_sequence'])
    print('Forward probability for mini weather sequence: ', forward_prob)

    viterbi_sequence = hmm.viterbi(mini_weather_sequences['observation_state_sequence'])
    print('Viterbi sequence for mini weather sequence: ', viterbi_sequence)
    

    full_weather_states = np.load('./data/full_weather_hmm.npz')

    for key, val in full_weather_states.items():
        print(key,': \n', val)

        full_weather_sequences = np.load('./data/full_weather_sequences.npz')
    
    print(full_weather_sequences)

    for key, val in full_weather_sequences.items():
        print(key,': \n', val)

    HMM = HiddenMarkovModel(
        observation_states=full_weather_states['observation_states'],
        hidden_states=full_weather_states['hidden_states'],
        prior_p=full_weather_states['prior_p'],
        transition_p=full_weather_states['transition_p'],
        emission_p=full_weather_states['emission_p']
    )

    for key, val in full_weather_sequences.items():
        print(key,': \n', val)
    full_forward_prob = HMM.forward(full_weather_sequences['observation_state_sequence'])
    print('Forward probability for full weather sequence: ', full_forward_prob)

    full_viterbi_sequence = HMM.viterbi(full_weather_sequences['observation_state_sequence'])
    print('Viterbi sequence for full weather sequence: ', full_viterbi_sequence)
    

if __name__ == "__main__":
    main()
