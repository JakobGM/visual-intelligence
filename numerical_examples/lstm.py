from pathlib import Path

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def tanh(x: np.ndarray) -> np.ndarray:
    numerator = np.exp(x) - np.exp(-x)
    denominator = np.exp(x) + np.exp(-x)
    return numerator / denominator


def softmax(x: np.ndarray) -> np.ndarray:
    exponential = np.exp(x)
    return exponential / exponential.sum()

class Parameters:
    def __init__(self, event_size: int, hidden_size: int):
        # Weights and biases for the sigmoid "f-function"
        self.event_forget_weights = np.zeros((hidden_size, event_size))
        self.event_forget_bias = 0
        self.hidden_forget_weights = np.zeros((hidden_size, hidden_size))
        self.hidden_forget_bias = 0

        # Weights and biases for the sigmoid "i-function"
        self.event_update_weights = np.zeros((hidden_size, event_size))
        self.event_update_bias = 0
        self.hidden_update_weights = np.zeros((hidden_size, hidden_size))
        self.hidden_update_bias = 0

        # Weights and biases for the tanh "C-bar-function"
        self.event_candidate_weights = np.zeros((hidden_size, event_size))
        self.event_candidate_bias = 0
        self.hidden_candidate_weights = np.zeros((hidden_size, hidden_size))
        self.hidden_candidate_bias = 0

        # Weights and biases for the "o-function"
        self.event_output_weights = np.zeros((hidden_size, event_size))
        self.event_output_bias = 0
        self.hidden_output_weights = np.zeros((hidden_size, hidden_size))
        self.hidden_output_bias = 0


def forget_gate(event, hidden_state, prev_cell_state, parameters):
    """Forget gate deciding how much of the previous cell state to keep."""
    forget_hidden = (
        parameters.hidden_forget_weights @ hidden_state
        + parameters.hidden_forget_bias
    )
    forget_event = (
        parameters.event_forget_weights @ event
        + parameters.event_forget_bias
    )
    # Values between zero and one indicating how much to forget
    forgetter = sigmoid(forget_hidden + forget_event)

    # Return the state that should be kept
    kept_state = forgetter * prev_cell_state
    return kept_state

def input_gate(event, hidden_state, parameters):
    """Input gate deciding how to update the cell state."""
    # We have certain candidates from the new event and the hidden state
    # we would like to update the cell state with
    hidden_candidates = (
        parameters.hidden_candidate_weights @ hidden_state
        + parameters.hidden_candidate_bias
    )
    event_candidates = (
        parameters.event_candidate_weights @ event
        + parameters.event_candidate_bias
    )

    # We must also determine how much to weigh these updates
    event_update = (
        parameters.event_update_weights @ event
        + parameters.event_update_bias
    )
    hidden_update = (
        parameters.hidden_update_weights @ hidden_state
        + parameters.hidden_update_bias
    )

    # Finally returning the update
    return (
        sigmoid(event_update + hidden_update)
        * tanh(event_candidates + hidden_candidates)
    )


def cell_state(forget_gate_output, input_gate_output):
    """
    New cell state, a combination of the partially forgotten cell state
    and the newly proposed state.
    """
    return forget_gate_output + input_gate_output


def output_gate(event, hidden_state, cell_state, parameters):
    """Decide what to output from the LSTM cell."""
    hidden_output = (
        parameters.hidden_output_weights @ hidden_state
        + parameters.hidden_output_bias
    )
    event_output = (
        parameters.event_output_weights @ event
        + parameters.event_output_bias
    )
    return (
        sigmoid(event_output + hidden_output)
        * tanh(cell_state)
    )
