"""
Author: Vivek Ramanujan

Holds basic training scripts for specifically formatted models which also include logging.
To have a model trainable by the functions here, they must be of the following format:

class Model

public methods:

partial_fit_step takes a batch X and Y, returns loss, summary, accuracy
(the last two only if you're logging)

predict, takes X and returns a forward pass on your logits

accuracy, takes X and Y and uses your logits to determine accuracy. Returns loss, accuracy
"""

