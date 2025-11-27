"""
pretty similar to previous steps but combines user agreement and HIT model prediction correction

This assumed neither the model nor the human is perfect, so we need to correct both.

Case 1: the model (or second human) and the human missed the object -> we might never find it. If it is close to a known object we might find when the labels are double checked. But a remote isolated object will likely never be found.
Case 2: the model (or second human) found an object but the human missed it -> we can add it to the ground truth, if on correction the (possiblly second) human agrees
Case 3: the model (or second human) missed an object but the human found it -> we can keep it in the ground truth, if on correction the (possiblly second) human agrees
Case 4: the model (or second human) and the human found the object -> we can keep it in the ground truth, but still might correct it for label position.


"""


# TODO Find the agreement, false positive and false negatives and submit all of them to CVAT for correction