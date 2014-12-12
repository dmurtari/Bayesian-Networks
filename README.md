Bayesian-Networks
=================

Handles user queries about a disease using Bayes Nets. Currently, conditional
and marginal probabilities are working, although conditional probability 
queries always seem to return 1. The joint probability distribution would
take advantage of both the conditional and marginal probabilities to define
a recursive function to perform the joint probability calculation

Fixed since last submitted:

For the resubmittal, I fixed the conditional probabilities to be correct (turns
out I was simply returning the wrong thing originally...). I have done work on 
the joint probabilities, but the numbers are still off. 