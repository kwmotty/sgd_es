# Convergence Analysis of SGD under Expected Smoothness

##abstract
Stochastic gradient descent (SGD) is the
workhorse of large-scale learning, yet classi-
cal analyses rely on assumptions that can
be either too strong (bounded variance) or
too coarse (uniform noise). The expected
smoothness (ES) condition has emerged as
a flexible alternative that ties the second mo-
ment of stochastic gradients to the objec-
tive value and the full gradient. This paper
presents a self-contained convergence analy-
sis of SGD under ES. We (i) deepen ES in
the main text, clarifying its interpretations
and sampling-dependent constants; (ii) derive
tight bounds; and (iii) prove O(1/K)-type
rates under a variety of step-size schedules.
All proofs are given in full detail in the ap-
pendix. Our treatment unifies and extends
recent threads (Khaled and Richtárik, 2020;
Umeda and Iiduka, 2025).
