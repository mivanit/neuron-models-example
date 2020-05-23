# neuron-models-example
Implements several different biological neuron models and accompanying tools in python under a common framework.

Sympy is used to allow for easier construction of different models, as well as (in theory) exporting the systems of equations as LaTeX. Dicts are used to store constant values for easy modification, and scipy is used for the solvers. The idea was to create a common framework that allows to very quickly implement differential equation models in the style of the Hodgkin-Huxley model.

Also provided at tools for generating waveforms for input voltages, extracting spike trains from waveforms, computing frequency-current curves, and other things. See `neuro_models/neuroUtil.py` for these tools that make use of the common `NM_model` class.

Originally made as a tool for the class Computational and Mathematical Neuroscience (Math 568) at the University of Michigan, Fall 2019

Written (except where specified otherwise) by Michael Ivanitskiy (mivanits at umich.edu). 

Huge thanks to [Victoria Booth](http://www.math.lsa.umich.edu/~vbooth/) for running such a wonderful course.

This code is not particularly well optimized or organized. I may clean it up at some point in the future for use in other projects.

# references:
mostly just the textbooks from the course. I'm missing a few of the exact papers for various neuron models, but those papers are cited somewhere in the books.
- Ermentrout, G. B., & Terman, D. H. (2010). Mathematical foundations of neuroscience (Vol. 35). Springer Science & Business Media.
- Hodgkin, A. L., & Huxley, A. F. (1952). A quantitative description of membrane current and its application to conduction and excitation in nerve. The Journal of physiology, 117(4), 500-544.
- giuseppebonaccorso/hodgkin-huxley-main.py  https://gist.github.com/giuseppebonaccorso/60ce3eb3a829b94abf64ab2b7a56aaef 
- Johnston, D., & Wu, S. M. S. (1994). Foundations of cellular neurophysiology. MIT press.
- Koch, C., & Segev, I. (Eds.). (1998). Methods in neuronal modeling: from ions to networks. MIT press.






licensed under `gpl-3.0`




<small><small><small>the commit history is a mess please dont look through it</small></small></small>