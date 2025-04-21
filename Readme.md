## Files in the folder (tested on Julia v1.8 and Python 3.9):
* test_ReverseDiff.jl : For testing the automatic differentiation packages in julia (read this first).

* GraphUtil.jl : Utility functions for read/write graphs, and various graph representations.
* SIR_xxx.jl : DMP + optimization on the usual SIR model
* mSIR_xxx.jl : DMP + optimization on a modified SIR model

* data_analysis.py : Visualize the result of "SIR_DMP_opt_sigma0.jl"
* artificial_networks : Directory for some networks
* results : Directory to dump results


## Spreading process models considered:
* SIR : the usual discrete time SIR model, used in [A. Y. Lokhov et al., Phys. Rev. E 91, 012811, 2015].
* mSIR : the modified SIR model, used in [Lokhov and Saad PNAS2017].


## Optimization method and tricks:
* Basic method: gradient ascent + (optional) backtracking line search
* Some reparametrization tricks are used to enforce the total resource constraint; see [Lokhov and Saad PNAS2017]
* Barrier functions are used to enforce some inequality constraints
