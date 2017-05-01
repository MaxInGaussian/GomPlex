# GomPlex

GomPlex is a machine learning toolkit designed for predictive modeling of
complex-valued predictions. The key of success is to find a tailormade function
that maps the inputs to the targets. In this sense, linear functions undoubtedly
would be too simple to solve the problem. Nevertheless, non-linear functions
with much flexibility are mostly concerned. In fact, there are uncountable ways
to define an non-linear function, and it's generally hard to tell which class of
mathematical functions specifically works for a problem. A machine-learning kind
of approach is to approximate such a function by 'supervising' data and
'learning' patterns. In statistics, this is traditionally coined
'regression'. Although such a data-driven function can be obtained through
optimization, the optimized models tend to lose generality, which is technically
regarded as 'overfitting'.

To prevent from overfitting, one feasible approach is to carry out Bayesian
inference over 'the distribution of functions'. In this sense, our desired
function should be a sample of certain function space. A strictforward approach
would be defining the distribution of parameters that can specify the class of
function. Another way would be taking advantages of well-established stochastic
processes. Due to mathematical brevity and elegance, Gaussian process was
employed to describe the distribution over functions. [Carl Edward Rasmussen and
Christopher K. I. Williams](http://www.gaussianprocess.org/gpml/), who pioneer
and popularize the idea of using Gaussian processes for machine learning tasks,
emphasize that one of the greatest advantages of Gaussian process is that we can
integrate all possible functions over the function distribution (Gaussian
process), and obtain an analytical solution because of nice properties of
Gaussian. It is pinpointed that this Bayesian routine is prefered over
optimization on a certain estimate of function.

The idea of GomPlex came from [Quasi-Monte Carlo Feature Maps](http://jmlr.org/papers/volume17/14-538/14-538.pdf)
and [Fourier features](https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines.pdf)
, in which the feature maps are obtained by sampling on the frequency distribution corresponding to a pre-determined
kernel function. In this setup, the randomized feature maps is no more than an approximation method.
Yet, in GomPlex, we take a new scope for the association of feature maps and kernel function.
We treat the feature maps as hyperparameters, and result in optimization of the mapping on the Gaussian process regression likelihood.
In this sence, we optimize the kernel properties without explicitly define a kernel.
One significant hurdle of this approach is the explosive amount of hyperparameters,
which in turns require careful regularization on optimization.

### Highlights of GomPlex

- GomPlex optimizes
the Fourier features so as to "learn" a tailmade covariance matrix from the data. 
This removes the necessity of deciding which kernel function to use in different problems.

- GomPlex implements a variety of formulation to transform the optimized Fourier features to covariance matrix, including the typical sin-cos concatenation introduced by [Miguel](http://www.jmlr.org/papers/v11/lazaro-gredilla10a.html), and the generalized approach described by [Yarin](http://jmlr.org/proceedings/papers/v37/galb15.html).

- GomPlex uses low-rank frequency matrix for sparse approximation of Fourier features. It is 
intended to show that low-rank frequency matrix is able to lower the computational 
burden in each step of optimization, and also render faster convergence and a stabler result.

- Compared with other state-of-the-art regressors, GomPlex usually gives the most accurate prediction on the benchmark datasets of regression.

# Installation
   
### GomPlex

To install GomPlex, clone this repo:

    $ git clone https://github.com/MaxInGaussian/GomPlex.git
    $ python setup.py install
   
# Try GomPlex with Only 3 Lines of Code
```python
from GomPlex import *
# <>: necessary inputs, {}: optional inputs
gp = GomPlex(<sparsity>)
gp.fit(<X>, <y>, {opt_rate}, {max_iter}, {iter_tol}, {nlml_tol})
predict_mean, predict_std = gp.predict(new_X)
```

# Analyze Training Process on Real Time
## Training on One-dimensional Inputs
```python
gp.fit(X, y, plot=True)
```
![Plot1DFunction](tests/demo_regression_real_imag.png?raw=true "Plot 1D Function")
# 1-d Toy Example
Given our defined true function:
<img src="http://bit.ly/2qv0Zd6" align="center" border="0" alt="y=x\sin x+i(\sin x+x\cos x)" width="239" height="19" />
![Plot1DFunction](tests/demo_3d_plot_true_function.png?raw=true "Plot 1D Function")
   
# License
Copyright (c) 2016, Max W. Y. Lam
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.