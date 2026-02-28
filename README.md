# DOSS: A Python Solver for Box-Constrained Black-Box Optimization

## Authors

-   **Limeng Liu**
-   **Muming Yang**
-   **Christine A. Shoemaker**
-   **Tingting Xie**

------------------------------------------------------------------------

## Overview

This repository provides a **Python implementation of DOSS**, a
surrogate-based algorithm for solving **box-constrained black-box
optimization problems**. The software accompanies the research article:

> Liu, L., Yang, M., Shoemaker, C. A., & Xie, T. (2024).\
> *Solving higher dimensional expensive black box global optimization
> problems using sparse directional search on surrogates.*\
> Mathematical Programming Computation, 16, 665--693.\
> https://doi.org/10.1007/s12532-024-00267-7

The package enables researchers to reproduce all computational
experiments reported in the paper and compare DOSS against several
state-of-the-art optimization algorithms.

------------------------------------------------------------------------

## Problem Formulation

The software solves optimization problems of the form:

$$(P) \quad \min \ f(x) $$

$$\text{subject to } x \in D$$

where:

-   f(x) is a continuous black-box objective function,
-   D = \[l,r\]\^d is a bounded box domain,
-   gradients are unavailable or expensive to compute.

Applications include:

-   simulation-based optimization
-   engineering design
-   robotics control
-   expensive scientific computation

------------------------------------------------------------------------

## Implemented Algorithms

### Proposed Method

-   **DOSS** --- Directional Optimization Search with Surrogate (proposed in
    the paper)

### Benchmark Algorithms

-   **RBFOpt** --- Radial basis function optimization
-   **TuRBO** --- Trust-region Bayesian optimization
-   **DYCORS** --- Dynamic coordinate search (pySOT)

------------------------------------------------------------------------

## Software Structure
The software contains the following folders:

1. **Alternative Algorithms**: Installation packages for alternative algorithms.
2. **solvers**: Convex and nonconvex nonlinear programs (NLPs) required by RBFOpt.
3. **test_functions**: All the test problems used in the paper.
4. **tests**: Test functions for surrogate models and package installations.

## Installation of Dependencies

1. **Switch to the root user**:
    ```bash
    sudo -i
    ```

2. **Install pySOT 0.2.3**:
    - [pySOT GitHub Repository](https://github.com/dme65/pySOT)
    ```bash
    cd path/to/pySOT-master
    python3 setup.py install
    ```

3. **Install RBFOpt**:
    - [RBFOpt GitHub Repository](https://github.com/coin-or/rbfopt)
    ```bash
    pip3 install rbfopt
    ```

4. **Install TuRBO**:
    - [TuRBO GitHub Repository](https://github.com/uber-research/TuRBO)
    ```bash
    # Download turbo folder to the local directory
    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
    pip3 install gpytorch
    ```

5. **Install Benchmark Functions**:
    ```bash
    # Conda install (optional)
    # conda install swig # needed to build Box2D in the pip install
    pip3 install box2d-py  # Repackaged version of pybox2d
    pip3 install gym
    pip3 install pygame
    ```

6. **Upgrade scipy**:
    ```bash
    pip3 install -U scipy
    ```

7. **Install psutil**:
    ```bash
    pip3 install psutil
    ```

*Note: When running figures, the results are saved as PDF files.*

## Reproducing the Computational Results

1. **Create a directory to store results**:
    ```bash
    mkdir ./results/temp
    ```

2. **Running command**:
    - `-p`: Test problems
    - `-d`: Dimensions
    - `-e`: Number of evaluations
    - `-s`: Alternative algorithms
    - `-t`: Number of trials
    - `-r`: Target directory for results

3. **Alternative algorithms**:
    - **SDSGDYCK**: DOSS algorithm proposed in the paper.
    - **TuRBO**: Gaussian process-based algorithm in the TuRBO package.
    - **RBFOpt**: RBF-based algorithm in the RBFOpt package.
    - **DYCORS**: RBF-based algorithm in the pySOT package.

4. **Test problems**:
    - Test functions: "Ackley," "Eggholder," "Keane," "Levy," "Michalewicz," etc.
    - Higher-dimensional problems: RobotPushing (14D), RoverTrajPlan (60D).

5. **Example to reproduce results**:
    ```bash
    python3 test.py -p RobotPushing -d 14 -e 3000 -s SDSGDYCK -t 1 -r ./results/temp
    ```

6. **Server Commands**:
    - **DOSS (cnt) algorithm**:
      ```bash
      bash pbseasy.sh "-s SDSGDYCK -d 36 -e 1850 -t 2 -r ./results/temp" SDSGDYCK
      ```
    - **DOSS (std) algorithm**:
      ```bash
      bash pbseasy.sh "-s SDSGDYCK_std -d 36 -e 1850 -t 2 -r ./results/temp" SDSGDYCK_std
      ```
    - **DYCORS algorithm**:
      ```bash
      bash pbseasy.sh "-s DYCORS -d 36 -e 1850 -t 2 -r ./results/temp" DYCORS
      ```
    - **TuRBO algorithm**:
      ```bash
      bash pbseasy.sh "-s TuRBO -d 36 -e 1850 -t 2 --num_tr 5 -r ./results/temp" TuRBO
      ```
    - **RBFOpt algorithm**:
      ```bash
      chmod +x ./solvers/bonmin
      bash pbseasy.sh "-s RBFOpt -d 36 -e 1850 -t 2 -r ./results/temp" RBFOpt
      ```

7. **Run All Experiments**:
    ```bash
    bash pbsscript.sh
    ```

8. **Plot Results**:
    ```bash
    pip3 install xlsxwriter
    echo "backend: Agg" > ~/.config/matplotlib/matplotlibrc
    python3 plotcurve_fig1.py
    python3 plotcurve_fig2_low.py
    python3 plotcurve_fig2_mid.py
    python3 plotcurve_fig2_high.py
    python3 plotcurve_fig3.py
    python3 plotcurve_fig4.py
    python3 plotcurve_fig5_Robot.py
    python3 plotcurve_fig5_Rover.py
    ```

## Code Structure

1. **Initial Sampling Method**: (`new_lhd.py`, Section 4.1)
   - `LatinHypercube`: Extends the base Latin Hypercube class to generate sample designs in an evenly distributed manner across multiple dimensions for simulation or optimization tasks.

2. **Surrogate Models**: (`new_surrogate.py`, Section 3.2)
   - `RBFInterpolant`: Enhances the base RBF interpolant class with complex initialization, including multiple kernel and tail strategies (e.g., linear, cubic, thin plate spline). It adjusts its internal strategies based on data availability.

3. **Evaluation Points Sampling and Selection Strategy**: (`new_strategy.py`, Sections 4.2, 4.3)
   - Various Strategy classes (e.g., DYCORSStrategy, SDSGCKDYCORSStrategy_std, SDSGCKDYCORSStrategy, etc.): These classes are variations of a dynamic coordinate search strategy (DYCORSStrategy), each implementing specific modifications or enhancements to the original strategy, such as different weighting schemes or hybrid approaches. They are likely used for adaptive optimization under different conditions or preferences.

4. **Utility Functions**: (`new_auxfunc.py`, Section 3.3)
   - Include major candidate points generating function and merit function to balance exploitation and exploration. 

5. **Main Functions**: (`test_rbf.py`, `test_turbo.py`, `test_rbfopt.py`)
   - Main scripts to run DOSS, TuRBO, and RBFOpt algorithms.

------------------------------------------------------------------------

## Citation

If you use this software, please cite:

@article{liu2024solving, title = {Solving higher dimensional expensive
black box global optimization problems using sparse directional search
on surrogates}, author = {Liu, Limeng and Yang, Muming and Shoemaker,
Christine A. and Xie, Tingting}, journal = {Mathematical Programming
Computation}, year = {2024}, volume = {16}, pages = {665--693}, doi =
{10.1007/s12532-024-00267-7}, url =
{https://doi.org/10.1007/s12532-024-00267-7} }

------------------------------------------------------------------------
