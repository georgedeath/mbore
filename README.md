# MBORE: Multi-objective Bayesian Optimisation by Density-Ratio Estimation

This repository contains the Python3 code for the `MBORE` method presented in:

> George De Ath, Tinkle Chugh, and Alma A. M. Rahat. 2022. MBORE: Multi-objective Bayesian Optimisation by Density-Ratio Estimation. In Genetic and Evolutionary Computation Conference (GECCO ’22), July 9–13, 2022, Boston, MA, USA. ACM, New York, NY, USA, 10 pages.
<br/>> **Paper:** <https://doi.org/10.1145/3512290.3528769> (to appear)
<br/>> **Preprint:** <https://arxiv.org/abs/2203.16912>
<br/>> **Optimisation results:** <https://doi.org/10.24378/exe.3943> (13 GB)

The repository also contains all training data used for the initialisation of
the optimisation runs carried out, as well as Jupyter notebooks to generate the
figures and tables in the paper and its supplement are also included.
Due to GitHub space constraints, the optimisation results cannot be included in
the repository. However, they have been hosted externally and can be downloaded
optimisation results link above. Each zip file must be extracted to a folder
with the same name as the zip file, e.g., the contents of `final_results.zip`
must be extracted a folder named `final_results` in the repository folder. Each
zip file prefixed with `results_` contains results for the named test problems,
and `processed_results` and `final_results` contain combined results that
include hypervolume and IGD+ calculations, and statistical test results
respectively. Only these latter two files need to be downloaded to reproduce
the plots created in the notebooks discussed below.

## Citation

If you use any part of this code in your work, please cite:

```bibtex
@inproceedings{death:mbore,
    author = {George {De Ath} and Tinkle Chugh and Alma A. M. Rahat},
    title = {MBORE: Multi-objective Bayesian Optimisation by Density-Ratio Estimation},
    year = {2022},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3512290.3528769},
    doi = {10.1145/3512290.3528769},
    booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference},
}
```

## Installation instructions (Anaconda)
The following commands should create an
[Anaconda environment](https://www.anaconda.com/products/distribution) that
contains all the packages required to carry out the experiments.
Note that the "cpuonly" option can be omitted from the
[PyTorch](https://github.com/pytorch/pytorch) install line if you have a GPU
and that Python version 3.7 is needed to install
[pygmo](https://esa.github.io/pygmo2/) via pip on Windows.

```shell
conda create --name mbore python=3.7 matplotlib numpy scipy ipykernel tqdm joblib scikit-learn statsmodels -y

conda activate mbore

conda install pytorch==1.9 cpuonly botorch gpytorch -c pytorch -c gpytorch -y

pip install pygmo==2.13 pymoo pyDOE2 xgboost tensorflow docopt tueplots

pip install -U git+https://github.com/ltiao/bore.git@v1.5.0
```

## Reproduction of experiments

The python file `run_exp.py` provides a convenient way to reproduce an
individual experimental evaluation carried out the paper. It has the following
syntax:

```script
> conda activate mbore
> python run_exp.py --help
MBORE Experiment Runner
Usage:
    run_exp.py <problem_name> <problem_id> <problem_dim> <problem_fdim> <run_no>
               <scalarization> <classifier> <optimizer> <gamma_start>
               [--gamma_end=<gamma_end>] [--gamma_schedule=<gamma_schedule>]
               [--budget=<budget>] [--verbose] [--allcores]

Arguments:
    <problem_name>   Type of problem to be optimised, e.g. DTLZ.
    <problem_id>     Version of the problem, e.g. 1.
    <problem_dim>    Dimensionality of the problem (decision space).
    <problem_fdim>   Number of objectives.
    <run_no>         Run number, typically 1 to 51 -- training data must exist
                     for this.
    <scalarization>  Method used to convert objective values to scalar, e.g.
                     HypI or DomRank.
    <classifier>     Type of <classifier>, either XGB or FCNet.
    <optimizer>      Optimization method for the <classifier>, choose from:
                     Sobol, CMAES or Grad (Grad only avaliable for FCNet).
    <gamma_start>    Starting gamma value.

Options:
    --gamma_end=<ge>       Ending gamma value (only for use with schedule)
                           [default: Unused]
    --gamma_schedule=<gs>  Scheduler for changing gamma values over time.
                           [default Unused]
    --budget=<b>           Number of function values to optimise for
                           [default: 300]
    --verbose              Print the status of the optimisation.
    --allcores             Use all processor cores on the machine.
```

## Training data
The initial training locations for each of the 21 sets of
[Latin hypercube](https://www.jstor.org/stable/1268522) samples for each
combination of test problem, problem dimensionality and number of objectives is
stored in the `data` directory. Files are named with the scheme
`{problem_suite}{problem_id}_d={dim}_o={n_obj}_{run_no}.npz`, where
`problem_suite` and `problem_id` corresponds to the specific test problem the
data is for, e.g. `DTLZ` and `1`, and `dim` and `n_obj` correspond to the
problem's dimensionality and number of objectives respectively, and `run_no`
corresponds to the run number. For example, the file `DTLZ1_d=2_o=3_15.npz`
contains data for the DTLZ1 test problem with 2 input dimensions and 3
objectives, and is the 15th run. Each training data file contains two arrays,
named `Xtr` and `Ytr`, which contain the `dim`-dimensional decision vectors and
their corresponding `n_obj` objective values:

```python
> python
>>> import numpy as np
>>> # here, we load the 15th set of training examples for
>>> # the 2-dimensional DTLZ1 problem with 3 objectives:
>>> with np.load("data/DTLZ1_d=2_o=3_15.npz") as d:
        Xtr, Ytr = d['Xtr'], d['Ytr']
>>> Xtr.shape, Ytr.shape
((20, 2), (20, 3))
```

The folder `pareto_fronts` contains the estimated Pareto fronts for each
evaluated test problem and number of objectives. Each file is named using the
convention `{problem_suite}{problem_id}_obj_{n_obj}.csv`, and
contains comma-separated rows of objective values on the front. For example,
`DTLZ3_obj_6.csv` contains the Pareto front for the 6-objective DTLZ3 test
problem. The fronts for the DTLZ and WFG test problems were found using
[PlatEMO framework](https://ieeexplore.ieee.org/document/8065138). The fronts
for the real-world test problems (prefixed with RW) were taken from the
benchmark's [GitHub repository](https://github.com/ryojitanabe/reproblems).

## Optimisation results files
Saved results files are initially stored in the `results_*` folders, and are
named with the naming convention
`{problem_suite}{problem_id}_d={dim}_o={n_obj}_{run_no}_{scalariser}_{model}_{optimiser}_g={gamma}.npz`,
and, in the case of those optimisation runs using a GP model,
`{model}_{optimiser}_g={gamma}.npz` is replaced with `GP_EI.npz`. For example,
`WFG1_d=20_o=10_1_DomRank_FCNet_Grad_g=0.33.npz` contains the results for the
10-dimensional `WFG1` test problem with 10 objectives using the DomRank
scalarisation method with a neural net classifier (FCNet), with new
locations selected via gradient-based optimisation, and gamma set to 0.33.
Each of these numpy files contain the same variables as in the training data,
i.e. `Xtr` and `Ytr` -- all the evaluated input locations and their
corresponding objective evaluations.

The results of the hypervolume and IGD+ calculations are carried out via the
included `preprocess_results.py` script. This stores its results in the
`processed_results` folder and contains files named similarly to above:
`{problem_suite}{problem_id}_d={dim}_o={n_obj}_{scalariser}_{model}_{optimiser}_g={gamma}.npz`.
Each file contains the evaluated input locations
(`Xtrs`, shape `(n_runs, n_evaluations, n_dims)`) and their corresponding
objective values (`Ytrs`, shape `(n_runs, n_evaluations, n_objectives)`),
as well as the hypervolume and IGD+ calculations
(`hvs` and `igds`, both shape `(n_runs, n_evaluations)`). The computational cost
of each iteration of the evaluated algorithm is also included
(`timing`, shape `(n_runs, n_evaluations)`).

```python
> python
>>> import numpy as np
>>> filepath = r"processed_results/DTLZ1_d=2_o=2_DomRank_FCNet_Grad_g=0.33.npz"
>>> with np.load(filepath) as d:
        Xtrs = data['Xtrs']
        Ytrs = data['Ytrs']
        timing = data['timing']
        hvs = data['hvs']
        igds = data['igds']
>>> Xtrs.shape, Ytrs.shape, timing.shape, hvs.shape, igds.shape
((21, 320, 2), (21, 320, 2), (21, 300), (21, 301), (21, 301))
```

## Reproduction of figures and tables in the paper
[Results_gathering.ipynb](Results_gathering.ipynb) takes the results files from
the folder `preprocessed_results` and extracts the performance
indicator results and computation time information, saving these results to the
`final_results` directory. It also carries out the statistical testing detailed
in the paper and saves these results to the same folder. Note that these files
are included in the repository.

[Results_plotting.ipynb](Results_plotting.ipynb) contains the code to load the
results summaries and statistical testing from the `final_results` directory,
and produces all the figures and tables shown in the paper.
