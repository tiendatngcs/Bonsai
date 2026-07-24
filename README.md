# Bonsai

Source code for our paper, "Bonsai: Efficient and Optimal Automatic Tensor Rematerialization for Memory-Constrained DNN Training."


## For OOPSLA'26 Artifact Reviewers 

**Quick training guide** on training time can be done in a Docker image. Please follow this guide [here](./QUICK-TRAIN-GUIDE.md) for more detail.

**Full workflow guide** on both training time and solving time with GUROBI solver requires
* Valid GUROBI licence
* To be done on a Anaconda virtual environment.
Please follow the guide [here](./FULL-WORKFLOW-GUIDE.md) for step-by-step installation guide. 


**ILP formulation** of Bonsai solver can be found from line 1100 in [this file](./rockmate-bonsai/rockmate/src/rockmate/solvers/ilp/ilp_segment_model.py).
