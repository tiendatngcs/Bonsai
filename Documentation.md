# Bonsai Documentation

## Bonsai

Source code for our paper, "Bonsai: Efficient and Optimal Automatic Tensor Rematerialization for Memory-Constrained DNN Training."

### For OOPSLA'26 Artifact Reviewers

**Quick training guide** on training time can be done in a Docker image. Please follow the guide for *Quick training* showed in `QUICK-TRAIN-GUIDE.md` in the submission.

**Full workflow guide** on both training time and solving time with GUROBI solver requires
* Valid GUROBI licence
* To be done on a Anaconda virtual environment.
Please follow the guide for *Full workflow* showed in `FULL-WORKFLOW-GUIDE.md` in the submission for step-by-step installation guide. 

**ILP formulation** of Bonsai solver can be found from line 1100 in `./rockmate-bonsai/rockmate/src/rockmate/solvers/ilp/ilp_segment_model.py`.

