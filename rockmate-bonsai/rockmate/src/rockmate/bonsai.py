"""
BonsaiTracer: a lightweight utility to train a torch model for one iteration.
"""

import os
import torch
import copy

from .module_wrappers import getRockmateModel, getCheckmateModel, getSegmentILPModel, getGreedySearchModel, getRecompOpListModel, getNoRecomputeModel, getOffloadAllModel, getOffmateModel


def BonsaiTracer(
    model,
    inputs,
    trace_dir,
    model_name=None,
    loss_fn=None,
    optimizer_cls=torch.optim.Adam,
    optimizer_kwargs=None,
):
    # Operates on a copy of the model to avoid modifying the original model's parameters.
    # model_clone = type(model)(*model.args, **model.kwargs)
    model_clone = copy.deepcopy(model)
    model_clone.load_state_dict(model.state_dict())
    
    # ensure that pytorch version is 2.3.0a0
    assert torch.__version__.startswith("2.3.0a0"), "BonsaiTracer requires customized PyTorch 2.3.0a0"
    
    # Enable the operator trace flag export WRITE_OPERATOR_TRACE=1
    os.environ["WRITE_OPERATOR_TRACE"] = "1"
    
    # Model name
    if model_name is None:
        model_name = type(model_clone).__name__
    
    # initialize the tracer
    torch.initialize(model_name=model_name, model=model_clone, trace_dir=trace_dir)
    
    

    # Train the model for one iteration (forward + backward + step).
    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    model_clone.train()

    optimizer = optimizer_cls(model_clone.parameters(), **optimizer_kwargs)

    optimizer.zero_grad()

    # --- forward pass ---
    if isinstance(inputs, dict):
        output = model_clone(**inputs)
    elif isinstance(inputs, (list, tuple)):
        output = model_clone(*inputs)
    else:
        output = model_clone(inputs)

    # --- loss ---
    if loss_fn is not None:
        loss = loss_fn(output)
    else:
        # default: mean of the first tensor-valued output
        if isinstance(output, (list, tuple)):
            loss = output[0].mean()
        elif isinstance(output, dict):
            first = next(iter(output.values()))
            loss = first.mean()
        else:
            loss = output.mean()

    # --- backward + step ---
    loss.backward()
    optimizer.step()

    # return {
    #     "loss": loss.item(),
    #     "output": output,
    #     "optimizer": optimizer,
    # }
    
    # disable the operator trace flag export WRITE_OPERATOR_TRACE
    os.environ.pop("WRITE_OPERATOR_TRACE", None)
    
    # assert that "WRITE_OPERATOR_TRACE" is not set
    assert "WRITE_OPERATOR_TRACE" not in os.environ, "WRITE_OPERATOR_TRACE should be unset after BonsaiTracer"
    
    # reinitialize pytorch to clear the operator trace flag
    torch.initialize(model_name=model_name, model=model_clone, trace_dir=trace_dir)


def Bonsai(model, inputs, budget_GB, trace_dir, model_name=None, loss_fn=None, optimizer_cls=torch.optim.Adam, optimizer_kwargs=None, trace_file_name=None, schedule_file=None, weight_MB=None):
    if weight_MB is not None:
        os.environ["WEIGHT_MB"] = str(weight_MB)

    # if schedule already exists, skip BonsaiTracer and return the model wrapped with Bonsai
    if schedule_file is not None and os.path.exists(schedule_file):
        print(f"Schedule file {schedule_file} already exists. Skipping BonsaiTracer.")
        os.environ["SCHEDULE_FILE"] = schedule_file
        return getSegmentILPModel(model, inputs, budget_GB)
    
    # Check if trace file already exists
    if trace_file_name is None:
        trace_path = os.path.join(trace_dir, f"operator_trace_{model_name}.txt")
    else:
        # will skip trace generation if the file exists
        trace_path = os.path.join(trace_dir, trace_file_name)

    if not os.path.exists(trace_path):
        print(f"Generating trace for model '{model_name}' and saving to {trace_path}...")
        BonsaiTracer(
            model=model,
            inputs=inputs,
            trace_dir=trace_dir,
            model_name=model_name,
            loss_fn=loss_fn,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
        )
    assert os.path.exists(trace_path), f"Trace file {trace_path} was not created."
    print(f"Trace for model '{model_name}' already exists at {trace_path}. Skipping trace generation.")
    os.environ["CUSTOM_TRACE_FILE_PATH"] = trace_path
    return getSegmentILPModel(model, inputs, budget_GB)

def Rockmate(model, inputs, budget_GB, schedule_file=None):
    if schedule_file is not None and os.path.exists(schedule_file):
        print(f"Schedule file {schedule_file} already exists.")
        os.environ["SCHEDULE_FILE"] = schedule_file
        return getRockmateModel(model, inputs, budget_GB)
    return getRockmateModel(model, inputs, budget_GB)