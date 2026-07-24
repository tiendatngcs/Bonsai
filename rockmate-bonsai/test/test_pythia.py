from models.LLM import getPythia
import torch
import ast
import logging
import rkgb

def test_rkgb_graph_builder(*args, **kwargs):
    rkgb_res = rkgb.rkgb.Result(*args, **kwargs)
    return rkgb_res


def test_dynamo_graph_builder(model, sample, **dynamo_kwargs):
    if True:
      try:
        dynamo_result : torch.export.ExportedProgram = torch.export.export(
                        model,
                        args = tuple(sample),
                        kwargs=None,
                        **dynamo_kwargs
                        )
        logging.debug('Dynamo graph builed with args=tuple(sample), kwargs=None  works')
      except Exception as e:
          logging.debug(f'Dynamo graph builder with args=tuple(sample), kwargs=None does not  work: {e}')
          raise RuntimeError
    if False:
        for input_key in ['x', 'src', 'input'][:]:
            try:
                input_dict = {input_key: sample[0]}
                dynamo_result : torch.export.ExportedProgram = torch.export.export(
                        model,
                        args = tuple(),
                        kwargs=input_dict,
                        **dynamo_kwargs
                        )
                logging.debug(f'Dynamo with args=tuple(), kwargs keys = {input_dict.keys()} works \n')
                break
            except Exception as e:
                logging.debug(f'Key {input_key} does not work: {e}')
                continue

    try:
      dynamo_graph = dynamo_result.graph
      dynamo_signature = dynamo_result.graph_signature
      whole_code_str = dynamo_graph.python_code("self").src
      whole_code_ast : ast.FunctionDef = ast.parse(whole_code_str).body[0]
    except Exception as e:
      logging.debug(f'Dynamo export failed: {e}')
      raise RuntimeError

if __name__ == '__main__':
    model, sample = getPythia(batch=1, seq_len=512)
    
    test_dynamo_graph_builder(model, sample)
    
    rkgb_res = test_rkgb_graph_builder(
        model,
        model_args=sample,
        # model_kwargs=model_kwargs,
        # verbose=verbose,
        # wanted_graphs={"FB"},
        # partitioners=[partitioner],
        inspection_device=torch.device("cuda"),
        # print_time_in_each_stage=True
        )
    
    print("RKGB Result:")
    print(rkgb_res)
    
    
    