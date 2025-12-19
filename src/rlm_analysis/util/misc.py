import gc
import torch
from vllm.distributed.parallel_state import destroy_model_parallel

def unload_vllm_model(llm):
    try:
        if llm is not None:
            if hasattr(llm, 'llm_engine') and hasattr(llm.llm_engine, 'driver_worker'):
                del llm.llm_engine.driver_worker
            del llm
            destroy_model_parallel()  # Important for releasing resources in parallel setups
            gc.collect()
            torch.cuda.empty_cache()
            print("Successfully deloaded the previous vLLM model and freed resources.")
    except Exception as e:
        print(f"[WARN] Exception while unloading vLLM: {e}")


def divide_reasoning_trace_from_solution(pred, model_name="Qwen3"):
    if "gpt-oss" in model_name:
        pred = pred.replace("<|channel|>analysis<|message|>", "").replace("<|end|>", "").replace("<|return|>", "")
        splitted = pred.split("<|start|>assistant<|channel|>final<|message|>")

        if len(splitted) >= 2:
            reasoning_trace = splitted[0]
            solution = splitted[1]
            return (reasoning_trace, solution)
        else:
            print("Trace parsing failed. Find prediction without <|start|>assistant<|channel|>final<|message|>")
            reasoning_trace, solution = pred, pred
        return (reasoning_trace, solution)
    else:
        pred_spliited = pred.split("</think>")
        if len(pred_spliited) >= 2:
            reasoning_trace = pred_spliited[0]
            solution = pred_spliited[1]
            return (reasoning_trace, solution)
        else:
            print("Find solution without closing </think>")
            return (pred, pred)

