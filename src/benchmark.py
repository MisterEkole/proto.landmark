''' A pytorch model benchmark script'''

import torch
from network import *
import torch.nn as nn
import torch.utils.benchmark as benchmark

from timeit import default_timer as timer



''' Func to measure latency on cpu'''

@torch.no_grad()

def measure_time_host(
    model:nn.Module,
    inputs:torch.Tensor,
    num_repeats: int=10,
    warmups: int=10,
    synchronize: bool=True,
    continuous_measure: bool=True) -> float:
    for _ in range(warmups):
        _ =model.forward(inputs)
        # torch.cuda.synchronize()
        elapsed_time_ms=0
    if continuous_measure:
        start=timer()
        for _ in range(num_repeats):
            _ = model.forward(inputs)

        # if synchronize:
        #     torch.cuda.synchronize()
        end=timer()
        elapsed_time_ms=(end-start)*1000
    else:
        for _ in range( num_repeats):
            start=timer()
            _ =model.forward(inputs)
            # if synchronize:
            #     torch.cuda.synchronize()
            end=timer()
            elapsed_time_ms+=(end-start)*1000
    return elapsed_time_ms/num_repeats




''' Helper func for latency mesaurement on GPU'''

# @torch.no_grad()
# def measure_device(model:nn.Module,inputs:torch.Tensor, num_repeats: int=100, warmups: int=10, synchronize: bool=True,continuous_measure: bool=True) -> float:
#     for _ in range(warmups):
#         _ =model.forward(inputs)
#         torch.cuda.synchronize()
#         elapsed_time_ms=0
#     if continuous_measure:
#         start_event=torch.cuda.Event(enable_timing=True)
#         end_event=torch.cuda.Event(enable_timing=True)
#         start_event.record()

#         for _ in range (num_repeats):
#             _ =model.forward(inputs)
#         end_event.record()

#         if synchronize:
#             torch.cuda.synchronize()
#         elapsed_time_ms=start_event.elapsed_time(end_event)

#     else:
#         for _ in range (num_repeats):
#             start_event=torch.cuda.Event(enable_timing=True)
#             end_event=torch.cuda.Event(enable_timing=True)
#             start_event.record()
#             _ =model.forward(inputs)

#             end_event.record()
#             if synchronize:
#                 torch.cuda.synchronize()

#             elapsed_time_ms+=start_event.elapsed_time(end_event)

#     return elapsed_time_ms/num_repeats


@torch.no_grad()
def run_inference(model:nn.Module,inputs: torch.Tensor) -> torch.Tensor:
    return model.forward(inputs)


def main()-> None:
    warmups=100
    num_repeats=10
    input_shape=(1,1,224,244)
    device= torch.device('cpu')

    ''' Loading the Xception Net Model to perform benchmark on'''

    model=XceptionNet()
    model.load_state_dict(torch.load('D:/Dev Projects/DeepStack/face-landmark/model/model_best.pth', map_location='cpu'))
    model.to(device)

    model.eval()

    inputs=torch.rand(input_shape, device=device)

    # torch.cuda.synchronize()

    print(" Benchmark measurements on CPU using timeit...")

    for continuous_measure in [True, False]:
        for synchronize in [True, False]:
            try:
                print("\tContinuous measure: {} Synchronize: {}".format(continuous_measure, synchronize))
                print("\t\tHost: CPU Benchmark {} ms".format(measure_time_host(model, inputs, num_repeats, warmups, synchronize, continuous_measure)))
            except Exception as e:
                print("\tContinuous measure: {} Synchronize: {}".format(continuous_measure, synchronize))
                print('\t\tHost: CPU Benchmark N/A ms')
            # torch.cuda.synchronize()
    

    # print(" Benchmark measurements  on GPU using Cuda event....")

    # for continuous_measure in [True, False]:
    #     for synchronize in [True, False]:
    #         try:
    #             print("\tContinuous measure: {} Synchronize: {}".format(continuous_measure, synchronize))
    #             print("\t\tGPU: {} Benchmark {} ms".format(device, measure_device(model, inputs, num_repeats, warmups, synchronize, continuous_measure)))
    #         except Exception  as e:
    #             print("\tContinuous measure: {} Synchronize: {}".format(continuous_measure, synchronize))
    #             print('\t\tGPU: {} Benchmark N/A ms'.format(device))

    #         torch.cuda.synchronize()

    # print(" Benchmark measurements using Pytorch built in benchmark")

    num_threads=1
    timer=benchmark.Timer(
        stmt='run_inference(model,inputs)',
        setup='from __main__ import  run_inference',
        globals={
            "model": model,
            'inputs': inputs
        },
        num_threads=num_threads,
        label='Pytorch Benchmark',
        sub_label='torch.utils.benchmark'

    )

    result=timer.timeit(num_repeats)
    print("\tPytorch Benchmark {} ms".format(result.mean*1000))

if __name__ == "__main__":
    main()


        

         










 


