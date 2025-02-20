import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from ofdmlib.frame import Frame
from ofdmlib.modulation import Modulation
from ofdmlib.channel import Channel
from ofdmlib.schmidlAndCox import SchmidlAndCox, SchmidlAndCoxBasic, SchmidlAndCoxAvg, SchmidlAndCoxAvgR1, SchmidlAndCoxAvgR2
from ofdmlib.demodulation import Demodulation

def run_simulation(K: int, CP: int, preamble_mod: str, payload_mod: str, 
             SNR: float, taps: list,
             sync_method: SchmidlAndCox, threshold: int
            ) -> tuple[bool, int, float, float, float]:
    """
    Run a complete simulation with the following steps:
    - Frame creation
    - Modulation
    - Channel
    - Schmidl and Cox timing synchronization
    - Demodulation
    - Equalization
    - BER calculation
    """
    # Frame creation
    M = 1; N = 2
    frame = Frame(K=K, CP=CP, CP_preamble=CP, M=M, N=N, preamble_mod=preamble_mod, payload_mod=payload_mod)
    
    # Modulation
    mod = Modulation(frame)
    mod.modulate()
    
    # Channel
    channel = Channel(frame)
    if taps is not None:
        channel.add_multipath(taps)
    channel.add_sto(K + 2 * CP)
    if SNR is not None:
        channel.add_noise(SNR)
    
    # Schmidl and Cox timing synchronization
    sync = sync_method(frame, channel)
    sync.run(threshold=threshold, min=CP, width=CP + 1)
    
    # Demodulation
    demod = Demodulation(frame, sync.get_sync_point(), channel)    
    demod.demodulate()
    demod.equalize()
    
    # Results
    found = True if sync.sync != -1 else False
    sync_error = sync.get_sync_error()
    ber = demod.get_ber()
    max_M = np.max(sync.M)
    max_N = np.max(sync.N) if sync.N is not None else None
    return found, sync_error, ber, max_M, max_N


def run_task(param, run):
    """
    Wrapper to run a simulation with a specific set of parameters.
    """
    found, sync_error, ber, max_M, max_N = run_simulation(**param)
    return {**param, 'run': run, 'sync_found': found, 'sync_error': sync_error, 'ber': ber, 'max_M': max_M, 'max_N': max_N}
    
    
    
def simulate(params: dict, nruns: int = 1, max_workers: int = 6) -> pd.DataFrame:
    """
    Run a complete simulation with the cross product of the parameters.
    """
    # Create the list of parameters
    keys = list(params.keys())
    param_list = [dict(zip(keys, values)) for values in itertools.product(*params.values())]
    print(f"Running {len(param_list)} simulation(s) {nruns} time(s)")    
    
    # Run the simulations
    results_list = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Prepare the tasks
        futures = [
            executor.submit(run_task, param, run)
            for param in param_list
            for run in range(nruns)
        ]
        
        # Run the tasks
        for future in tqdm(as_completed(futures), total=len(futures)):
            results_list.append(future.result())
    
    # Create the DataFrame
    results = pd.DataFrame(results_list)
    return results
    
params = {
        'K': [512, 1024, 2048],
        'CP': [0, 64, 128, 256],
        'preamble_mod': ['BPSK'],
        'payload_mod': ['BPSK', 'QPSK', '16QAM'],
        'SNR': np.arange(-5, 31, 5).tolist(),
        'taps': [[(1, 0.5), (2, 0.3), (3, 0.2)], None],
        'sync_method': [SchmidlAndCoxBasic, SchmidlAndCoxAvg, SchmidlAndCoxAvgR1, SchmidlAndCoxAvgR2],
        'threshold': [0.5]
    }

result = simulate(params)
result.to_csv('simulation_results.csv', index=False)
print("Simulation results saved in 'simulation_results.csv'")
