import flwr as fl
import numpy as np
from typing import Optional

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd: int,
        results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: list[BaseException],
    ) -> Optional[fl.common.Weights]:
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights

# Create strategy and run server
strategy = SaveModelStrategy(
    fraction_fit=0.1,  # Sample 10% of available clients for the next round
    min_fit_clients=2,  # Minimum number of clients to be sampled for the next round
    min_eval_clients=2,
    min_available_clients=2,
)
fl.server.start_server(config={"num_rounds": 20}, strategy=strategy)