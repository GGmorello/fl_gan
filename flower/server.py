import flwr as fl

hist = fl.server.start_server(config={"num_rounds": 30})
