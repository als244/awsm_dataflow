from matmul_dispatcher import CublasLtDispatcher

dispatcher = CublasLtDispatcher(round_multiple=32)
dispatcher_secondary = CublasLtDispatcher(round_multiple=32)