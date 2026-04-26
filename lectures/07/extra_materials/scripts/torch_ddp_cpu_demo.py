import os
import re
import socket
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim


def get_node_rank_from_hostname() -> int:
    hostname = socket.gethostname()
    match = re.search(r"-(\d+)$", hostname)
    if not match:
        return 0
    return int(match.group(1)) - 1


def main() -> None:
    world_size = int(os.getenv("WORLD_SIZE", "2"))
    master_addr = os.getenv("MASTER_ADDR", "torch-node-1")
    master_port = os.getenv("MASTER_PORT", "29500")

    node_rank = get_node_rank_from_hostname()
    if node_rank >= world_size:
        print(f"[{socket.gethostname()}] rank {node_rank} outside world_size={world_size}, idle")
        time.sleep(5)
        return

    os.environ["RANK"] = str(node_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    dist.init_process_group(backend="gloo", init_method="env://")

    rank = dist.get_rank()
    model = nn.Linear(8, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    torch.manual_seed(42 + rank)
    x = torch.randn(32, 8)
    y = torch.randint(0, 2, (32,))

    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()

    # Aggregate gradients between nodes.
    for p in model.parameters():
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad /= world_size

    optimizer.step()

    if rank == 0:
        print(f"DDP CPU demo completed with world_size={world_size}, loss={loss.item():.4f}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
