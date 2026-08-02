import os

import torch
import torch.nn
import torch.distributed
import torch.multiprocessing


def worker(proc_idx: int, proc_size: int) -> None:
    print("hello from", proc_idx)

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    torch.distributed.init_process_group(backend="gloo", rank=proc_idx, world_size=proc_size)
    # rank0 ↔ rank1
    # distributed communication possible

    print(f"proc_id={proc_idx}, proc_size={torch.distributed.get_world_size()}")

    tensor = torch.tensor([proc_idx + 0.5], dtype=torch.float32)
    print(f"ck: {proc_idx} {tensor}")
    torch.distributed.all_reduce(tensor)
    # Because the operation explicitly sends the reduced result back to every process.
    print(f"ck: {proc_idx} {tensor}")

    torch.distributed.destroy_process_group()


def main() -> int:
    process_count: int = 2
    torch.multiprocessing.spawn(worker, args=(process_count,), nprocs=process_count)
    return 0


if __name__ == "__main__":
    SystemExit(main())
