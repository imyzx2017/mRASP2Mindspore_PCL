from yizx_utils import *
import os
import subprocess
import random
import socket
import torch.distributed as dist

def is_master(args):
    return args.distributed_rank == 0

def infer_init_method(args, force_distributed=False):
    if args.distributed_init_method is not None or getattr(args, "tpu", False):
        return

    if args.pipeline_model_parallel:
        balance_exists = (
            args.pipeline_balance is not None
            or args.pipeline_encoder_balance is not None
            or args.pipeline_decoder_balance is not None
        )
        devices_exist = (
            args.pipeline_devices is not None
            or args.pipeline_encoder_devices is not None
            or args.pipeline_decoder_devices is not None
        )
        if not balance_exists:
            raise ValueError(
                "--pipeline-balance is currently required for pipeline model parallelism"
            )
        if not devices_exist:
            raise ValueError(
                "--pipeline-devices is currently required for pipeline model parallelism"
            )

        args.pipeline_balance = eval_str_list(args.pipeline_balance, type=int)
        if args.pipeline_devices is not None:
            args.pipeline_devices = eval_str_list(args.pipeline_devices, type=int)
            num_pipeline_devices = len(set(args.pipeline_devices))
        else:
            args.pipeline_encoder_devices = eval_str_list(
                args.pipeline_encoder_devices, type=int
            )
            args.pipeline_decoder_devices = eval_str_list(
                args.pipeline_decoder_devices, type=int
            )
            num_pipeline_devices = len(
                set(args.pipeline_encoder_devices + args.pipeline_decoder_devices)
            )
        gpus_per_node = torch.cuda.device_count()
        assert (
            gpus_per_node >= num_pipeline_devices
            and gpus_per_node % num_pipeline_devices == 0
        ), (
            "the number of unique device IDs in --pipeline-devices must evenly divide "
            "the number of GPUs per node (multi-node pipelining is not yet supported)"
        )
        num_pipelines_per_node = gpus_per_node // num_pipeline_devices

    # support torch.distributed.launch
    if all(
        key in os.environ
        for key in ["MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK"]
    ):
        args.distributed_init_method = "env://"
        args.distributed_world_size = int(os.environ["WORLD_SIZE"])
        args.distributed_rank = int(os.environ["RANK"])
        # processes are created by torch.distributed.launch
        args.distributed_no_spawn = True

    # we can determine the init method automatically for Slurm
    elif args.distributed_port > 0:
        node_list = os.environ.get("SLURM_STEP_NODELIST")
        if node_list is None:
            node_list = os.environ.get("SLURM_JOB_NODELIST")
        if node_list is not None:
            try:
                hostnames = subprocess.check_output(
                    ["scontrol", "show", "hostnames", node_list]
                )
                args.distributed_init_method = "tcp://{host}:{port}".format(
                    host=hostnames.split()[0].decode("utf-8"),
                    port=args.distributed_port,
                )
                nnodes = int(os.environ.get("SLURM_NNODES"))
                ntasks_per_node = os.environ.get("SLURM_NTASKS_PER_NODE")
                if ntasks_per_node is not None:
                    ntasks_per_node = int(ntasks_per_node)
                else:
                    ntasks = int(os.environ.get("SLURM_NTASKS"))
                    nnodes = int(os.environ.get("SLURM_NNODES"))
                    assert ntasks % nnodes == 0
                    ntasks_per_node = int(ntasks / nnodes)
                if ntasks_per_node == 1:
                    gpus_per_node = torch.cuda.device_count()
                    node_id = int(os.environ.get("SLURM_NODEID"))
                    args.distributed_rank = node_id * gpus_per_node
                    args.distributed_world_size = nnodes * gpus_per_node
                elif args.pipeline_model_parallel:
                    assert ntasks_per_node == num_pipelines_per_node, (
                        "SLURM --ntasks-per-node must match number of pipelines per "
                        "node (={})".format(num_pipelines_per_node)
                    )
                    args.distributed_no_spawn = True
                    # For 4-way MP on nodes with 8 GPUs, ranks will be [0, 1] on
                    # the first node, [1, 2] on the second node, etc. This
                    # matches torch.distributed.launch.
                    node_id = int(os.environ.get("SLURM_NODEID"))
                    local_id = int(os.environ.get("SLURM_LOCALID"))
                    args.distributed_rank = node_id * num_pipelines_per_node + local_id
                    # In the above example, device_id will always be in [0, 1],
                    # which also matches torch.distributed.launch.
                    args.device_id = local_id
                    # We also want to set distributed_world_size to be the total
                    # number of pipelines across all nodes.
                    args.distributed_world_size = nnodes * num_pipelines_per_node
                else:
                    assert ntasks_per_node == args.distributed_world_size // nnodes
                    args.distributed_no_spawn = True
                    args.distributed_rank = int(os.environ.get("SLURM_PROCID"))
                    args.device_id = int(os.environ.get("SLURM_LOCALID"))
            except subprocess.CalledProcessError as e:  # scontrol failed
                raise e
            except FileNotFoundError:  # Slurm is not installed
                pass

    elif args.distributed_world_size > 1 or force_distributed:
        # fallback for single node with multiple GPUs
        assert args.distributed_world_size <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        args.distributed_init_method = "tcp://localhost:{port}".format(port=port)

    if args.pipeline_model_parallel:
        if not args.distributed_no_spawn:
            # When distributed_no_spawn is False, we expect distributed_rank and
            # distributed_world_size to be based on the total number of GPUs, so
            # we need to correct them to be based on the number of pipelines.
            assert args.distributed_world_size % num_pipeline_devices == 0
            args.distributed_world_size = (
                args.distributed_world_size // num_pipeline_devices
            )
            # In the case of 4-way MP on nodes with 8 GPUs, we want
            # distributed_rank to be the starting GPU index for each pipeline
            # i.e., 0, 2, ...
            assert args.distributed_rank % gpus_per_node == 0
            assert args.distributed_rank % num_pipeline_devices == 0
            args.distributed_rank = args.distributed_rank // num_pipeline_devices
            # launch one process per pipeline
            args.distributed_num_procs = num_pipelines_per_node

        # if we have 4-way MP on a node with 8 GPUs, we want device_ids to be 0
        # and 4, indicating the starting device IDs for each pipeline
        args.device_id *= num_pipeline_devices

        if args.device_id > 0:
            # if there's multiple pipelines on a node (e.g., 4-way MP on an 8
            # GPU node), we need to adjust pipeline_devices accordingly
            logger.debug(
                "setting CUDA device={} on rank {}".format(
                    args.device_id, args.distributed_rank
                )
            )
            torch.cuda.set_device(args.device_id)
            args.pipeline_devices = [args.device_id + d for d in args.pipeline_devices]
            logger.info(
                "setting pipeline_devices={} on rank {}".format(
                    args.pipeline_devices, args.distributed_rank
                ),
            )
    elif not args.distributed_no_spawn:
        args.distributed_num_procs = min(
            torch.cuda.device_count(),
            args.distributed_world_size,
        )

def verify_checkpoint_directory(save_dir: str) -> None:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    temp_file_path = os.path.join(save_dir, "dummy")
    try:
        with open(temp_file_path, "w"):
            pass
    except OSError as e:
        logger.warning(
            "Unable to access checkpoint save directory: {}".format(save_dir)
        )
        raise e
    else:
        os.remove(temp_file_path)

def distributed_init(args):
    if not getattr(args, "tpu", False):
        if torch.distributed.is_initialized():
            warnings.warn(
                "Distributed is already initialized, cannot initialize twice!"
            )
        else:
            logger.info(
                "distributed init (rank {}): {}".format(
                    args.distributed_rank,
                    args.distributed_init_method,
                )
            )
            dist.init_process_group(
                backend=args.distributed_backend,
                init_method=args.distributed_init_method,
                world_size=args.distributed_world_size,
                rank=args.distributed_rank,
            )
            logger.info(
                "initialized host {} as rank {}".format(
                    socket.gethostname(),
                    args.distributed_rank,
                )
            )

            # perform a dummy all-reduce to initialize the NCCL communicator
            if torch.cuda.is_available():
                dist.all_reduce(torch.zeros(1).cuda())

        args.distributed_rank = torch.distributed.get_rank()
    else:
        import torch_xla.core.xla_model as xm

        assert xm.xrt_world_size() == args.distributed_world_size
        args.device_id = xm.get_local_ordinal()
        args.distributed_rank = xm.get_ordinal()
        xm.rendezvous("distributed_init")  # wait for all workers
        xm.mark_step()

    if not is_master(args):
        logging.getLogger().setLevel(logging.WARNING)

    if args.model_parallel_size > 1:
        try:
            from fairseq.model_parallel.megatron.mpu import (
                get_model_parallel_rank,
                initialize_model_parallel,
                model_parallel_cuda_manual_seed,
            )
        except ImportError:
            raise ImportError(
                "\n\nPlease install the megatron submodule:"
                "\n\n  git submodule update --init "
                "fairseq/model_parallel/megatron"
            )
        initialize_model_parallel(args.model_parallel_size)
        model_parallel_cuda_manual_seed(args.seed)
        model_part_number = get_model_parallel_rank()
        args.checkpoint_suffix += "-model_part-{0}".format(model_part_number)
    return args.distributed_rank

def distributed_main(i, main, args, kwargs):
    args.device_id = i
    if torch.cuda.is_available() and not args.cpu and not getattr(args, "tpu", False):
        torch.cuda.set_device(args.device_id)
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = kwargs.pop("start_rank", 0) + i

    args.distributed_rank = distributed_init(args)

    after_distributed_init_fn = kwargs.pop("after_distributed_init_fn", None)
    if after_distributed_init_fn:
        args = after_distributed_init_fn(args)

    main(args, **kwargs)

def call_main(args, main, **kwargs):
    if args.distributed_init_method is None:
        infer_init_method(args)

    if args.distributed_init_method is not None:
        # distributed training
        if not args.distributed_no_spawn:
            start_rank = args.distributed_rank
            args.distributed_rank = None  # assign automatically
            kwargs["start_rank"] = start_rank
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(main, args, kwargs),
                nprocs=args.distributed_num_procs,
            )
        else:
            distributed_main(args.device_id, main, args, kwargs)
    elif getattr(args, "tpu", False) and args.distributed_world_size > 1:
        import torch_xla.distributed.xla_multiprocessing as xmp

        torch.multiprocessing.set_sharing_strategy("file_system")
        xmp.spawn(
            fn=distributed_main,
            args=(main, args, kwargs),
            nprocs=8,  # use all 8 TPU cores
        )
    else:
        # single GPU main
        main(args, **kwargs)

if __name__ == '__main__':
    pass
