from typing import Iterator, List, Tuple, Optional, Union, Dict
from itertools import chain, count
from functools import cached_property
import logging
import time

import msgspec
import torch
import traceback

#from vllm.anyscale.shm.msgspec_shm import SharedMsgspecBufferWithEvent
from vllm.sequence import (SamplerOutput, SequenceGroupMetadata,
                           ExecuteModelData, SequenceOutput, SequenceData,
                           SequenceGroupOutput)
from vllm.worker.worker import Worker
from vllm.model_executor.parallel_utils.parallel_state import get_tensor_model_parallel_group
from vllm.model_executor.layers.sampler import CoLLMSampler
from vllm.config import CacheConfig
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.base_worker import BaseWorker
#from vllm.anyscale.profiler_utils import TorchProfiler, nvtx_range, Profilable
from vllm.utils import in_wsl

SeqId = int
TargetSeqId = int
TokenId = int

logger = logging.getLogger(__name__)


class CoLLMWorker(BaseWorker):
    @classmethod
    def from_workers(cls, base_worker: Worker,
                     asst_worker: Worker, threshold: Optional[float]=0.5) -> "LogitMixWorker":
        return cls(base_worker, asst_worker, threshold=threshold)

    def __init__(
        self,
            base_worker: Worker,
            asst_worker: Worker,
            threshold: Optional[float]=0.5,
    ):
        self.asst_worker = asst_worker
        self.base_worker = base_worker

        self.collm_sampler = None # init this later
        self.threshold = threshold
        self.device = None

        # We don't have a device set yet.
        self._copy_stream: Optional[torch.cuda.Stream] = None

        pin_memory = not in_wsl()
        self._aggregate_num_accepted_tokens = torch.tensor(
            0, dtype=torch.long, device="cpu", pin_memory=pin_memory)
        self._aggregate_num_emitted_tokens = torch.tensor(
            0, dtype=torch.long, device="cpu", pin_memory=pin_memory)
        self._aggregate_num_asst_tokens = 0

        self._rejsample_metrics_collect_interval_s = 5.0
        self._last_metrics_collect_time = 0

    def _configure_samplers(self):
        """Configure model samplers to return a probability tensor in the
        SamplerOutput. This simplifies the data wrangling logic in speculative
        decoding.
        """
        self.asst_worker.model_runner.model.sampler.include_gpu_logits_tensor = True
        self.base_worker.model_runner.model.sampler.include_gpu_logits_tensor = True

    def init_model(self):
        self.base_worker.init_model()
        self.asst_worker.init_model()

        self.device = self.base_worker.device
        self._copy_stream = torch.cuda.Stream()

    def load_model(self):
        self.base_worker.load_model()
        self.asst_worker.load_model()

        self._configure_samplers()
        vocab_size = self._vocab_size
        self.collm_sampler = CoLLMSampler(
            vocab_size=vocab_size,
            threshold=self.threshold
        )
        print(f"loaded models, base = {self.base_worker.model_runner.model_config.model}, asst={self.asst_worker.model_runner.model_config.model}, threshold={self.threshold}")

    def warm_up_model(self):
        self.base_worker.warm_up_model()
        self.asst_worker.warm_up_model()


    def profile_num_available_blocks(self, block_size: int,
                                     gpu_memory_utilization: float,
                                     cpu_swap_space: int):
        print("IN PROFILE")
        print(f"BLOCK SIZE: {block_size}")
        print(f"GPU MEM UTIL: {gpu_memory_utilization}")
        num_gpu_blocks, num_cpu_blocks = (
            self.base_worker.profile_num_available_blocks(
                block_size, gpu_memory_utilization, cpu_swap_space))

        print(f"NUM BLOCKS, according to base worker: {num_gpu_blocks}")

        new_num_gpu_blocks = self._calculate_gpu_blocks(
            block_size, num_gpu_blocks)
        return new_num_gpu_blocks, num_cpu_blocks

    def _calculate_gpu_blocks(self, block_size: int,
                              total_num_gpu_blocks: int) -> int:
        """Given total_num_gpu_blocks, the number of GPU blocks that could be
        allocate to the base model, this function calculates how many blocks
        should be given to the asst and base model.

        Note that usually the block size, in bytes, of each model is different,
        as it's a function of number of KV/layer, number of heads, and hidden
        dimension size.

        Since the base and asst models allocate the same number of blocks, we
        simply calculate the number of blocks where if allocated by both models,
        the total memory usage from KV cache is no larger than the number of
        blocks allocatable by the base model alone.
        """
        print("IN BLOCK CALCULATE")
        print(f"ORIG BLOCK NUM: {total_num_gpu_blocks}")

        base_kv_size_bytes = CacheEngine.get_cache_block_size(
            block_size,
            self.base_worker.model_config,
            self.base_worker.parallel_config,
        )

        print(f"BASE MODEL BLOCK SIZE IN BYTES: {base_kv_size_bytes}")

        asst_kv_size_bytes = CacheEngine.get_cache_block_size(
            block_size,
            self.asst_worker.model_config,
            self.asst_worker.parallel_config,
        )

        print(f"ASST MODEL BLOCK SIZE IN BYTES: {asst_kv_size_bytes}")

        new_num_gpu_blocks = int(total_num_gpu_blocks * base_kv_size_bytes /
                                 (asst_kv_size_bytes + base_kv_size_bytes))

        print(f"NEW BLOCK NUM: {new_num_gpu_blocks}")

        return new_num_gpu_blocks

    def init_cache_engine(self, cache_config: CacheConfig):
        self.base_worker.init_cache_engine(cache_config)
        self.asst_worker.init_cache_engine(cache_config)

    @property
    def rank(self):
        return self.base_worker.rank

    def get_metadata_cache_len(self) -> int:
        """Metadata cache not currently supported.
        """
        return 0

    def get_runtime_context(self) -> Optional[dict]:
        return self.base_worker.get_runtime_context()

    def _get_max_model_len(self) -> Tuple[int, int]:
        asst_max_model_len = (self.asst_worker.model_config.max_model_len)
        base_max_model_len = (self.base_worker.model_config.max_model_len)

        assert asst_max_model_len is not None
        assert base_max_model_len is not None

        return asst_max_model_len, base_max_model_len

    def execute_model_shared_memory(
            self,
            shared_memory_input,
            shared_memory_output, SharedMsgspecBufferWithEvent,
            participant_id
    ):
        shared_memory_input.decoder = msgspec.msgpack.Decoder(ExecuteModelData)
        logger.info("Worker shared memory input buffer id: "
                    f"{shared_memory_input.participant_id}")
        logger.info("Worker shared memory output buffer id: "
                    f"{shared_memory_input.participant_id}")
        parallel_group = get_tensor_model_parallel_group()
        try:
            while True:
                shared_memory_input.wait_for_incoming_data()
                data = shared_memory_input.get_data()
                torch.distributed.barrier(group=parallel_group)
                shared_memory_input.clear()
                outputs = self.execute_model(data)
                if self.rank < 1:
                    shared_memory_output.set_data(outputs)
        except Exception:
            traceback.print_exc()
            shared_memory_output.set_error()
            raise

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
        return_python_output: bool = True  # pylint: disable=unused-argument
    ) -> List[SamplerOutput]:

        execute_model_data = ExecuteModelData(
            seq_group_metadata_list,
            blocks_to_swap_in,
            blocks_to_swap_out,
            blocks_to_copy,
            0,
            True
        )

        if len(execute_model_data.seq_group_metadata_list) == 0:
            return self._run_for_empty_input(execute_model_data)

        return self._run_logit_mix_step(execute_model_data)

    def _run_for_empty_input(
            self, execute_model_data: ExecuteModelData) -> List[SamplerOutput]:
        """If there are no sequences in the input, simply call the models with
        the inpiut. This allows them to process metadata, such as cleaning up
        after a request finishes.
        """
        self.asst_worker.execute_model(*execute_model_data,
                                        return_python_output=False)
        base_output, = self.base_worker.execute_model(*execute_model_data)
        return [base_output]

    def _run_logit_mix_step(
        self,
        execute_model_data: ExecuteModelData,
    ) -> List[SamplerOutput]:
        """Execute a single step of speculative decoding.

        This runs the asst model k times, then scores each token using the
        base model. Rejection sampling is performed on the asst and base
        outputs to determine which tokens can be accepted without modifying the
        true distribution.

        Args:
            execute_model_data: The input sequences that will be speculated
                upon.
        Returns:
            A List of SamplerOutput, as if the base worker were simply called
            multiple times.
        """

        base_sampler_output = self.base_worker.execute_model(
            execute_model_data.seq_group_metadata_list,
            execute_model_data.blocks_to_swap_in,
            execute_model_data.blocks_to_swap_out,
            execute_model_data.blocks_to_copy,
        )

        asst_sampler_output = self.asst_worker.execute_model(
            execute_model_data.seq_group_metadata_list,
            execute_model_data.blocks_to_swap_in,
            execute_model_data.blocks_to_swap_out,
            execute_model_data.blocks_to_copy,
        )

        seq_group_metadata_list = execute_model_data.seq_group_metadata_list

        # repeated from model_runner execute_model code.
        # need to generate sampling metadata
        is_prompt = seq_group_metadata_list[0].is_prompt
        if is_prompt:
            inputs = self.base_worker.model_runner._prepare_prompt(seq_group_metadata_list)
            input_tokens, input_positions, input_metadata = inputs
        else:
            inputs = self.base_worker.model_runner._prepare_decode(seq_group_metadata_list)
            input_tokens, input_positions, input_metadata = inputs
        sampling_metadata = self.base_worker.model_runner._prepare_sample(seq_group_metadata_list,
                                                 input_metadata.prompt_lens)

        logits1 = base_sampler_output.logits#torch.stack([to.samples[0].probs for to in base_sampler_output])
        logits2 = asst_sampler_output.logits#torch.stack([do.samples[0].probs for do in asst_sampler_output])

        outputs = self.collm_sampler(logits1, logits2, sampling_metadata=sampling_metadata)

        # todo: i think this is supposed to be batched
        #for to, do in zip(base_sampler_output, asst_sampler_output):
        #    probs1 = to.samples[0].probs
        #    probs2 = do.samples[0].probs
        #    outputs = self.mixture_sampler(probs1, probs2, sampling_metadata=sampling_metadata)

        return outputs

    @cached_property
    def _vocab_size(self) -> int:
        """Get the vocab size of the model and make sure it's consistent between
        asst and base workers.
        """
        vocab_sizes = [
            worker.model_runner.model.config.vocab_size
            for worker in [self.asst_worker, self.base_worker]
        ]
        print(vocab_sizes)
        #assert all(vocab_sizes[0] == vocab_size for vocab_size in vocab_sizes)
        return vocab_sizes[0]
