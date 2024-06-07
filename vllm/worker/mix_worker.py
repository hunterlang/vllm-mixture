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
from vllm.model_executor.layers.sampler import MixtureSampler
from vllm.config import CacheConfig
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.base_worker import BaseWorker
#from vllm.anyscale.profiler_utils import TorchProfiler, nvtx_range, Profilable
from vllm.utils import in_wsl

SeqId = int
TargetSeqId = int
TokenId = int

logger = logging.getLogger(__name__)


class LogitMixWorker(BaseWorker):
    @classmethod
    def from_workers(cls, draft_worker: Worker,
                     target_worker: Worker, mixture_coef: Optional[float]=0.5) -> "LogitMixWorker":
        return cls(draft_worker, target_worker, mixture_coef=mixture_coef)

    def __init__(
        self,
            draft_worker: Worker,
            target_worker: Worker,
            mixture_coef: Optional[float]=0.5,
    ):
        self.draft_worker = draft_worker
        self.target_worker = target_worker

        self.mixture_sampler = None # init this later
        self.mixture_coef = mixture_coef
        self.device = None

        # We don't have a device set yet.
        self._copy_stream: Optional[torch.cuda.Stream] = None

        pin_memory = not in_wsl()
        self._aggregate_num_accepted_tokens = torch.tensor(
            0, dtype=torch.long, device="cpu", pin_memory=pin_memory)
        self._aggregate_num_emitted_tokens = torch.tensor(
            0, dtype=torch.long, device="cpu", pin_memory=pin_memory)
        self._aggregate_num_draft_tokens = 0

        self._rejsample_metrics_collect_interval_s = 5.0
        self._last_metrics_collect_time = 0

    def _configure_samplers(self):
        """Configure model samplers to return a probability tensor in the
        SamplerOutput. This simplifies the data wrangling logic in speculative
        decoding.
        """
        self.draft_worker.model_runner.model.sampler.include_gpu_logits_tensor = True
        self.target_worker.model_runner.model.sampler.include_gpu_logits_tensor = True

    def init_model(self):
        # Intitialize the target model before the draft model.
        # This allows the draft model to have a smaller TP degree than the
        # larger model without refactors to parallel_state.
        self.target_worker.init_model()
        self.draft_worker.init_model()

        self.device = self.target_worker.device
        self._copy_stream = torch.cuda.Stream()

    def load_model(self):
        self.target_worker.load_model()
        self.draft_worker.load_model()

        self._configure_samplers()
        vocab_size = self._vocab_size
        self.mixture_sampler = MixtureSampler(
            vocab_size=vocab_size,
            mixture_coef=self.mixture_coef
        )
        print(f"loaded models, logitmixture = {self.target_worker.model_runner.model_config.model} + {self.mixture_coef} * {self.draft_worker.model_runner.model_config.model}")

    def warm_up_model(self):
        self.target_worker.warm_up_model()
        self.draft_worker.warm_up_model()


    def profile_num_available_blocks(self, block_size: int,
                                     gpu_memory_utilization: float,
                                     cpu_swap_space: int):
        num_gpu_blocks, num_cpu_blocks = (
            self.target_worker.profile_num_available_blocks(
                block_size, gpu_memory_utilization, cpu_swap_space))

        new_num_gpu_blocks = self._calculate_gpu_blocks(
            block_size, num_gpu_blocks)
        return new_num_gpu_blocks, num_cpu_blocks

    def _calculate_gpu_blocks(self, block_size: int,
                              total_num_gpu_blocks: int) -> int:
        """Given total_num_gpu_blocks, the number of GPU blocks that could be
        allocate to the target model, this function calculates how many blocks
        should be given to the draft and target model.

        Note that usually the block size, in bytes, of each model is different,
        as it's a function of number of KV/layer, number of heads, and hidden
        dimension size.

        Since the target and draft models allocate the same number of blocks, we
        simply calculate the number of blocks where if allocated by both models,
        the total memory usage from KV cache is no larger than the number of
        blocks allocatable by the target model alone.
        """
        target_kv_size_bytes = CacheEngine.get_cache_block_size(
            block_size,
            self.target_worker.model_config,
            self.target_worker.parallel_config,
        )

        draft_kv_size_bytes = CacheEngine.get_cache_block_size(
            block_size,
            self.draft_worker.model_config,
            self.draft_worker.parallel_config,
        )

        new_num_gpu_blocks = int(total_num_gpu_blocks * target_kv_size_bytes /
                                 (draft_kv_size_bytes + target_kv_size_bytes))

        return new_num_gpu_blocks

    def init_cache_engine(self, cache_config: CacheConfig):
        self.target_worker.init_cache_engine(cache_config)
        self.draft_worker.init_cache_engine(cache_config)

    @property
    def rank(self):
        return self.target_worker.rank

    def get_metadata_cache_len(self) -> int:
        """Metadata cache not currently supported.
        """
        return 0

    def get_runtime_context(self) -> Optional[dict]:
        return self.target_worker.get_runtime_context()

    def _get_max_model_len(self) -> Tuple[int, int]:
        draft_max_model_len = (self.draft_worker.model_config.max_model_len)
        target_max_model_len = (self.target_worker.model_config.max_model_len)

        assert draft_max_model_len is not None
        assert target_max_model_len is not None

        return draft_max_model_len, target_max_model_len

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
        self.draft_worker.execute_model(*execute_model_data,
                                        return_python_output=False)
        target_output, = self.target_worker.execute_model(*execute_model_data)
        return [target_output]

    def _run_logit_mix_step(
        self,
        execute_model_data: ExecuteModelData,
    ) -> List[SamplerOutput]:
        """Execute a single step of speculative decoding.

        This runs the draft model k times, then scores each token using the
        target model. Rejection sampling is performed on the draft and target
        outputs to determine which tokens can be accepted without modifying the
        true distribution.

        Args:
            execute_model_data: The input sequences that will be speculated
                upon.
        Returns:
            A List of SamplerOutput, as if the target worker were simply called
            multiple times.
        """

        target_sampler_output = self.target_worker.execute_model(
            execute_model_data.seq_group_metadata_list,
            execute_model_data.blocks_to_swap_in,
            execute_model_data.blocks_to_swap_out,
            execute_model_data.blocks_to_copy,
        )

        draft_sampler_output = self.draft_worker.execute_model(
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
            inputs = self.target_worker.model_runner._prepare_prompt(seq_group_metadata_list)
            input_tokens, input_positions, input_metadata = inputs
        else:
            inputs = self.target_worker.model_runner._prepare_decode(seq_group_metadata_list)
            input_tokens, input_positions, input_metadata = inputs
        sampling_metadata = self.target_worker.model_runner._prepare_sample(seq_group_metadata_list,
                                                 input_metadata.prompt_lens)

        logits1 = target_sampler_output.logits#torch.stack([to.samples[0].probs for to in target_sampler_output])
        logits2 = draft_sampler_output.logits#torch.stack([do.samples[0].probs for do in draft_sampler_output])
        outputs = self.mixture_sampler(logits1, logits2, sampling_metadata=sampling_metadata)

        # todo: i think this is supposed to be batched
        #for to, do in zip(target_sampler_output, draft_sampler_output):
        #    probs1 = to.samples[0].probs
        #    probs2 = do.samples[0].probs
        #    outputs = self.mixture_sampler(probs1, probs2, sampling_metadata=sampling_metadata)

        return outputs

    @cached_property
    def _vocab_size(self) -> int:
        """Get the vocab size of the model and make sure it's consistent between
        draft and target workers.
        """
        vocab_sizes = [
            worker.model_runner.model.config.vocab_size
            for worker in [self.draft_worker, self.target_worker]
        ]
        assert all(vocab_sizes[0] == vocab_size for vocab_size in vocab_sizes)
        return vocab_sizes[0]
