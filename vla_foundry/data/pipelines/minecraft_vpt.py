"""On-demand VPT video/action decoding for Minecraft training."""

import webdataset as wds

from vla_foundry.data.minecraft.streaming import index_shards_to_samples
from vla_foundry.data.pipelines.robotics import RoboticsPipeline
from vla_foundry.data.utils import deterministic_shuffle, log_and_continue


class MinecraftVPTStreamingPipeline(RoboticsPipeline):
    """Emit the same robotics batches as prepared tar shards without materializing images."""

    def create_pipeline(self, datastring, checkpoint_num):
        def stream_samples(source):
            return index_shards_to_samples(
                source,
                data_root=self.data_params.vpt_data_root,
                image_size=self.data_params.image_size,
                horizon=self.data_params.vpt_horizon,
            )

        return [
            wds.SimpleShardList(datastring),
            deterministic_shuffle(
                bufsize=self.data_params.shuffle_buffer_size,
                initial=self.data_params.shuffle_initial,
                seed=self.data_params.seed,
                epoch=checkpoint_num,
            ),
            wds.split_by_node,
            wds.split_by_worker,
            stream_samples,
            wds.batched(self.batch_size, partial=False),
            wds.map(
                lambda batch: self.robotics_processor.process_inputs(
                    batch,
                    image_names=self.data_params.image_names,
                    max_text_seq_len=self.data_params.max_text_seq_len,
                ),
                handler=log_and_continue,
            ),
            wds.map(
                lambda batch: self.robotics_processor.add_action_and_proprioception_fields(
                    batch,
                    action_fields=self.data_params.action_fields,
                    proprioception_fields=self.data_params.proprioception_fields,
                ),
                handler=log_and_continue,
            ),
            wds.map(lambda batch: {**batch, "images": None}, handler=log_and_continue),
        ]
