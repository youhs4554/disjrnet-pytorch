import os
from typing import Counter
import torch
from pathlib import Path
import numpy as np
import decord
import math
import matplotlib.pyplot as plt
from einops.layers.torch import Rearrange
from natsort import natsorted
from utils.util import Container
from .transforms import DEFAULT_MEAN, DEFAULT_STD, get_sync_transforms, get_transforms, denormalize
from collections import OrderedDict
import pandas as pd
import re
import json


class VideoDataset:
    def __init__(self, root, split_file=None, sample_length=10, num_samples=1, sample_step=1, sample_stride=1, video_ext=".avi", use_person_label=True):

        self.video_root = Path(root) / "video"
        self.train = "trainlist" in split_file
        self.split_file = split_file
        self.sample_length = sample_length
        self.num_samples = num_samples
        self.sample_step = sample_step
        self.presample_length = sample_length * sample_step
        self.sample_stride = sample_stride
        self.video_ext = video_ext
        self.use_person_label = use_person_label

        self.transform = get_transforms(self.train)
        self.dataset = self.load_dataset()

        self.video_counter = Counter()

    def subsample_validationset(self, validation_idx):
        self.train = False
        self.dataset = self.load_dataset(validation_idx)

    def load_dataset(self, validation_idx=None):
        data = Container(dict(
            paths=[],
            labels=[],
            indices=[],
            num_frames=[]
        ))

        def get_selected_lines(fp, line_numbers):
            all_lines = fp.readlines()
            res = []
            for line_num in line_numbers:
                res.append(all_lines[line_num])

            return res

        with open(self.split_file, "r") as fp:
            if validation_idx is not None:
                lines = get_selected_lines(fp, validation_idx)
            else:
                lines = fp.readlines()

        for idx, row in enumerate(lines):
            video_path, video_label = row.strip().split(" ")
            video_path = os.path.join(
                self.video_root, video_path) + self.video_ext
            if not Path(video_path).exists():
                continue
            num_frames = len(decord.VideoReader(video_path))
            if num_frames > self.presample_length:
                num_subclips = math.floor((num_frames-self.sample_length) /
                                          self.sample_stride+1)
            else:
                num_subclips = 1

            if self.train:
                num_subclips = 1  # only one subclip is used for training

            data.labels.extend([int(video_label)] * num_subclips)
            data.indices.extend([idx] * num_subclips)
            data.paths.append(video_path)
            data.num_frames.append(num_frames)

        return Container(data)

    def _sample_offsets(self, video_idx):
        """
        Create a list of frame-wise offsets. For training dataset 'random shift' is used, and for valid/test set perform a sliding window sample.
        Args:
            video_idx (int):
        Return:
            list: Segment offsets (start indices)
        """
        num_frames = self.dataset.num_frames[video_idx]
        if num_frames > self.presample_length:
            if self.train:
                # Random sample <- for train data
                offsets = np.sort(
                    np.random.randint(
                        num_frames - self.presample_length + 1,
                        size=self.num_samples,
                    )
                )
            else:
                # Sliding-window sample <- for valid/test data
                n_windows = math.floor((num_frames-self.sample_length) /
                                       self.sample_stride+1)
                offsets = np.array(
                    [
                        i * self.sample_stride for i in range(n_windows)
                    ]
                )
        else:
            # very short video
            offsets = np.zeros((self.num_samples,), dtype=int)

        return offsets

    def _get_box(self, video_name, offset):
        frame_dir = self.video_root.parent / "frames" / video_name
        annotation_files = natsorted(frame_dir.glob("*.txt"))
        num_frames = len(annotation_files)

        def load_rcnn_box_annotations(self, annotation_file):
            boxes = np.loadtxt(annotation_file)
            if "MulticamFD" in str(self.video_root):
                segment_ann = pd.read_csv(
                    self.video_root.parent / "Multicam_Annotations.csv")
                delay_dict = json.load(
                    open(self.video_root.parent / "delays_multicam.json"))
                frame_ix = int(re.findall(r'\d+', annotation_file.name)[0])
                scenario_id, camera_num = map(lambda s: int(s), re.findall(
                    r'chute(\d+)-camera(\d+)', annotation_file.parent.name)[0])
                segment_start = segment_ann[segment_ann.id ==
                                            scenario_id].iloc[0].start
                delay = delay_dict["camera"+str(camera_num)][str(scenario_id)]
                # flag variable to detect person is entered in
                is_entered = frame_ix >= segment_start + delay
                if not is_entered:
                    return np.zeros((1, 5))

            if len(boxes) == 0:
                return np.zeros((1, 5))
            if boxes.ndim == 1:
                # case when single object is detected
                boxes = boxes[np.newaxis, :]

            is_person = np.where(
                boxes[:, 0] == 0)

            if open(annotation_file).read() == "" or len(is_person[0]) == 0:
                # empty case or no person
                return np.zeros((1, 5))

            box_dims = np.multiply.accumulate(
                boxes[is_person][:, -2:], 1)[:, -1]

            if box_dims[0] < 0.05:
                return np.zeros((1, 5))

            return boxes[is_person][:1]  # select first detection result

        boxes = []
        for ix in range(offset, offset + self.sample_length):
            ix = min(ix, num_frames-1)  # max clipping
            boxes.append(
                load_rcnn_box_annotations(self, annotation_files[ix])
            )

        boxes = np.row_stack(boxes)[:, 1:]  # drop detection object idx column

        return np.row_stack(boxes)

    def _get_frames(self, vreader, offset):

        clip = list()

        # decord.seek() seems to have a bug. use seek_accurate().
        vreader.seek_accurate(offset)

        # first frame
        clip.append(vreader.next().asnumpy())

        # remaining frames
        try:
            for i in range(self.sample_length - 1):
                step = self.sample_step
                if step > 1:
                    vreader.skip_frames(step - 1)
                cur_frame = vreader.next().asnumpy()
                clip.append(cur_frame)

        except StopIteration:
            # pass when video has ended
            pass

        # if clip needs more frames, simply duplicate the last frame in the clip.
        while len(clip) < self.sample_length:
            clip.append(clip[-1].copy())

        return clip

    def __len__(self):
        return len(self.dataset.indices)

    def __getitem__(self, clip_idx):
        video_idx = self.dataset.indices[clip_idx]
        video_path = self.dataset.paths[video_idx]
        label = self.dataset.labels[clip_idx]

        vreader = decord.VideoReader(video_path)
        offsets = self._sample_offsets(video_idx)
        current_offset = offsets[self.video_counter[video_path] % len(offsets)]

        original_clip = np.array(self._get_frames(
            vreader, offset=current_offset))

        sample_dict = OrderedDict({})

        if self.use_person_label:
            # get boxes for each frame
            video_name = video_path[len(
                str(self.video_root))+1:-len(self.video_ext)]

            boxes = self._get_box(video_name, current_offset)  # T x 4
            clip_local = np.zeros_like(
                original_clip)  # T x H x W x C

            height, width = original_clip.shape[1:-1]
            box_flags = [False for _ in range(len(boxes))]
            for t in range(len(boxes)):
                xc, yc, bw, bh = boxes[t]

                left = int(np.round(
                    max((xc - bw / 2) * width, 0)))
                top = int(np.round(
                    max((yc - bh / 2) * height, 0)))
                bw = int(np.round(
                    bw * width
                ))
                bh = int(np.round(
                    bh * height
                ))

                if bw > 0 and bh > 0:
                    # fill local image of detected area
                    clip_local[t, top:top+bh, left:left+bw] = original_clip[t, top:top +
                                                                            bh, left:left+bw]
                    # toggle box flag to be True
                    box_flags[t] = True
                else:
                    # do nothing
                    pass

            # interpolate missed boxes by ffill->bfill
            # Code is from https://stackoverflow.com/questions/41190852/most-efficient-way-to-forward-fill-nan-values-in-numpy-array

            # forward fill
            idx = np.where(box_flags, np.arange(len(box_flags)), 0)
            np.maximum.accumulate(idx, axis=0, out=idx)
            clip_local = clip_local[idx]

            # backward fill
            idx = np.where(box_flags, np.arange(
                len(box_flags)), len(box_flags)-1)
            idx = np.minimum.accumulate(idx[::-1], axis=0)[::-1]
            clip_local = clip_local[idx]

            sample_dict["clip_local"] = torch.from_numpy(
                clip_local)  # exclude last frame

        # update video counter
        self.video_counter.update([video_path])

        # apply transforms
        if len(sample_dict) > 0:
            sample_values = list(sample_dict.values())
            # same transforms are applied
            sync_transform = get_sync_transforms(self.transform)
            samples = sync_transform(torch.from_numpy(
                original_clip), *sample_values)
        else:
            samples = [self.transform(torch.from_numpy(original_clip))]

        # update sample_dict
        for key in sample_dict:
            sample_dict[key] = samples[1+list(sample_dict.keys()).index(key)]
        sample_dict["clip"] = samples[0]
        sample_dict["label"] = label

        return sample_dict

    @property
    def labels(self):
        return self.dataset.labels

    @property
    def clip_list(self):
        res = []
        for num_frames, p in zip(self.dataset.num_frames, self.dataset.paths):
            vname = p[len(str(self.video_root))+1:]
            vname, _ = os.path.splitext(vname)

            if num_frames > self.presample_length:
                num_subclips = math.floor((num_frames-self.sample_length) /
                                          self.sample_stride+1)
            else:
                num_subclips = 1

            res += [vname] * num_subclips
        return res

    def _show_samples(self, samples, mean=DEFAULT_MEAN, std=DEFAULT_STD):

        has_local_frames = "clip_local" in samples[0]

        if has_local_frames:
            local_images = [item.get("clip_local") for item in samples]

        images = [item.get("clip") for item in samples]
        labels = [item.get("label") for item in samples]

        rows = len(images)
        plt.tight_layout()
        fig, axes = plt.subplots(
            rows,
            self.sample_length-1,
            figsize=(4 * (self.sample_length-1), 3*rows)
        )

        if axes.ndim == 1:
            axes = np.expand_dims(axes, 0)

        for i, ax in enumerate(axes):
            clip = images[i]
            if has_local_frames:
                clip_local = local_images[i]
                clip_local = Rearrange("c t h w -> t c h w")(clip_local)

            clip = Rearrange("c t h w -> t c h w")(clip)

            if not isinstance(ax, np.ndarray):
                ax = [ax]
            for j, a in enumerate(ax):
                a.axis("off")
                img = np.moveaxis(denormalize(
                    clip[j], mean, std).numpy(), 0, -1)
                if has_local_frames:
                    local_img = np.moveaxis(denormalize(
                        clip_local[j], mean, std).numpy(), 0, -1)
                    img = np.column_stack((img, local_img))

                a.imshow(img)

                # display label/label_name on the first image
                if j == 0:
                    a.text(
                        x=3,
                        y=15,
                        s=f"{labels[i]}",
                        fontsize=20,
                        bbox=dict(facecolor="white", alpha=0.80),
                    )

    def show_samples(self, rows=2):
        samples = [self.__getitem__(i) for i in range(rows)]
        self._show_samples(samples)
