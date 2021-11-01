# Referred torchvision:
# https://github.com/pytorch/vision/blob/master/torchvision/transforms/_transforms_video.py

from inspect import signature
import torch
import torchvision
import torchvision.transforms._transforms_video as transforms
from utils.util import Container
from torchvision.transforms.transforms import Compose
import math
import numbers
import random

import torch.nn as nn

from torchvision.transforms import _functional_video as F

__all__ = [
    "ResizeVideo",
    "RandomCropVideo",
    "RandomResizedCropVideo",
    "CenterCropVideo",
    "NormalizeVideo",
    "ToTensorVideo",
    "RandomHorizontalFlipVideo",
]

DEFAULT_MEAN = (0.43216, 0.394666, 0.37645)
DEFAULT_STD = (0.22803, 0.22145, 0.216989)


def denormalize(frame, mean=DEFAULT_MEAN, std=DEFAULT_STD):
    result = frame.clone()
    for t, m, s in zip(result, mean, std):
        t.mul_(s).add_(m)
    return result


class ResizeVideo(object):
    def __init__(self, size, keep_ratio=True, interpolation_mode="bilinear"):
        if isinstance(size, tuple):
            assert len(size) == 2, "size should be tuple (height, width)"
        self.size = size
        self.keep_ratio = keep_ratio
        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        size, scale = None, None
        if isinstance(self.size, numbers.Number):
            if self.keep_ratio:
                scale = self.size / min(clip.shape[-2:])
            else:
                size = (int(self.size), int(self.size))
        else:
            if self.keep_ratio:
                scale = min(
                    self.size[0] / clip.shape[-2],
                    self.size[1] / clip.shape[-1],
                )
            else:
                size = self.size

        return nn.functional.interpolate(
            clip,
            size=size,
            scale_factor=scale,
            mode=self.interpolation_mode,
            align_corners=False,
        )


class RandomCropVideo(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W).
        Returns:
            torch.tensor: randomly cropped/resized video clip.
                size is (C, T, OH, OW)
        """
        i, j, h, w = self.get_params(clip, self.size)
        return F.crop(clip, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + "(size={0})".format(self.size)

    @staticmethod
    def get_params(clip, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W).
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = clip.shape[3], clip.shape[2]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw


class RandomResizedCropVideo(object):
    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation_mode="bilinear",
    ):
        if isinstance(size, tuple):
            assert len(size) == 2, "size should be tuple (height, width)"
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation_mode = interpolation_mode
        self.scale = scale
        self.ratio = ratio

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W).
        Returns:
            torch.tensor: randomly cropped/resized video clip.
                size is (C, T, H, W)
        """
        i, j, h, w = self.get_params(clip, self.scale, self.ratio)
        return F.resized_crop(
            clip, i, j, h, w, self.size, self.interpolation_mode
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(size={0}, interpolation_mode={1}, scale={2}, ratio={3})".format(
                self.size, self.interpolation_mode, self.scale, self.ratio
            )
        )

    @staticmethod
    def get_params(clip, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        _w, _h = clip.shape[3], clip.shape[2]
        area = _w * _h

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= _w and h <= _h:
                i = random.randint(0, _h - h)
                j = random.randint(0, _w - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = _w / _h
        if in_ratio < min(ratio):
            w = _w
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = _h
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = _w
            h = _h
        i = (_h - h) // 2
        j = (_w - w) // 2
        return i, j, h, w


class RandomColorJitterVideo(object):
    """
    Reference : https://github.com/hassony2/torch_videovision/blob/master/torchvideotransforms/video_transforms.py

    Randomly change the brightness, contrast and saturation and hue of the clip
    Args:
    brightness (float): How much to jitter brightness. brightness_factor
    is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
    contrast (float): How much to jitter contrast. contrast_factor
    is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
    saturation (float): How much to jitter saturation. saturation_factor
    is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
    hue(float): How much to jitter hue. hue_factor is chosen uniformly from
    [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def get_params(self, brightness, contrast, saturation, hue):
        if brightness > 0:
            brightness_factor = random.uniform(
                max(0, 1 - brightness), 1 + brightness)
        else:
            brightness_factor = None

        if contrast > 0:
            contrast_factor = random.uniform(
                max(0, 1 - contrast), 1 + contrast)
        else:
            contrast_factor = None

        if saturation > 0:
            saturation_factor = random.uniform(
                max(0, 1 - saturation), 1 + saturation)
        else:
            saturation_factor = None

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
        else:
            hue_factor = None
        return brightness_factor, contrast_factor, saturation_factor, hue_factor

    def __call__(self, clip):
        """
        Args:
        clip (torch.Tensor) : clip tensors
        Returns:
        list PIL.Image : list of transformed PIL.Image
        """
        brightness, contrast, saturation, hue = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue)

        # Create img transform function sequence
        img_transforms = []
        if brightness is not None:
            img_transforms.append(
                lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
        if saturation is not None:
            img_transforms.append(
                lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
        if hue is not None:
            img_transforms.append(
                lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
        if contrast is not None:
            img_transforms.append(
                lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
        random.shuffle(img_transforms)

        if len(img_transforms) == 0:
            return clip

        # Apply to all images
        jittered_clip = []
        for img in clip.transpose(1, 0):
            for func in img_transforms:
                jittered_img = func(img)
            jittered_clip.append(jittered_img)
        jittered_clip = torch.stack(jittered_clip).transpose(1, 0)
        return jittered_clip


class CenterCropVideo(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: central cropping of video clip. Size is
            (C, T, size, size)
        """
        return F.center_crop(clip, self.size)

    def __repr__(self):
        return self.__class__.__name__ + "(size={0})".format(self.size)


class NormalizeVideo(object):
    """
    Normalize the video clip by mean subtraction and division by standard deviation
    Args:
        mean (3-tuple): pixel RGB mean
        std (3-tuple): pixel RGB standard deviation
        inplace (boolean): whether do in-place normalization
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): video clip to be normalized. Size is (C, T, H, W)
        """
        return F.normalize(clip, self.mean, self.std, self.inplace)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(mean={0}, std={1}, inplace={2})".format(
                self.mean, self.std, self.inplace
            )
        )


class ToTensorVideo(object):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimenions of clip tensor
    """

    def __init__(self):
        pass

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor, dtype=torch.uint8): Size is (T, H, W, C)
        Return:
            clip (torch.tensor, dtype=torch.float): Size is (C, T, H, W)
        """
        return F.to_tensor(clip)

    def __repr__(self):
        return self.__class__.__name__


class RandomHorizontalFlipVideo(object):
    """
    Flip the video clip along the horizonal direction with a given probability
    Args:
        p (float): probability of the clip being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Size is (C, T, H, W)
        Return:
            clip (torch.tensor): Size is (C, T, H, W)
        """
        if random.random() < self.p:
            clip = F.hflip(clip)
        return clip

    def __repr__(self):
        return self.__class__.__name__ + "(p={0})".format(self.p)


class SyncCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *inputs):
        res = inputs
        for t in self.transforms:
            res = t(*res)
        return res

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class SyncResizeVideo(ResizeVideo):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.transform = ResizeVideo(*args, **kwargs)

    def __call__(self, *inputs):
        # return result with the same transformation
        return [self.transform(x) for x in inputs]


class SyncNormalizeVideo(NormalizeVideo):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.transform = NormalizeVideo(*args, **kwargs)

    def __call__(self, *inputs):
        # return result with the same transformation
        return [
            self.transform(x) for x in inputs
        ]


class SyncCenterCropVideo(CenterCropVideo):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.transform = CenterCropVideo(*args, **kwargs)

    def __call__(self, *inputs):
        # return result with the same transformation
        return [self.transform(x) for x in inputs]


# create custom class transform
class SyncToTensorVideo(ToTensorVideo):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.transform = ToTensorVideo(*args, **kwargs)

    def __call__(self, *inputs):
        # return result with the same transformation
        return [self.transform(x) for x in inputs]


class SyncRandomCropVideo(RandomCropVideo):
    def __call__(self, *inputs):
        assert len(inputs) > 0
        # fix parameter
        i, j, h, w = self.get_params(inputs[0], self.size)

        # return result with the same transformation
        return [F.crop(x, i, j, h, w) for x in inputs]


class SyncRandomResizedCropVideo(RandomResizedCropVideo):
    def __call__(self, *inputs):
        assert len(inputs) > 0
        # fix parameter
        i, j, h, w = self.get_params(inputs[0], self.scale, self.ratio)

        # return result with the same transformation
        return [
            F.resized_crop(x, i, j, h, w, self.size,
                           self.interpolation_mode) for x in inputs
        ]


class SyncRandomHorizontalFlipVideo(RandomHorizontalFlipVideo):
    def __call__(self, *inputs):
        assert len(inputs) > 0
        is_flip = False
        # fix parameter
        if random.random() < self.p:
            is_flip = True

        # apply same transformation
        res = []
        for x in inputs:
            if is_flip:
                res.append(F.hflip(x))
            else:
                res.append(x)  # do-not flip

        return res


class SyncRandomColorJitterVideo(RandomColorJitterVideo):
    def stack_inputs(self, inputs):
        # (len(inputs), height, channel, length, width)
        inputs_stacked = torch.stack(inputs).permute((0, 3, 1, 2, 4))
        # (len(inputs)*height, channel, length, width)
        inputs_stacked = inputs_stacked.reshape(-1, *inputs_stacked.shape[2:])
        # (channel, length, len(inputs)*height, width)
        inputs_stacked = inputs_stacked.permute(1, 2, 0, 3)

        return inputs_stacked

    def __call__(self, *inputs):
        """
        Args:
        inputs (list of Tensors)
        Returns:
        list PIL.Image : list of transformed PIL.Image
        """

        assert len(inputs) > 0

        # fix parameter
        brightness, contrast, saturation, hue = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue)

        # Create img transform function sequence
        img_transforms = []
        if brightness is not None:
            img_transforms.append(
                lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
        if saturation is not None:
            img_transforms.append(
                lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
        if hue is not None:
            img_transforms.append(
                lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
        if contrast is not None:
            img_transforms.append(
                lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
        random.shuffle(img_transforms)

        if len(img_transforms) == 0:
            return inputs

        inputs_stacked = self.stack_inputs(inputs)

        # Apply to all images
        n_frames = inputs_stacked.size(1)
        jittered_clip_stacked = []
        for t in range(n_frames):
            for func in img_transforms:
                jittered_img = func(inputs_stacked[:, t])
            jittered_clip_stacked.append(jittered_img)

        jittered_clip_stacked = torch.stack(
            jittered_clip_stacked).transpose(1, 0)
        return torch.chunk(jittered_clip_stacked, len(inputs), dim=2)


def get_sync_transforms(transforms):
    tfms = []
    for t in transforms.transforms:
        sig = signature(t.__init__)
        params = sig.parameters.keys()
        params = {p: getattr(t, p) for p in params}
        sync_t = eval("Sync" + t.__class__.__name__)(**params)
        tfms.append(sync_t)

    return SyncCompose(tfms)


def get_transforms(train, tfms_config=None):
    """ Get default transformations to apply depending on whether we're applying it to the training or the validation set. If no tfms configurations are passed in, use the defaults.
    Args:
        train: whether or not this is for training
        tfms_config: Config object with tranforms-related configs
    Returns:
        A list of transforms to apply
    """
    if tfms_config is None:
        tfms_config = get_default_tfms_config(train=train)

    # 1. resize
    tfms = [
        ToTensorVideo(),
        # ResizeVideo(224, keep_ratio=False),
        ResizeVideo(
            tfms_config.im_scale, tfms_config.resize_keep_ratio
        ),
    ]

    # 2. crop
    if tfms_config.random_crop:
        if tfms_config.random_crop_scales:
            crop = RandomResizedCropVideo(
                tfms_config.input_size, tfms_config.random_crop_scales
            )
        else:
            crop = RandomCropVideo(tfms_config.input_size)
    else:
        crop = CenterCropVideo(tfms_config.input_size)
    tfms.append(crop)

    # 3. flip
    tfms.append(RandomHorizontalFlipVideo(tfms_config.flip_ratio))

    if train:
        jitter_levels = (0.5, 0.0, 0.0, 0.0)
    else:
        jitter_levels = (0.0, 0.0, 0.0, 0.0)

    # 4. brightness jittering
    tfms.append(RandomColorJitterVideo(*jitter_levels))

    # 5. normalize
    tfms.append(NormalizeVideo(tfms_config.mean, tfms_config.std))

    return Compose(tfms)


def get_default_tfms_config(train):
    """
    Args:
        train: whether or not this is for training
    Settings:
        input_size (int or tuple): Model input image size.
        im_scale (int or tuple): Resize target size.
        resize_keep_ratio (bool): If True, keep the original ratio when resizing.
        mean (tuple): Normalization mean.
        if train:
        std (tuple): Normalization std.
        flip_ratio (float): Horizontal flip ratio.
        random_crop (bool): If False, do center-crop.
        random_crop_scales (tuple): Range of size of the origin size random cropped.
    """
    flip_ratio = 0.5 if train else 0.0
    random_crop = True if train else False
    random_crop_scales = (0.6, 1.0) if train else None

    return Container(
        dict(
            input_size=112,
            im_scale=128,
            resize_keep_ratio=True,
            mean=DEFAULT_MEAN,
            std=DEFAULT_STD,
            flip_ratio=flip_ratio,
            random_crop=random_crop,
            random_crop_scales=random_crop_scales,
        )
    )
