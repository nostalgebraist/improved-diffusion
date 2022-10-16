import torch
import torchvision.transforms.functional as TF
import random, math

def roll_minmax(low, high):
    roll = random.random()
    return roll*(high-low) + low

class RandomResizedProtectedCropLazy(torch.nn.Module):
    def __init__(
        self, size, min_area, max_area=1, interpolation=TF.InterpolationMode.BILINEAR, do_resize=True, debug=False
    ):
        super().__init__()
        self.size = size
        self.min_area = min_area
        self.max_area = max_area
        self.interpolation = interpolation
        self.do_resize = do_resize
        self.debug = debug

    def get_params(self, img, safebox, pre_applied_rescale_factor=None, return_n=True, debug=True):
        def dprint(*args, **kwargs):
            if debug:
                print(*args, **kwargs)

        width, height = TF.get_image_size(img)
        # area = height * width
        max_possible_edgesize = min(height, width)
        area = max_possible_edgesize ** 2  # square crops --> don't try target edgesize bigger than shortest edge

        left_s, top_s, right_s, bottom_s = safebox
        protected_space_h = right_s - left_s
        protected_space_v = bottom_s - top_s

        if pre_applied_rescale_factor is None:
            pre_applied_rescale_factor = (1, 1)

        pre_applied_rescale_factor = max(pre_applied_rescale_factor)

        res_model = self.size
        if not isinstance(res_model, int):
            res_model = res_model[0]

        if pre_applied_rescale_factor <= 1:
            pass
            # dprint('on irrelevant branch\n')
            # dprint(f"edgesize_ratio: 1")
        else:
            dprint('on relevant branch\n')
            dprint(f"pre_applied_rescale_factor: {pre_applied_rescale_factor}\n")
            dprint(f"before: {max(protected_space_h, protected_space_v)}")

            # Res_Saved / Res_Orig = pre_applied_rescale_factor
            # Res_Model = self.size
            #
            # criterion:
            #               Res_Dynamic > Res_Model * (Res_Saved / Res_Orig)
            protected_edgesize_from_pre_applied_rescale = res_model * pre_applied_rescale_factor

            # don't protect more than the image we have on hand
            protected_edgesize_from_pre_applied_rescale = min(
                protected_edgesize_from_pre_applied_rescale,
                min(width, height)
            )

            edgesize_ratio = protected_edgesize_from_pre_applied_rescale / max(protected_space_h, protected_space_v)
            dprint(f"protected_edgesize_from_pre_applied_rescale: {protected_edgesize_from_pre_applied_rescale}")
            dprint(f"edgesize_ratio: {edgesize_ratio}")
            protected_space_h = max(protected_space_h, protected_edgesize_from_pre_applied_rescale)
            protected_space_v = max(protected_space_v, protected_edgesize_from_pre_applied_rescale)

            dprint(f"after: {max(protected_space_h, protected_space_v)}")

        protected_edgesize = max(protected_space_h, protected_space_v)

        protected_edgesize = max(protected_edgesize, min(max_possible_edgesize, res_model))

        protected_area = (protected_edgesize) * (protected_edgesize)

        min_area = max(self.min_area, protected_area / area)
        max_area = self.max_area

        roll = random.random()

        target_area = area * roll_minmax(min_area, max_area)

        target_edgesize = math.sqrt(target_area)

        dprint(f'\tprotected_edgesize:\t{protected_edgesize:.1f}')
        dprint(f'\ttarget_edgesize:\t{target_edgesize:.1f}')
        dprint(f'\tmax_possible_edgesize:\t{max_possible_edgesize:.1f}')

        ok_h, ok_v = False, False
        n = 0
        if (target_edgesize <= protected_edgesize) or (target_edgesize >= max_possible_edgesize) or (protected_edgesize >= max_possible_edgesize):
            dprint('nocrop path')
            cropbox_left, cropbox_top, cropbox_right, cropbox_bottom = (0, 0, width, height)
            ok_h, ok_v = True, True
        else:
            dprint('crop path')
        while not (ok_h and ok_v):
            if not ok_h:
                doleft = random.random() < 0.5
                if doleft:
                    cropbox_left = roll_minmax(0, left_s)
                    cropbox_right = cropbox_left + target_edgesize
                    ok_h = right_s <= cropbox_right <= width
                else:
                    cropbox_right = roll_minmax(right_s, width)
                    cropbox_left = cropbox_right - target_edgesize
                    ok_h = 0 <= cropbox_left <= left_s

            if not ok_v:
                dotop = random.random() < 0.5
                if dotop:
                    cropbox_top = roll_minmax(0, top_s)
                    cropbox_bottom = cropbox_top + target_edgesize
                    ok_v = bottom_s <= cropbox_bottom <= height
                else:
                    cropbox_bottom = roll_minmax(bottom_s, height)
                    cropbox_top = cropbox_bottom - target_edgesize
                    ok_v = 0 <= cropbox_top <= top_s

            n+=1

            if n > 10000:
                dprint('struggling w/ image, returning uncropped')
                dprint(f"safebox: {safebox}")
                dprint(f"attempt: {(cropbox_left, cropbox_top, cropbox_right, cropbox_bottom)}")
                dprint(f"target_edgesize: {target_edgesize}")
                dprint(f"protected_edgesize: {protected_edgesize}")
                dprint(f"max_possible_edgesize: {max_possible_edgesize}")
                cropbox_left, cropbox_top, cropbox_right, cropbox_bottom = (0, 0, width, height)
                break

        dprint(("target_area/min_area_allowed", target_area/(area*min_area)))
        dprint(("target_area/area", target_area/area))
        dprint(("target_edgesize", target_edgesize))
        dprint(("safebox", safebox))
        dprint(("cropbox", (cropbox_left, cropbox_top, cropbox_right, cropbox_bottom)))

        if return_n:
            return (cropbox_left, cropbox_top, cropbox_right, cropbox_bottom), n

        return (cropbox_left, cropbox_top, cropbox_right, cropbox_bottom)

    def forward(self, img, safebox, pre_applied_rescale_factor=None):
        cropbox = self.get_params(img, safebox, pre_applied_rescale_factor, return_n=False, debug=self.debug)
        i, j = cropbox[1], cropbox[0]
        h, w = cropbox[2] - j, cropbox[3] - i

        cropped = TF.crop(img, i, j, h, w)
        if self.do_resize:
            cropped = TF.resize(cropped, self.size, self.interpolation, antialias=True)
        return cropped
