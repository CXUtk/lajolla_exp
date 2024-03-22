from typing import List
import drjit as dr
import mitsuba as mi
import torch

mi.set_variant('cuda_ad_rgb')

sequence = mi.TensorXf(torch.zeros(1, 100).to('cuda'))

class PSSSampler(mi.Sampler):
    def __init__(self, props):
        mi.Sampler.__init__(self, props)
        self.sampleIndex = 0
        if props.has_property('rng_sequence'):
            self.rng_sequence = props['rng_sequence']
        else:
            self.rng_sequence = mi.TensorXf(torch.zeros(1, 100).to('cuda'))

        if True:
            print()

    def clone(self: mi.Sampler) -> mi.Sampler:
        copy = PSSSampler(mi.Properties())
        copy.sampleIndex = self.sampleIndex
        copy.rng_sequence = self.rng_sequence
        return copy

    def fork(self: mi.Sampler) -> mi.Sampler:
        return PSSSampler(mi.Properties())

    def next_1d(self: mi.Sampler, active: bool = True) -> mi.Float:
        ret = self.rng_sequence[0][self.sampleIndex]
        self.sampleIndex += 1
        return mi.Float(dr.ravel(ret))

    def next_2d(self: mi.Sampler, active: bool = True) -> mi.Point2f:
        x = self.next_1d()
        y = self.next_1d()
        return mi.Point2f(x, y)

    def parameters_changed(self: mi.Object, keys: List[str] = []) -> None:
        pass

    def traverse(self, callback):
        callback.put_parameter('rng_sequence', self.rng_sequence, mi.ParamFlags.Differentiable)


def change_sequence(x: mi.TensorXf):
    global sequence
    sequence = x