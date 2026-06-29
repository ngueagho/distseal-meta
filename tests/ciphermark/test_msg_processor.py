import numpy as np
import torch

from distseal.ciphermark.msg_processor import CipherMarkMsgProcessor


def test_shape_preserved():
    proc = CipherMarkMsgProcessor(nbits=256, hidden_size=128, msg_mult=0.5)
    z = torch.randn(2, 128, 8, 8)
    msg = torch.randint(0, 2, (2, 256))
    z_w = proc(z, msg)
    assert z_w.shape == z.shape


def test_zero_msg_repro():
    proc = CipherMarkMsgProcessor(nbits=256, hidden_size=64, msg_mult=1.0)
    z = torch.randn(1, 64, 4, 4)
    msg = torch.zeros(1, 256, dtype=torch.int64)
    z_w1 = proc(z, msg)
    z_w2 = proc(z, msg)
    assert torch.allclose(z_w1, z_w2)


def test_different_msg_diff_output():
    proc = CipherMarkMsgProcessor(nbits=256, hidden_size=64, msg_mult=1.0)
    z = torch.zeros(1, 64, 4, 4)
    msg1 = torch.zeros(1, 256, dtype=torch.int64)
    msg2 = torch.ones(1, 256, dtype=torch.int64)
    z1 = proc(z, msg1)
    z2 = proc(z, msg2)
    assert not torch.allclose(z1, z2)


if __name__ == "__main__":
    test_shape_preserved()
    test_zero_msg_repro()
    test_different_msg_diff_output()
    print("[msg_processor] OK")
