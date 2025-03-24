import os
import ctypes

libpath = os.path.abspath("4090.so")
lib = ctypes.CDLL(libpath)
lib.deepgemm.argtypes = [
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,  # (m, n, k)
    ctypes.c_void_p, ctypes.c_void_p,  # (A, AS)
    ctypes.c_void_p, ctypes.c_void_p,  # (B, BS)
    ctypes.c_void_p  # (C)
]

import torch
torch.manual_seed(233)
m = 4096
n = 4096
k = 4096
a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
b = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
c = a @ b.T

def fp8_quantize(x: torch.Tensor, block_m, block_n):
    m, n = x.shape
    assert m % block_m == 0
    assert n % block_n == 0
    m_blocks = m // block_m
    n_blocks = n // block_n
    x = x.reshape(m_blocks, block_m, n_blocks, block_n)
    amax = x.abs().max(dim=1, keepdim=True).values.max(dim=-1, keepdim=True).values.float()
    scale = 448 / amax
    scale_inv = amax / 448
    qx = (x * scale).view(m, n).to(torch.float8_e4m3fn).contiguous()
    dqx = (qx.to(torch.float32).reshape(m_blocks, block_m, n_blocks, block_n) * scale_inv).view(m, n)
    return qx, scale_inv, dqx

qa, qas, dqa = fp8_quantize(a, 1, 128)
qb, qbs, dqb = fp8_quantize(b, 128, 128)

qc = torch.empty_like(c)

lib.deepgemm(
    m, n, k,
    qa.data_ptr(), qas.data_ptr(),
    qb.data_ptr(), qbs.data_ptr(),
    qc.data_ptr()
)

print(c - qc)
dqc = dqa @ dqb.T
print(dqc - qc)

warmup = 10
loop = 300

st = torch.cuda.Event(True)
ed = torch.cuda.Event(True)

for _ in range(warmup):
    lib.deepgemm(
        m, n, k,
        qa.data_ptr(), qas.data_ptr(),
        qb.data_ptr(), qbs.data_ptr(),
        qc.data_ptr()
    )

st.record()

for _ in range(loop):
    lib.deepgemm(
        m, n, k,
        qa.data_ptr(), qas.data_ptr(),
        qb.data_ptr(), qbs.data_ptr(),
        qc.data_ptr()
    )

ed.record()
ed.synchronize()

time_s = st.elapsed_time(ed) / 1000 / loop

print(f"{m * n * k * 2 / time_s / 1e12}TFLOPS")
