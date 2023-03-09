import rune as rn
import numpy as np

a = np.ones((64, 128)).astype(np.float32)
b = np.ones((64, 128)).astype(np.float32)

buf_a = rn.RBuffer(a)
buf_b = rn.RBuffer(b)

buf_c = buf_a + buf_b

print('==================== TESTING RBuffer ====================')
print('Rust ndarray RBuffer: ', buf_c)

c = buf_c.detach()

print('Python numpy array: ', c)
print(type(c))
print(c.dtype)
print(c.shape)

print('==================== TESTING Tensor ====================')
t_a = rn.Tensor(a)
t_b = rn.Tensor(b)
t_c = rn.tadd(t_a, t_b).detach()
print(t_c)

