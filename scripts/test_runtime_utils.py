from runtime_utils import resolve_device, resolve_dtype

cpu = resolve_device('cpu')
assert str(cpu) == 'cpu'
assert resolve_device('auto').type in {'cpu', 'cuda', 'mps'}
assert str(resolve_dtype('float32', cpu)) == 'torch.float32'
try:
    resolve_dtype('float16', cpu)
except ValueError:
    pass
else:
    raise AssertionError('CPU float16 should be rejected')
print('runtime_utils_smoke=PASS')
