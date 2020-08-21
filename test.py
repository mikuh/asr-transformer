import numpy as np
with open("data/train.tsv", 'r', encoding='utf-8') as f:
    a = f.readline()

a = a.split()
inp = np.reshape(np.array(a[:-1], dtype=np.float64), (-1, 26))
print(inp.shape)
mask = [1 if i < inp.shape[0] else 0 for i in range(1000)]
inp = np.pad(inp, ((0, 1000 - inp.shape[0]), (0, 0)))
print(inp.shape)
print(mask)
tar_inp = [1]
print(a[-1])