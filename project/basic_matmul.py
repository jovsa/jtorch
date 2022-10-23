import numpy as np


def mat_mul(a, b):
  assert a.shape[1] == b.shape[0]
  c = np.zeros((a.shape[0], b.shape[1]))

  for i in range(a.shape[0]):
    for j in range(b.shape[1]):
      for k in range(a.shape[1]):
        c[i, j] += a[i, k] * b[k, j]
  return c




if __name__ == "__main__":
  a = np.random.rand(1,4)
  b = np.random.rand(4,2)

  print(mat_mul(a, b) == np.matmul(a, b))