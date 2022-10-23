from jtorch import tensor

class BasicTensorData:
  def __init__(self, data, shape=None):
    self.data = data
    self.shape = shape




class BasicTensor:

  def __init__(self, data, shape=None):
    if isinstance(data, BasicTensorData):
      self.data = data.data
      self.shape = data.shape
    else:
      self.shape = (len(data),)
      self.data = data

  def make(self):
    return BasicTensor(BasicTensorData(self.data, self.shape))


def basic_tensor(data, shape=None):
  tensor = BasicTensor(data, shape)
  return tensor.make()





if __name__ == "__main__":
  a = tensor([[1, 2, 3], [4, 5, 6]])
  b = basic_tensor([[1, 2, 3], [4, 5, 6]])
