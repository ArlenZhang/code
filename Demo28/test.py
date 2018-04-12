import torch
import numpy as np
from torch.autograd import Variable

buffer = torch.from_numpy(np.array([[1, 2, 3], [4, 5, 6]]))
buffer = Variable(buffer, volatile=True).type(torch.FloatTensor)
result = torch.split(buffer, split_size=1, dim=1)
print(result)
