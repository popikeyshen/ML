##https://en.wikipedia.org/wiki/Softmax_function

import numpy as np
a = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]
res = np.exp(a) / np.sum(np.exp(a)) 

print(res)
