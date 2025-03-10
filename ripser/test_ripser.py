import ripserplusplus as rpp_py
import numpy as np
from tqdm import tqdm
import sys
from ripser import ripser
import time

start = time.time()
num_iters= 1
for i in tqdm(range(num_iters)):
  d= rpp_py.run("--format point-cloud --dim 2 --sparse",np.random.random((2000,300)))
  if (i%100==0):
    print(d)
    sys.stdout.write('\033[2K\033[1G')
end = time.time()
print("ripser++ total time: ", end-start)