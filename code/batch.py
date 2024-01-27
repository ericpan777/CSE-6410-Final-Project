import subprocess
import os
import random
import time

test_files = os.listdir("DATA")

start = time.time()

# random seed within [0,99] to generate best Approx result for each citi
# seed = [1, 52, 32, 5, 26, 20, 10, 16, 20, 74, 39, 0, 43]
random.seed(1)
seed = random.sample(range(1,100),10)

citi = ['Atlanta.tsp', 'Cincinnati.tsp', 'UKansasState.tsp']

for file in test_files:
    if file in citi:
        continue
    else:
        for s in seed: 
            subprocess.run(["python3","TSP.py", "-inst", file, "-alg", "LS", "-time", "100", "-seed", str(s)])

# for f in citi:
#     for i in seed:
#         subprocess.run(["python3","TSP.py", "-inst", f, "-alg", "LS", "-time", "10", "-seed", str(i)])

print('Hours:',(time.time()-start)/3600)