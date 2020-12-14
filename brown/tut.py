import pandas as pd
import numpy as np

xld = pd.read_csv("straightmassD.csv")
xlc = pd.read_csv("straightmassC.csv")
xlb = pd.read_csv("straightmassB.csv")
xla = pd.read_csv("straightmassA.csv")

arr = [xla, xlb, xlc, xld]

for x in arr:
    i = 1
    npx = x['x']
    diffx = np.ediff1d(npx)
    is_smaller = np.all(diffx < 10)
    print(i, ' x ', is_smaller)
    print(np.max(diffx))
    npy = x['y']
    diffy = np.ediff1d(npy)
    is_smaller = np.all(diffy < 10)
    print(i, ' y ', is_smaller)
    print(np.max(diffy))
    print(len(npx))
    i += 1
