# Source - https://stackoverflow.com/a/25079162
# Posted by Peque, modified by community. See post 'Timeline' for change history
# Retrieved 2026-09-01, License - CC BY-SA 4.0

import pickle

path = 'results_grid_conditional/test3_exp_sepalpha_sepmem_prelec/result.pkl'

with open(path, 'rb') as f:
    data = pickle.load(f)

print(data)