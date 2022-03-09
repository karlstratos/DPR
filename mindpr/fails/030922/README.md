# Fail
`run.sh` using 8 GPUs produced
```
k=1     k=5     k=20    k=100   filename        num_queries
44.8%   66.7%   78.1%   85.3%   nq-test.csv     3610
```
which is again worse than the DPR's results
```
k=1     k=5     k=20    k=100   filename        num_queries
46.0%   68.2%   79.1%   86.3%   nq-test.csv     3610
```
actually comparable to what I got in the previous version of minDPR (44.7, 67.4, 78.3, 85.3), so it seems I'm still missing something.

- TODO: Use their data iterator, random seeds, follow line by line.
