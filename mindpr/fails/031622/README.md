This is just a checkpoint where I try making  sure negative shuffling is randomized. It doesn't matter, as long as the data iterator is properly randomized with set_epoch. Without setting random.set_seed before, I was getting
```
k=1     k=5     k=20    k=100   filename        num_queries
45.8%   67.6%   79.8%   85.8%   nq-test.csv     3610
```
With the random.set_seed set, I get
```
k=1     k=5     k=20    k=100   filename        num_queries
45.5%   68.2%   79.3%   85.9%   nq-test.csv     3610
```
That's essentially the same performancee, still behind using the DPR data iterator
```
k=1     k=5     k=20    k=100   filename        num_queries
46.0%   68.2%   79.9%   86.4%   nq-test.csv     3610
```
