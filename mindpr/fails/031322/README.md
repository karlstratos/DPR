# Fail
Using the DPR data iterator works: `./mindpr/run.sh 0,1,2,3,4,5,6,7`
```
k=1     k=5     k=20    k=100   filename        num_queries
46.0%   68.2%   79.1%   86.3%   nq-test.csv     3610
```
This is the same as the DPR's results.
```
k=1     k=5     k=20    k=100   filename        num_queries
46.0%   68.2%   79.1%   86.3%   nq-test.csv     3610
```
I found that DPR always throws away the trailing batches so I did that, but the result using my own loader is still worse (actually more worse now): `./mindpr/run.sh 0,1,2,3,4,5,6,7 --use_my_loader`
```
k=1     k=5     k=20    k=100   filename        num_queries
44.8%   66.9%   78.6%   85.6%   nq-test.csv     3610
```
(vs what I got before not throwing away the trailing batches
```
k=1     k=5     k=20    k=100   filename        num_queries
44.8%   66.7%   78.1%   85.3%   nq-test.csv     3610
```
). I thought this was the fix because small batches may have harmful effects on in-batch negative sampling, but it turns out this is not it.

- TODO: Try emulating their explicit random seed shuffling.
