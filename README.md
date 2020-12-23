# CSEG6516 Project

## Project Goal
* Adoptiong Reinforcement Learning into Storage System
* based on the papers below
    1. Pan, Y., Jia, Z., Shen, Z., Li, B., Chang, W., & Shao, Z. (2021, December). Reinforcement Learning-Assisted Cache Cleaning to Mitigate Long-Tail Latency in DM-SMR. In 2021 58th ACM/IEEE Design Automation Conference (DAC) (pp. 103-108). IEEE.
    2. Aghayev, A., Shafaei, M., & Desnoyers, P. (2015). Skylight—a window on shingled disk operation. ACM Transactions on Storage (TOS), 11(4), 1-28.

## members
120210397, 120220172, 120220406

## how to reproduce
1. compile the module on the kernel version of linux-4.10-generic.
2. insert module and use device mappers to simulate SMR on CMR HDD(at least 1tb), 24-hr trace for each 
```
sudo dmsetup create target_module --table "0 20684800 sadc /dev/sd$2 1048576 10 1 10737418240
```
3. get blk traces from http://iotta.snia.org/traces/block-io/391
4. use block trace replayer to replay the traces (take at least a few hours)
