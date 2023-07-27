# Hair Simulation with DER and IPC

## DER 

For **curly** hair, you may well reset the parameters as `ks_ = 3e3`, `kb_ = 6e-3`, `kt_ = 1e-2` (as is given in `test/Curly` and `test/CurlyGravity`).

I tried hard to adjust the parameters, but still the curly hair would be stretched straight gradually under gravity. Maybe a damping force is needed...

## IPC

Employ the [ipc-toolkit](https://github.com/ipc-sim/ipc-toolkit).

## BUG！

I should not have broken the linear system apart and solved it individually for each hair! It is incorrect and this may be where my bug lies. (But I have no time and energy to fix it now ＞﹏＜.)
