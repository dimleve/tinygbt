# TinyGBT

TinyGBT(Tiny Gradient Boosted Trees) is a 200 line gradient boosted trees implementation written in pure python.

Since this code is not for production, it is not optimized for speed and memory usage.

### Experiment

- Data: [LightGBM's regression example data](https://github.com/Microsoft/LightGBM/tree/master/examples/regression)
- TinyGBT is quite slower than LightGBM, but achieve almost same testset RMSE on similar parameter settings.

| - | LightGBM | TinyGBT |
| --- | --- | --- |
| RMSE of TestSet | 0.45652 | 0.45934 |

#### Reproduce experiment

- TinyGBT

```
git clone https://github.com/lancifollia/tinygbt.git
cd tinygbt
python example.py
```

- LightGBM
    - run [this code](https://github.com/Microsoft/LightGBM/blob/master/examples/python-guide/simple_example.py)


### Features

- For now, Regression with L2 loss supported only.
- Added Binary Classification with Logarithmic Loss (cross-entropy) support 

### Loss functions notes

| Task | target | Loss Function | gradient | hessian |
| --- | --- | --- | --- | --- |
| Regression | continuous | (y-p)^2 | 2*(y-p) | 2 (const)
| Classification | {0,1} | -(y log(p) + (1 - y) log(1 - p)) | p-y | p*(1-p)

[How to find the 1st and 2nd derivates of Log Loss function](https://stats.stackexchange.com/questions/231220/how-to-compute-the-gradient-and-hessian-of-logarithmic-loss-question-is-based)

### References

- [1] T. Chen and C. Guestrin. XGBoost: A Scalable Tree Boosting System. 2016.
- [2] G. Ke et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. 2017.

### License

MIT
