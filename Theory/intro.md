Monte Carlo Method: 一个大框架，可以采用多种更新策略
-> Metropolis 策略：一种常用的、基于能量变化和接受概率的更新准则，contains：
1. Random-site update（随机选择格点）
2. Checkerboard update（利用子格结构并行更新，GPU favored）
两者都是 Metropolis 算法的不同实现方式，并且都属于 Monte Carlo 模拟方法的一部分。先介绍常用的Random-site update and its history，然后引入问题：time limitations（64*64 5000 MCS要跑~1天），所以Checkerboard update thus use GPU，convert code to cupy and vectorize。（and their history）
