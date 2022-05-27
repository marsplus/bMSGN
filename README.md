# bMSGN
This is the code to replicate the results reported in the paper ["Learning Binary Multi-Scale Games on Networks"](https://openreview.net/pdf?id=BGe6r8i9x5).
The codebase now supports the replication of Figure 1 and 2; we will add the code to replicate other figures soon.

#### 1. Preparation
- Unzip the files ```data/BTER.zip``` and ```result/BTER.zip```. They are the ```BTER``` network structures used in the experiments.
- Creat three folders: ```result/LikRatio```,  ```result/synthetic``` and ```result/figures```, which will be used to store experimental results.
- Install the needed packages with ```conda create -n bMSGN python=3.7 --file requirements.txt -c mosek -c conda-forge -c pytorch```.
- We used the [Mosek](https://www.mosek.com/) solver to solve the maximum likelihood estimation (MLE) problem in the paper. A license is needed to use the solver; free license is available for students on their [website](https://www.mosek.com/products/academic-licenses/).


#### 2. Generate Figure 1
- Under the ```src/``` folder, run the following to generate the synthetic data and then estimate the game parameters 
  (**the estimation is memory intensive**):
  ```
    ./exp.sh simulation
    ./exp.sh estimation
  ```
- Plot the results by
  ```
    python plot.py --figure=1
  ```
  
#### 3. Generate Figure 2
- Under the ```src/``` folder, run the following to generate the synthetic data and then estimate the game parameters 
  (**the estimation is memory intensive**):
  ```
    ./liktest.sh simulation
    ./liktest.sh liktest
  ```
 - Plot the results by
   ```
     python plot.py --figure=2
   ```


