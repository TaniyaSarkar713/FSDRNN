To run this code:

(base) taniyamac@Taniyas-MacBook-Air-2 ~ % ssh ts22u@hpc-login.rcc.fsu.edu                                                 

# (base) [ts22u@h22-login-25 ~]$ cd /gpfs/home/ts22u/FSDRNN/code_taniya 
# (base) [ts22u@h22-login-25 code_taniya]$ conda env create -f environment_spd.yml

(base) [ts22u@h22-login-25 code_taniya]$ conda activate spd_frechet
(spd_frechet) [ts22u@h22-login-25 code_taniya]$ sbatch run_frechet.sh 
Submitted batch job 11754644
(spd_frechet) [ts22u@h22-login-25 code_taniya]$ sbatch run_sdr.sh 
Submitted batch job 11754645



