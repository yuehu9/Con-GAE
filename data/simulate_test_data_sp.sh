echo "simulate_test_data for spactial anomaly"

for i in 0.05 0.1 0.2
do
    for j in 0.5 0.25
    do
      python3 ../data/sythetic_test_sp.py --magitude $i --frac_link $j --frac_anomaly 0.1 --randomseed 1 --log_dir '../data/test_synthetic/'
      
      python3 ../data/sythetic_test_sp.py --magitude $i --frac_link $j --frac_anomaly 0.1 --randomseed 10 --log_dir '../data/test_synthetic2/'
      
      python3 ../data/sythetic_test_sp.py --magitude $i --frac_link $j --frac_anomaly 0.1 --randomseed 100 --log_dir '../data/test_synthetic3/'
      
      python3 ../data/sythetic_test_sp.py --magitude $i --frac_link $j --frac_anomaly 0.1 --randomseed 1000 --log_dir '../data/test_synthetic4/'
      
      python3 ../data/sythetic_test_sp.py --magitude $i --frac_link $j --frac_anomaly 0.1 --randomseed 10000 --log_dir '../data/test_synthetic5/'
      
    done
done