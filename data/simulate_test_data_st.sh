echo "simulate_test_data for spactial and temporal anomaly with 5 diffenrent seeds, for differnet anomaly fraction"

for i in 0.05 0.1 0.2 
do
  python3 ../data/simulate_test_data_st.py --frac_anomaly $i --randomseed 1 --log_dir '../data/test_synthetic/'
done

for i in 0.05 0.1 0.2 
do
  python3 ../data/simulate_test_data_st.py --frac_anomaly $i --randomseed 10 --log_dir '../data/test_synthetic2/'
done


for i in 0.05 0.1 0.2
do
  python3 ../data/simulate_test_data_st.py --frac_anomaly $i --randomseed 100 --log_dir '../data/test_synthetic3/'
done


for i in 0.05 0.1 0.2 
do
  python3 ../data/simulate_test_data_st.py --frac_anomaly $i --randomseed 1000 --log_dir '../data/test_synthetic4/'
done


for i in 0.05 0.1 0.2 
do
  python3 ../data/simulate_test_data_st.py --frac_anomaly $i --randomseed 10000 --log_dir '../data/test_synthetic5/'
done

