echo "simulate_test_data for spactial and temporal anomaly, for differnet anomaly fraction"

for i in 0.05 0.1 0.2 
do
  python3 ./simulate_test_data_st.py --frac_anomaly $i --randomseed 1 --log_dir '../data/test_synthetic_Q2/'
done

for i in 0.05 0.1 0.2 
do
  python3 ./simulate_test_data_st.py --frac_anomaly $i --randomseed 10 --log_dir '../data/test_synthetic2_Q2/'
done


for i in 0.05 0.1 0.2
do
  python3 ./simulate_test_data_st.py --frac_anomaly $i --randomseed 100 --log_dir '../data/test_synthetic3_Q2/'
done


for i in 0.05 0.1 0.2 
do
  python3 ./simulate_test_data_st.py --frac_anomaly $i --randomseed 1000 --log_dir '../data/test_synthetic4_Q2/'
done


for i in 0.05 0.1 0.2 
do
  python3 ./simulate_test_data_st.py --frac_anomaly $i --randomseed 10000 --log_dir '../data/test_synthetic5_Q2/'
done