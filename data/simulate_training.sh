####
# synthetically inject anomalies into training data to test model robustness.

####################### Spatial anomaly ###############################

echo "simulate polluted training data for differnet anomaly fraction"

for i in 0.05 0.1
do
  python3 ./simulate_training.py --frac_anomaly $i --randomseed 1 --log_dir '../data/train_synthetic_pollute1/'
done


for i in 0.05 0.1
do
  python3 ./simulate_training.py --frac_anomaly $i --randomseed 10 --log_dir '../data/train_synthetic_pollute2/'
done

for i in 0.05 0.1
do
  python3 ./simulate_training.py --frac_anomaly $i --randomseed 100 --log_dir '../data/train_synthetic_pollute3/'
done

for i in 0.05 0.1
do
  python3 ./simulate_training.py --frac_anomaly $i --randomseed 1000 --log_dir '../data/train_synthetic_pollute4/'
done

for i in 0.05 0.1
do
  python3 ./simulate_training.py --frac_anomaly $i --randomseed 10000 --log_dir '../data/train_synthetic_pollute5/'
done

####################### temporal anomaly ###############################

echo "simulate polluted training data for differnet anomaly fraction"

for i in 0.05 0.1
do
  python3 ./simulate_training.py --frac_anomaly $i --randomseed 1 --log_dir '../data/train_synthetic_pollute1/' --pollute_type 'time'
done


for i in 0.05 0.1
do
  python3 ./simulate_training.py --frac_anomaly $i --randomseed 10 --log_dir '../data/train_synthetic_pollute2/' --pollute_type 'time'
done

for i in 0.05 0.1
do
  python3 ./simulate_training.py --frac_anomaly $i --randomseed 100 --log_dir '../data/train_synthetic_pollute3/' --pollute_type 'time'
done

for i in 0.05 0.1
do
  python3 ./simulate_training.py --frac_anomaly $i --randomseed 1000 --log_dir '../data/train_synthetic_pollute4/' --pollute_type 'time'
done

for i in 0.05 0.1
do
  python3 ./simulate_training.py --frac_anomaly $i --randomseed 10000 --log_dir '../data/train_synthetic_pollute5/' --pollute_type 'time'
done
