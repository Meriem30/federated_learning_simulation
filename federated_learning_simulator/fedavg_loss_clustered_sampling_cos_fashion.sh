python simulator.py --config-name clustered_sampling_algo1/fashion_mnist.yaml ++clustered_sampling_algo1.worker_number=75
python simulator.py --config-name clustered_sampling_algo1/fashion_mnist.yaml ++clustered_sampling_algo1.worker_number=80
python simulator.py --config-name clustered_sampling_algo1/fashion_mnist.yaml ++clustered_sampling_algo1.worker_number=85
python simulator.py --config-name clustered_sampling_algo1/fashion_mnist.yaml ++clustered_sampling_algo1.worker_number=90
python simulator.py --config-name clustered_sampling_algo1/fashion_mnist.yaml ++clustered_sampling_algo1.worker_number=95
python simulator.py --config-name clustered_sampling_algo1/fashion_mnist.yaml ++clustered_sampling_algo1.worker_number=100

python simulator.py --config-name clustered_sampling_algo2/fashion_mnist.yaml ++clustered_sampling_algo2.worker_number=75
python simulator.py --config-name clustered_sampling_algo2/fashion_mnist.yaml ++clustered_sampling_algo2.worker_number=80
python simulator.py --config-name clustered_sampling_algo2/fashion_mnist.yaml ++clustered_sampling_algo2.worker_number=85
python simulator.py --config-name clustered_sampling_algo2/fashion_mnist.yaml ++clustered_sampling_algo2.worker_number=90
python simulator.py --config-name clustered_sampling_algo2/fashion_mnist.yaml ++clustered_sampling_algo2.worker_number=95
python simulator.py --config-name clustered_sampling_algo2/fashion_mnist.yaml ++clustered_sampling_algo2.worker_number=100


python simulator.py --config-name fed_avg/fashion_mnist.yaml ++fed_avg.worker_number=75
python simulator.py --config-name fed_avg/fashion_mnist.yaml ++fed_avg.worker_number=80
python simulator.py --config-name fed_avg/fashion_mnist.yaml ++fed_avg.worker_number=85
python simulator.py --config-name fed_avg/fashion_mnist.yaml ++fed_avg.worker_number=90
python simulator.py --config-name fed_avg/fashion_mnist.yaml ++fed_avg.worker_number=95
python simulator.py --config-name fed_avg/fashion_mnist.yaml ++fed_avg.worker_number=100


python simulator.py --config-name fed_avg/fashion_mnist.yaml ++fed_avg.worker_number=75 ++fed_avg.algorithm_kwargs.node_random_selection=False ++fed_avg.algorithm_kwargs.loss_client_selection=True ++fed_avg.exp_name="Fash_additional_loss_iid"
python simulator.py --config-name fed_avg/fashion_mnist.yaml ++fed_avg.worker_number=80 ++fed_avg.algorithm_kwargs.node_random_selection=False ++fed_avg.algorithm_kwargs.loss_client_selection=True ++fed_avg.exp_name="Fash_additional_loss_iid"
python simulator.py --config-name fed_avg/fashion_mnist.yaml ++fed_avg.worker_number=85 ++fed_avg.algorithm_kwargs.node_random_selection=False ++fed_avg.algorithm_kwargs.loss_client_selection=True ++fed_avg.exp_name="Fash_additional_loss_iid"
python simulator.py --config-name fed_avg/fashion_mnist.yaml ++fed_avg.worker_number=90 ++fed_avg.algorithm_kwargs.node_random_selection=False ++fed_avg.algorithm_kwargs.loss_client_selection=True ++fed_avg.exp_name="Fash_additional_loss_iid"
python simulator.py --config-name fed_avg/fashion_mnist.yaml ++fed_avg.worker_number=95 ++fed_avg.algorithm_kwargs.node_random_selection=False ++fed_avg.algorithm_kwargs.loss_client_selection=True ++fed_avg.exp_name="Fash_additional_loss_iid"
python simulator.py --config-name fed_avg/fashion_mnist.yaml ++fed_avg.worker_number=100 ++fed_avg.algorithm_kwargs.node_random_selection=False ++fed_avg.algorithm_kwargs.loss_client_selection=True ++fed_avg.exp_name="Fash_additional_loss_iid"


#python simulator.py --config-name graph_fed_avg/fashion_mnist.yaml ++graph_fed_avg.worker_number=75 
#python simulator.py --config-name graph_fed_avg/fashion_mnist.yaml ++graph_fed_avg.worker_number=80
#python simulator.py --config-name graph_fed_avg/fashion_mnist.yaml ++graph_fed_avg.worker_number=85
#python simulator.py --config-name graph_fed_avg/fashion_mnist.yaml ++graph_fed_avg.worker_number=90
#python simulator.py --config-name graph_fed_avg/fashion_mnist.yaml ++graph_fed_avg.worker_number=95
#python simulator.py --config-name graph_fed_avg/fashion_mnist.yaml ++graph_fed_avg.worker_number=100


#try num_neib++ learning_rate++ for graph_fed_avg