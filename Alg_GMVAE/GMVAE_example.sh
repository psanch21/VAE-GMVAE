
# GMVAE MNIST
python3 GMVAE_main.py --model_type=2 --dataset_name=MNIST --sigma=0.001 --z_dim=8 --w_dim=2 --K_clusters=8 --hidden_dim=64 --num_layers=2 --epochs=20 --batch_size=32 --drop_prob=0.3 --l_rate=0.01 --train=1 --results=1 --plot=1 --restore=1 --early_stopping=1

# GMVAE FREY
python3 GMVAE_main.py --model_type=2 --dataset_name=FREY --sigma=0.001 --z_dim=8 --w_dim=2 --K_clusters=8 --hidden_dim=64 --num_layers=2 --epochs=20 --batch_size=32 --drop_prob=0.3 --l_rate=0.01 --train=1 --results=1 --plot=1 --restore=1 --early_stopping=1

# GMVAECNN MNIST
python3 GMVAE_main.py --model_type=3 --dataset_name=MNIST --sigma=0.001 --z_dim=8 --w_dim=2 --K_clusters=8 --hidden_dim=64 --num_layers=2 --epochs=20 --batch_size=32 --drop_prob=0.3 --l_rate=0.01 --train=1 --results=1 --plot=1 --restore=1 --early_stopping=1
