
# Model1 MNIST
python3 VAE.py --epochs=300 --z_dim=10 --hidden_dim=64 --num_layers=3 --sigma=0.001 --num_imgs=300 --model_name=model1 --dataset_name=MNIST --train=1 --restore=0 --plot=0 --generate=1

# Model1 FREY
python3 VAE.py --epochs=300 --z_dim=10 --hidden_dim=64 --num_layers=3 --sigma=0.001 --num_imgs=300 --model_name=model1 --dataset_name=FREY --train=1 --restore=0 --plot=0 --generate=1

# Model2 MNIST
python3 VAE.py --epochs=300 --z_dim=10 --hidden_dim=64 --num_layers=3 --sigma=0.001 --num_imgs=300 --model_name=model2 --dataset_name=MNIST --train=1 --restore=0 --plot=0 --generate=1

# Model2 FREY
python3 VAE.py --epochs=300 --z_dim=10 --hidden_dim=64 --num_layers=3 --sigma=0.001 --num_imgs=300 --model_name=model2 --dataset_name=FREY --train=1 --restore=0 --plot=0 --generate=1

# Model3 MNIST
python3 VAE.py --epochs=300 --z_dim=2 --w_dim=2 --hidden_dim=128 --num_layers=2 --sigma=0.01 --K_clusters=10 --num_imgs=300 --learning_rate=1e-4 --dataset_name=MNIST --model_name=model3  --train=1 --restore=0 --plot=1 --generate=1

# Model3 FREY
python3 VAE.py --epochs=300 --z_dim=2 --w_dim=2 --hidden_dim=128 --num_layers=2 --sigma=0.01 --K_clusters=10 --num_imgs=300 --learning_rate=1e-4 --dataset_name=FREY --model_name=model3 --train=1 --restore=0 --plot=1 --generate=1

# Model4_bias MNIST
python3 GMVAE.py --epochs=300 --z_dim=2 --w_dim=2 --hidden_dim=128 --num_layers=2 --sigma=0.01 --K_clusters=10 --num_imgs=300 --learning_rate=1e-4 --dataset_name=MNIST --model_name=model4_bias --train=1 --restore=0 --plot=1 --generate=1

# Model4_bias FREY
python3 GMVAE.py --epochs=300 --z_dim=2 --w_dim=2 --hidden_dim=128 --num_layers=2 --sigma=0.01 --K_clusters=10 --num_imgs=300 --learning_rate=1e-4 --dataset_name=FREY --model_name=model4_bias --train=1 --restore=0 --plot=1 --generate=1

# Model5 MNIST
python3 GMVAE.py --epochs=300 --sigma=0.001 --z_dim=10 --w_dim=2 --hidden_dim=64 --num_layers=3 --K_clusters=10 --num_imgs=300 --learning_rate=1e-3 --dataset_name=MNIST --model_name=model5 --train=1 --restore=0 --plot=0 --generate=1

# Model5 FREY
python3 GMVAE.py --epochs=300 --sigma=0.001 --z_dim=10 --w_dim=2 --hidden_dim=64 --num_layers=3 --K_clusters=10 --num_imgs=300 --learning_rate=1e-3 --dataset_name=FREY --model_name=model5 --train=1 --restore=0 --plot=0 --generate=1
