pip install --upgrade pip
pip install -r requirements.txt

python preprocess.py \
--num_train=4950 \
--sp_dim=80 \
--corpus="jsut"

python train.py \
--num_train=4950 \
--sp_dim=80 \
--corpus="jsut"
