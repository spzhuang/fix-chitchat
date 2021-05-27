python train.py --device=0 --train_raw_path='./data/train.txt' --raw --epochs=1 --batch_size=2 --train_mmi

python train.py --device=0 --train_raw_path='./data/LCCD.txt' --raw --epochs=1 --batch_size=4 --train_mmi --train_mmi_tokenized_path='./data/lccd_mmi_tokenized.txt'
# 将训练好的模型，进行测试，看效果

python train.py --device=0,1 --train_raw_path='./data/LCCD.txt' --epochs=10 --batch_size=8 --train_mmi --train_mmi_tokenized_path='./data/lccd_mmi_tokenized.txt'  
# 由于已经处理好了raw数据，因此不需要指定--raw参数

python interact_mmi.py --device=0 --mmi_model_path='./mmi_model/mmi_model_epoch3/'

python interact.py --device=0 --dialogue_model_path='./mmi_model/mmi_model_epoch3/'