import os
import logging
import time

def console_out(logFilename,content):
    ''''' Output log to file and console '''
    # Define a Handler and set a format which output to file
    logging.basicConfig(
        level=logging.INFO,  # 定义输出到文件的log级别，大于此级别的都被输出
        format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s',  # 定义输出log的格式
        datefmt='%Y-%m-%d %A %H:%M:%S',  # 时间
        filename=logFilename,  # log文件名
        filemode='a')  # 写入模式“w”或“a”
    # Define a Handler and set a format which output to console
    console = logging.StreamHandler()  # 定义console handler
    console.setLevel(logging.INFO)  # 定义该handler级别
    formatter = logging.Formatter('%(asctime)s  %(filename)s : %(levelname)s  %(message)s')  # 定义该handler格式
    console.setFormatter(formatter)
    # Create an instance
    logging.getLogger().addHandler(console)  # 实例化添加handler
 
    # Print information              # 输出日志级别
    logging.info(content)



for i in range(0,2): 
    console_out('log/logging.log',"math {} begin".format(i))
    # os.system(r'python3 train.py --time_steps 25 --dataset as --GPU_ID 0 --epochs 100 --batch_size 1024  --seed {}'.format(i))
    # os.system(r'python3 train.py --time_steps 8 --dataset bitcoinotc --GPU_ID 1 --epochs 100 --batch_size 1024 --seed {}'.format(i))
    # os.system(r'python3 train.py --time_steps 26 --dataset math --GPU_ID 1 --epochs 100 --batch_size 1024 --seed {}'.format(i))
    # os.system(r'python3 train.py --time_steps 26 --dataset mooc --GPU_ID 1 --epochs 100 --batch_size 1024  --seed {}'.format(i))
    
    # os.system(r'python3 dyngraph2vec_train.py --time_steps 6 --dataset email --gpu 0 --epochs 100  --seed {}'.format(i))
    # os.system(r'python3 dyngraph2vec_train.py --time_steps 7 --dataset uci --gpu 0 --epochs 100  --seed {}'.format(i))
    # os.system(r'python3 dyngraph2vec_train.py --time_steps 8 --dataset bitcoinotc --gpu 0 --epochs 100  --seed {}'.format(i))
    # os.system(r'python3 dyngraph2vec_train.py --time_steps 14 --dataset wiki --gpu 1 --epochs 100  --seed {}'.format(i))
    # os.system(r'python3 dyngraph2vec_train.py --time_steps 18 --dataset reality_call --gpu 1 --epochs 100  --seed {}'.format(i))
    # os.system(r'python3 dyngraph2vec_train.py --time_steps 26 --dataset math --gpu 0 --epochs 100  --seed {}'.format(i))
    # os.system(r'python3 dyngraph2vec_train.py --time_steps 25 --dataset as --gpu 1 --epochs 100  --seed {}'.format(i))
    # os.system(r'python3 dyngraph2vec_train_multisteps.py --time_steps 14 --dataset wiki --gpu 1 --epochs 100  --seed {}'.format(i))
    # os.system(r'python3 dyngraph2vec_train_multisteps.py --time_steps 18 --dataset reality_call --gpu 1 --epochs 100  --seed {}'.format(i))
    
    
    # os.system(r'python STGSN_train.py --time_steps 6 --dataset email --gpu 0 --epochs 100  --seed {}'.format(i))
    # os.system(r'python STGSN_train.py --time_steps 7 --dataset uci --gpu 0 --epochs 100  --seed {}'.format(i))
    # os.system(r'python STGSN_train.py --time_steps 8 --dataset bitcoinotc --gpu 0 --epochs 100  --seed {}'.format(i))
    # os.system(r'python STGSN_train.py --time_steps 26 --dataset math --gpu 0 --epochs 100  --seed {}'.format(i))
    # os.system(r'python3 STGSN_train.py --time_steps 18 --dataset reality_call --gpu 0 --epochs 100  --seed {}'.format(i))
    # os.system(r'python3 STGSN_train.py --time_steps 14 --dataset wiki --gpu 1 --epochs 100  --seed {}'.format(i))
    # os.system(r'python3 STGSN_train_multisteps.py --time_steps 18 --dataset reality_call --gpu 1 --epochs 100  --seed {}'.format(i))
    # os.system(r'python3 STGSN_train_multisteps.py --time_steps 14 --dataset wiki --gpu 0 --epochs 100  --seed {}'.format(i))
    
    
    # os.system(r'python3 EvolveGCN_train.py --time_steps 6 --dataset email --gpu 0 --epochs 100  --egcn_type EGCNH --seed {}'.format(i))
    # os.system(r'python3 EvolveGCN_train.py --time_steps 7 --dataset uci --gpu 0 --epochs 100 --egcn_type EGCNH --seed {}'.format(i))
    # os.system(r'python3 EvolveGCN_train.py --time_steps 8 --dataset bitcoinotc --gpu 0 --epochs 100  --egcn_type EGCNH --seed {}'.format(i))
    # os.system(r'python3 EvolveGCN_train.py --time_steps 18 --dataset reality_call --gpu 1 --epochs 100  --egcn_type EGCNH --seed {}'.format(i))
    # os.system(r'python3 EvolveGCN_train.py --time_steps 14 --dataset wiki --gpu 1 --epochs 100  --egcn_type EGCNH --seed {}'.format(i))
    # os.system(r'python3 EvolveGCN_train_multisteps.py --time_steps 18 --dataset reality_call --gpu 0 --epochs 100  --egcn_type EGCNH --seed {}'.format(i))
    # os.system(r'python3 EvolveGCN_train_multisteps.py --time_steps 14 --dataset wiki --gpu 0 --epochs 100  --egcn_type EGCNH --seed {}'.format(i))
    
    # os.system(r'python3 gru_gcn_train.py --time_steps 6 --dataset email --gpu 0 --epochs 100  --seed {}'.format(i))
    # os.system(r'python3 gru_gcn_train.py --time_steps 7 --dataset uci --gpu 0 --epochs 100 --seed {}'.format(i))
    # os.system(r'python3 gru_gcn_train.py --time_steps 8 --dataset bitcoinotc --gpu 0 --epochs 100  --seed {}'.format(i))
    # os.system(r'python3 gru_gcn_train.py --time_steps 18 --dataset reality_call --gpu 1 --epochs 100 --seed {}'.format(i))
    # os.system(r'python3 gru_gcn_train.py --time_steps 14 --dataset wiki --gpu 1 --epochs 100  --seed {}'.format(i))
    # os.system(r'python3 gru_gcn_train_multisteps.py --time_steps 18 --dataset reality_call --gpu 0 --epochs 100 --seed {}'.format(i))
    # os.system(r'python3 gru_gcn_train_multisteps.py --time_steps 14 --dataset wiki --gpu 0 --epochs 100  --seed {}'.format(i))
    
        
    # os.system(r'python3 NDCN_train.py --time_steps 7 --dataset uci --gpu 1 --epochs 100  --seed {}'.format(i))
    # os.system(r'python3 NDCN_train.py --time_steps 6 --dataset email --gpu 0 --epochs 100   --seed {}'.format(i))
    # os.system(r'python3 NDCN_train.py --time_steps 8 --dataset bitcoinotc --gpu 0 --epochs 100  --seed {}'.format(i))
    # os.system(r'python3 NDCN_train.py --time_steps 14 --dataset wiki --gpu 0 --epochs 100   --seed {}'.format(i))
    # os.system(r'python3 NDCN_train.py --time_steps 18 --dataset reality_call --gpu 0 --epochs 100   --seed {}'.format(i))
    # os.system(r'python3 NDCN_train.py --time_steps 27 --dataset reddit --gpu 0 --epochs 100   --seed {}'.format(i))
    # os.system(r'python3 NDCN_train.py --time_steps 26 --dataset math --gpu 0 --epochs 100   --seed {}'.format(i))
    # os.system(r'python3 NDCN_train_multisteps.py --time_steps 14 --dataset wiki --gpu 1 --epochs 100   --seed {}'.format(i))
    # os.system(r'python3 NDCN_train_multisteps.py --time_steps 18 --dataset reality_call --gpu 0 --epochs 100   --seed {}'.format(i))
    
    # os.system(r'python3 NDCN_train_multisteps.py --time_steps 14 --dataset wiki --gpu 0 --multisteps_pre True --epochs 100   --seed {}'.format(i))
    # os.system(r'python3 NDCN_train_multisteps.py --time_steps 18 --dataset reality_call --gpu 0 --multisteps_pre True --epochs 100   --seed {}'.format(i))
    # os.system(r'python3 dyngraph2vec_train_multisteps.py --time_steps 18 --dataset reality_call --gpu 0 --multisteps_pre True --epochs 100   --seed {}'.format(i))
    # os.system(r'python3 NDCN_train_multisteps.py --time_steps 27 --dataset reddit  --gpu 0 --multisteps_pre True --epochs 100   --seed {}'.format(i))
    # os.system(r'python3 NDCN_train_multisteps.py --time_steps 26 --dataset math  --gpu 0 --multisteps_pre True --epochs 100   --seed {}'.format(i))
    # os.system(r'python3 gru_gcn_train_multisteps.py --time_steps 19 --dataset reddit  --gpu 0  --epochs 100 --multisteps_pre True  --seed {}'.format(i))
    # os.system(r'python3 EvolveGCN_train_multisteps.py --time_steps 21 --dataset math  --gpu 0  --epochs 100 --multisteps_pre True  --seed {}'.format(i))
    # os.system(r'python3 dyngraph2vec_datascare.py  --time_steps 27 --dataset reddit --gpu 0 --epochs 100 --batch_size 1024 --tasktype data_scarce --scare_snapshot 20,21,22,23,24 --seed {}'.format(i))
    # os.system(r'python3 STGSN_datascare.py  --time_steps 27 --dataset reddit --gpu 0 --epochs 100 --batch_size 1024 --tasktype data_scarce --scare_snapshot 20,21,22,23,24 --seed {}'.format(i))
    # os.system(r'python3 NDCN_datascare.py  --time_steps 27 --dataset reddit --gpu 0 --epochs 100 --batch_size 1024 --tasktype data_scarce --scare_snapshot 20,21,22,23,24 --seed {}'.format(i))
    # os.system(r'python3 EvolveGCN_datascare.py  --time_steps 27 --dataset reddit --gpu 0 --epochs 100 --batch_size 1899 --tasktype data_scarce --scare_snapshot 20,21,22,23,24 --seed {}'.format(i))
    # os.system(r'python3 gru_gcn_datascare.py  --time_steps 27 --dataset reddit --gpu 0 --epochs 100 --batch_size 1024 --tasktype data_scarce --scare_snapshot 20,21,22,23,24 --seed {}'.format(i))
    # os.system(r'python3 gru_gcn_datascare.py  --time_steps 8 --dataset bitcoinotc --gpu 0 --epochs 100 --batch_size 1024 --tasktype data_scarce --scare_snapshot 5,6 --seed {}'.format(i))
    
    
    
    # os.system(r'python3 dynamic.py --gpu 0 --network grid --physical_equation gene --seed {}'.format(i))
    # os.system(r'python3 dynamic.py --gpu 0 --network grid --physical_equation heat --seed {}'.format(i))
    # os.system(r'python3 dynamic.py --gpu 0 --network grid --physical_equation mutualistic --seed {}'.format(i))
    
    # os.system(r'python3 dynamic.py --gpu 0 --network random --physical_equation gene --seed {}'.format(i))
    # os.system(r'python3 dynamic.py --gpu 0 --network random --physical_equation heat --seed {}'.format(i))
    # os.system(r'python3 dynamic.py --gpu 0 --network random --physical_equation mutualistic --seed {}'.format(i))
    
    # os.system(r'python3 dynamic.py --gpu 0 --network power_law --physical_equation gene --seed {}'.format(i))
    # os.system(r'python3 dynamic.py --gpu 1 --network power_law --physical_equation heat --seed {}'.format(i))
    # os.system(r'python3 dynamic.py --gpu 1 --network power_law --physical_equation mutualistic --seed {}'.format(i))
    
    # os.system(r'python3 dynamic.py --gpu 1 --network small_world --physical_equation gene --seed {}'.format(i))
    # os.system(r'python3 dynamic.py --gpu 1 --network small_world --physical_equation heat --seed {}'.format(i))
    # os.system(r'python3 dynamic.py --gpu 1 --network small_world --physical_equation mutualistic --seed {}'.format(i))
    
    # os.system(r'python3 dynamic.py --gpu 1 --network community --physical_equation gene --seed {}'.format(i))
    # os.system(r'python3 dynamic.py --gpu 1 --network community --physical_equation heat --seed {}'.format(i))
    # os.system(r'python3 dynamic.py --gpu 1 --network community --physical_equation mutualistic --seed {}'.format(i))
    # os.system(r'python3 dyngraph2vec_datascare.py  --time_steps 6 --dataset email --gpu 0 --epochs 100 --batch_size 1891 --tasktype data_scarce --scare_snapshot 2,3 --seed {}'.format(i))
    # os.system(r'python3 dyngraph2vec_datascare.py  --time_steps 7 --dataset uci --gpu 0 --epochs 100 --batch_size 1899 --tasktype data_scarce --scare_snapshot 3,4 --seed {}'.format(i))
    # os.system(r'python3 dyngraph2vec_datascare.py  --time_steps 8 --dataset bitcoinotc --gpu 0 --epochs 100 --batch_size 1024 --tasktype data_scarce --scare_snapshot 4,5 --seed {}'.format(i))
    
    # os.system(r'python3 dyngraph2vec_datascare.py  --time_steps 18 --dataset reality_call --gpu 1 --epochs 100 --batch_size 1024 --tasktype data_scarce --scare_snapshot 12,13,14,15 --seed {}'.format(i))
    # os.system(r'python3 dyngraph2vec_datascare.py  --time_steps 14 --dataset wiki --gpu 1 --epochs 100 --batch_size 1024 --tasktype data_scarce --scare_snapshot 9,10,11 --seed {}'.format(i))
    
    # os.system(r'python3 EvolveGCN_datascare.py  --time_steps 6 --dataset email --gpu 0 --epochs 100 --batch_size 1891 --tasktype data_scarce --scare_snapshot 2,3 --seed {}'.format(i))
    # os.system(r'python3 EvolveGCN_datascare.py  --time_steps 7 --dataset uci --gpu 0 --epochs 100 --batch_size 1899 --tasktype data_scarce --scare_snapshot 3,4 --seed {}'.format(i))
    # os.system(r'python3 EvolveGCN_datascare.py  --time_steps 8 --dataset bitcoinotc --gpu 0 --epochs 100 --batch_size 1024 --tasktype data_scarce --scare_snapshot 4,5 --seed {}'.format(i))
    
    # os.system(r'python3 EvolveGCN_datascare.py  --time_steps 18 --dataset reality_call --gpu 1 --epochs 100 --batch_size 1024 --tasktype data_scarce --scare_snapshot 12,13,14,15 --seed {}'.format(i))
    # os.system(r'python3 EvolveGCN_datascare.py  --time_steps 14 --dataset wiki --gpu 1 --epochs 100 --batch_size 1024 --tasktype data_scarce --scare_snapshot 9,10,11 --seed {}'.format(i))    
    
    
    # os.system(r'python3 gru_gcn_datascare.py  --time_steps 6 --dataset email --gpu 0 --epochs 100 --batch_size 1891 --tasktype data_scarce --scare_snapshot 2,3 --seed {}'.format(i))
    # os.system(r'python3 gru_gcn_datascare.py  --time_steps 7 --dataset uci --gpu 0 --epochs 100 --batch_size 1899 --tasktype data_scarce --scare_snapshot 3,4 --seed {}'.format(i))
    # os.system(r'python3 gru_gcn_datascare.py  --time_steps 8 --dataset bitcoinotc --gpu 0 --epochs 100 --batch_size 1024 --tasktype data_scarce --scare_snapshot 4,5 --seed {}'.format(i))
    
    # os.system(r'python3 gru_gcn_datascare.py  --time_steps 18 --dataset reality_call --gpu 1 --epochs 100 --batch_size 1024 --tasktype data_scarce --scare_snapshot 12,13,14,15 --seed {}'.format(i))
    # os.system(r'python3 gru_gcn_datascare.py  --time_steps 14 --dataset wiki --gpu 1 --epochs 100 --batch_size 1024 --tasktype data_scarce --scare_snapshot 9,10,11 --seed {}'.format(i))    
    
    # os.system(r'python3 NDCN_datascare.py  --time_steps 6 --dataset email --gpu 0 --epochs 100 --batch_size 1891 --tasktype data_scarce --scare_snapshot 2,3 --seed {}'.format(i))
    # os.system(r'python3 NDCN_datascare.py  --time_steps 7 --dataset uci --gpu 0 --epochs 100 --batch_size 1899 --tasktype data_scarce --scare_snapshot 3,4 --seed {}'.format(i))
    # os.system(r'python3 NDCN_datascare.py  --time_steps 8 --dataset bitcoinotc --gpu 0 --epochs 100 --batch_size 1024 --tasktype data_scarce --scare_snapshot 4,5 --seed {}'.format(i))
    
    # os.system(r'python3 NDCN_datascare.py  --time_steps 18 --dataset reality_call --gpu 0 --epochs 100 --batch_size 1024 --tasktype data_scarce --scare_snapshot 12,13,14,15 --seed {}'.format(i))
    # os.system(r'python3 NDCN_datascare.py  --time_steps 14 --dataset wiki --gpu 0 --epochs 100 --batch_size 1024 --tasktype data_scarce --scare_snapshot 9,10,11 --seed {}'.format(i))

    # os.system(r'python3 STGSN_datascare.py  --time_steps 6 --dataset email --gpu 0 --epochs 100 --batch_size 1891 --tasktype data_scarce --scare_snapshot 2,3 --seed {}'.format(i))
    # os.system(r'python3 STGSN_datascare.py  --time_steps 7 --dataset uci --gpu 0 --epochs 100 --batch_size 1899 --tasktype data_scarce --scare_snapshot 3,4 --seed {}'.format(i))
    # os.system(r'python3 STGSN_datascare.py  --time_steps 8 --dataset bitcoinotc --gpu 0 --epochs 100 --batch_size 1024 --tasktype data_scarce --scare_snapshot 4,5 --seed {}'.format(i))
    
    # os.system(r'python3 STGSN_datascare.py  --time_steps 18 --dataset reality_call --gpu 1 --epochs 100 --batch_size 1024 --tasktype data_scarce --scare_snapshot 12,13,14,15 --seed {}'.format(i))
    # os.system(r'python3 STGSN_datascare.py  --time_steps 14 --dataset wiki --gpu 1 --epochs 100 --batch_size 1024 --tasktype data_scarce --scare_snapshot 9,10,11 --seed {}'.format(i))        
    
    # os.system(r'python3 dyngraph2vec_user.py --time_steps 6 --dataset email --gpu 0 --epochs 100  --seed {}'.format(i))
    # os.system(r'python3 dyngraph2vec_user.py --time_steps 7 --dataset uci --gpu 0 --epochs 100  --seed {}'.format(i))
    # os.system(r'python3 dyngraph2vec_user.py --time_steps 8 --dataset bitcoinotc --gpu 0 --epochs 100  --seed {}'.format(i))
    # os.system(r'python3 dyngraph2vec_user.py --time_steps 18 --dataset reality_call --gpu 1 --epochs 100   --seed {}'.format(i))
    # os.system(r'python3 dyngraph2vec_user.py --time_steps 14 --dataset wiki --gpu 1 --epochs 100   --seed {}'.format(i))
    
    
    # os.system(r'python3 EvolveGCN_user.py --time_steps 6 --dataset email --gpu 1 --epochs 100  --seed {}'.format(i))
    # os.system(r'python3 EvolveGCN_user.py --time_steps 7 --dataset uci --gpu 1 --epochs 100  --seed {}'.format(i))
    # os.system(r'python3 EvolveGCN_user.py --time_steps 8 --dataset bitcoinotc --gpu 1 --epochs 100  --seed {}'.format(i))
    # os.system(r'python3 EvolveGCN_user.py --time_steps 18 --dataset reality_call --gpu 1 --epochs 100   --seed {}'.format(i))
    # os.system(r'python3 EvolveGCN_user.py --time_steps 7 --dataset wiki --gpu 1 --epochs 100   --seed {}'.format(i))
    
    # os.system(r'python3 NDCN_user.py --time_steps 6 --dataset email --gpu 1 --epochs 100  --seed {}'.format(i))
    # os.system(r'python3 NDCN_user.py --time_steps 7 --dataset uci --gpu 1 --epochs 100  --seed {}'.format(i))
    # os.system(r'python3 NDCN_user.py --time_steps 8 --dataset bitcoinotc --gpu 1 --epochs 100  --seed {}'.format(i))
    # os.system(r'python3 NDCN_user.py --time_steps 18 --dataset reality_call --gpu 1 --epochs 100   --seed {}'.format(i))
    # os.system(r'python3 NDCN_user.py --time_steps 14 --dataset wiki --gpu 1 --epochs 100   --seed {}'.format(i))

    
    # os.system(r'python3 STGSN_user.py --time_steps 6 --dataset email --gpu 0 --epochs 100  --seed {}'.format(i))
    # os.system(r'python3 STGSN_user.py --time_steps 7 --dataset uci --gpu 0 --epochs 100  --seed {}'.format(i))
    # os.system(r'python3 STGSN_user.py --time_steps 8 --dataset bitcoinotc --gpu 0 --epochs 100  --seed {}'.format(i))
    # os.system(r'python3 STGSN_user.py --time_steps 18 --dataset reality_call --gpu 1 --epochs 100   --seed {}'.format(i))
    # os.system(r'python3 STGSN_user.py --time_steps 14 --dataset wiki --gpu 1 --epochs 100   --seed {}'.format(i))
    
    # os.system(r'python3 gru_gcn_user.py --time_steps 6 --dataset email --gpu 0 --epochs 100  --seed {}'.format(i))
    # os.system(r'python3 gru_gcn_user.py --time_steps 7 --dataset uci --gpu 0 --epochs 100  --seed {}'.format(i))
    # os.system(r'python3 gru_gcn_user.py --time_steps 8 --dataset bitcoinotc --gpu 0 --epochs 100  --seed {}'.format(i))
    # os.system(r'python3 gru_gcn_user.py --time_steps 10 --dataset reality_call --gpu 1 --epochs 100   --seed {}'.format(i))
    # os.system(r'python3 gru_gcn_user.py --time_steps 7 --dataset wiki --gpu 1 --epochs 100   --seed {}'.format(i))

    
    os.system(r'python3 dyngraph2vec_user.py  --time_steps 27 --dataset reddit --gpu 1 --epochs 100 --batch_size 1024  --seed {}'.format(i))
    os.system(r'python3 STGSN_user.py  --time_steps 27 --dataset reddit --gpu 1 --epochs 100 --batch_size 1024  --seed {}'.format(i))
    os.system(r'python3 NDCN_user.py  --time_steps 27 --dataset reddit --gpu 1 --epochs 100 --batch_size 1024  --seed {}'.format(i))
    os.system(r'python3 EvolveGCN_user.py  --time_steps 27 --dataset reddit --gpu 1 --epochs 100 --batch_size 1024  --seed {}'.format(i))
    # os.system(r'python3 gru_gcn_datascare.py  --time_steps 27 --dataset reddit --gpu 0 --epochs 100 --batch_size 1024 --tasktype data_scarce --scare_snapshot 20,21,22,23,24 --seed {}'.format(i))
    
    console_out('log/logging.log',"as seed{} finish".format(i))
    



