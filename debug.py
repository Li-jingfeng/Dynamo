# exp2-----------------------------
# 构建新的train_files.txt以及test_files.txt
import os
data_dir = "/data/ljf/pro/waymo_dynamic/waymo/processed/training"
train_files_path = "splits/notr/train_files.txt"
test_files_path = "splits/notr/test_files.txt"
# 获取当前目录下的所有文件和文件夹
contents = os.listdir(data_dir)
folders = [os.path.join(data_dir,item) for item in contents if os.path.isdir(os.path.join(data_dir,item))]
# 挑选两个序列作为测试集 016 and 021
test_folders = [os.path.join(data_dir, '016'), os.path.join(data_dir, '021')]
train_folders = [folder for folder in folders if folder not in test_folders]

all_folders = {'train': train_folders, 'test': test_folders}
write_files = {'train': train_files_path, 'test': test_files_path}
print(len(test_folders),"\n",len(train_folders))
for mode in all_folders:
    with open(write_files[mode],'w') as writer:
        for sub_folder in all_folders[mode]:
            images_path=os.path.join(sub_folder, 'images')
            images_files = os.listdir(images_path)
            last_index = int(len(images_files)/5) - 1
            for image_file in images_files:
                if image_file.split('/')[-1].split('_')[-1] == '0.jpg' and int(image_file.split('/')[-1].split('_')[0]) not in [0, last_index]:# 只要第0个camera
                    writer.write(f"{sub_folder.split('/')[-1]} {image_file.split('/')[-1].split('_')[0]}\n")
    writer.close()     