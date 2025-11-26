import os
import os.path

if __name__ == "__main__":

    # pic_path = 'data/custom/images/train/'   # 要遍历的图片文件夹路径
    # save_txtfile = open('data/custom/train.txt','w') # 保存路径的记事本文件

    pic_path = r'data/paired10/unpaired_images'  # 要遍历的图片文件夹路径
    train_txtfile = open(r'data/paired10/unpaired_train.txt', 'w')  # 保存路径的记事本文件
    #test_txtfile = open(r'../data/paired10/test.txt', 'w')  # 保存路径的记事本文件
    #val_txtfile = open(r'../data/paired10/val.txt', 'w')  # 保存路径的记事本文件

    # i = 0
    for root, dirs, files in os.walk(pic_path):
        for file in files:
            #if '_3T' in file:
                file = file.replace('.png','')
                print(file)
                #if 'sub-10' in file:
                #   val_txtfile.write(file + '\n')

                #else:
                train_txtfile.write(file + '\n')

    train_txtfile.close()
    #test_txtfile.close()
    #val_txtfile.close()
