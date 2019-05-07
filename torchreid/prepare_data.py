import os
from shutil import copyfile
import pickle
from utils.iotools import *

root_dir = '/home/lzhpc/home_lan/data'
# root_dir = 'D:/0_ZJUAI/AICITY/AI_CITY/data'
# You only need to change this line to your dataset download path
download_path_aicity = root_dir+'/aiCity_s'
# download_path_veri = root_dir+'/VeRi'
save_path = root_dir+'/aiCity_vp'

if not os.path.isdir(save_path):
    os.mkdir(save_path)

make_list(download_path_aicity,'./imglist_b.txt', './labels_b.txt')
# write the train data to pickle
write_pickle_aicity(download_path_aicity)
# write_pickle_veri(download_path_veri)

f_aicity = open('tnames_aic.pkl','rb')
# f_veri = open('tnames_veri.pkl','rb')
tnames_aicity = pickle.load(f_aicity)    # all train data info
# tnames_veri = pickle.load(f_veri)
#print('tnames',len(tnames))
f_aicity.close()
# f_veri.close()

print('aiCity train num imgs: \t',len(tnames_aicity))
# print('VeRi train vid num: \t',(tnames_veri[574]))

# #---------------------------------------
#train_all
train_path = download_path_aicity + '/image_train'
ta_save_path = save_path + '/image_train_all'
copy_ori2dst(tnames_aicity,train_path,ta_save_path)

# #---------------------------------------
# #train_query_gallery
# train_save_path = save_path + '/image_train'
# query_save_path = save_path + '/image_query'
# gallery_save_path = save_path + '/image_test'

# ori2dst_split(tnames_train,train_path,train_save_path)
# ori2dst_split(tnames,train_path,ta_save_path)
# ori2dst_split(tnames_query,train_path,query_save_path)
# ori2dst_split(tnames_gallery,train_path,gallery_save_path)




