import numpy as np
import os
import SimpleITK as sitk
from PIL import Image
import shutil
from tqdm import tqdm


def colour(label, ra, ga, ba):
    size = np.shape(label)
    # print(max(np.all(label)))
    # print(min(np.all(label)))

    for i in range(size[0]):
        for j in range(size[1]):
            if label[i][j] == 1.0:
                ra[i][j] = 255
                ga[i][j] = 0
                ba[i][j] = 0
            elif label[i][j] == 2.0:
                ra[i][j] = 0
                ga[i][j] = 255
                ba[i][j] = 0
            elif label[i][j] == 3.0:
                ra[i][j] = 0
                ga[i][j] = 0
                ba[i][j] = 255
            elif label[i][j] == 4.0:
                ra[i][j] = 255
                ga[i][j] = 255
                ba[i][j] = 0
            elif label[i][j] == 5.0:
                ra[i][j] = 255
                ga[i][j] = 0
                ba[i][j] = 255
            elif label[i][j] == 6.0:
                ra[i][j] = 0
                ga[i][j] = 255
                ba[i][j] = 255
            elif label[i][j] == 7.0:
                ra[i][j] = 112
                ga[i][j] = 48
                ba[i][j] = 160
            elif label[i][j] == 8.0:
                ra[i][j] = 255
                ga[i][j] = 192
                ba[i][j] = 0
    return ra, ga, ba


def make_img(ra, ga, ba, img_array, label_path, rows):
    image = Image.fromarray(np.uint8(img_array * 255)).convert('RGB')
    shape = np.shape(image)

    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            if ra[i][j] != 0 or ga[i][j] != 0 or ba[i][j] != 0:
                image.putpixel((i, j), (ra[i][j], ga[i][j], ba[i][j]))

    image.save(label_path)


def RGB(raw_path, res_path):
    for i in os.listdir(raw_path):
        case = os.path.join(raw_path, i)
        case_name = i[0:7]
        ds_array = sitk.ReadImage(case)  # 读取dicom文件的相关信息
        img_array = sitk.GetArrayFromImage(ds_array)  # 获取array
        img_array = img_array.transpose(1, 2, 0)

        # print(np.max(img_array))
        # print(np.min(img_array))

        print(np.shape(img_array))

        # img_size=np.shape(img_array)
        # for slice in range(img_size[2]):
        #     ra=np.zeros((raws,raws),dtype=int)
        #     ga=np.zeros((raws,raws),dtype=int)
        #     ba=np.zeros((raws,raws),dtype=int)
        #     ra,ga,ba=colour(img_array[:,:,slice],ra,ga,ba)
        #     img_path=os.path.join(gen_path,case_name+'_'+str(slice)+'.png')
        # print(img_path)
        # make_img(ra,ga,ba,img_path,raws)


def RGB_case(img_path, label_path, res_path):
    raws = 512
    dirpath, filepath = os.path.split(label_path)
    case_name = filepath[0:8]

    img_array = sitk.ReadImage(img_path)  # 读取dicom文件的相关信息
    img_array = sitk.GetArrayFromImage(img_array)  # 获取array
    img_array = img_array.transpose(2, 1, 0)

    label_array = sitk.ReadImage(label_path)  # 读取dicom文件的相关信息
    label_array = sitk.GetArrayFromImage(label_array)  # 获取array
    label_array = label_array.transpose(1, 2, 0)

    print(np.unique(label_array))

    print(np.shape(label_array))
    img_size = np.shape(label_array)
    start = 0

    for slice in range(start, img_size[2]):
        print('slice:' + str(slice))
        ra = np.zeros((raws, raws), dtype=int)
        ga = np.zeros((raws, raws), dtype=int)
        ba = np.zeros((raws, raws), dtype=int)
        print('label')
        print(np.unique(label_array[:, :, slice]))

        ra, ga, ba = colour(label_array[:, :, slice], ra, ga, ba)
        print('RGB')
        print(max(np.unique(ra)))
        print(min(np.unique(ga)))
        print(max(np.unique(ba)))

        label_path = os.path.join(res_path, case_name + '_' + str(slice) + '.png')
        # print(img_path)
        make_img(ra, ga, ba, img_array[:, :, slice], label_path, raws)


def mk_case(img_path, label_path, res_path):
    raws = 512
    dirpath, filepath = os.path.split(label_path)
    case_name = filepath[0:8]

    img_array = sitk.ReadImage(img_path)  # 读取dicom文件的相关信息
    img_array = sitk.GetArrayFromImage(img_array)  # 获取array
    img_array = img_array.transpose(2, 1, 0)

    label_array = sitk.ReadImage(label_path)  # 读取dicom文件的相关信息
    label_array = sitk.GetArrayFromImage(label_array)  # 获取array
    label_array = label_array.transpose(1, 2, 0)

    print(np.unique(label_array))

    print(np.shape(label_array))
    img_size = np.shape(label_array)
    start = 0

    for slice in range(start, img_size[2]):
        print('slice:' + str(slice))
        ra = np.zeros((raws, raws), dtype=int)
        ga = np.zeros((raws, raws), dtype=int)
        ba = np.zeros((raws, raws), dtype=int)
        print('label')
        # print(np.unique(label_array[:,:,slice]))

        # ra,ga,ba=colour(label_array[:,:,slice],ra,ga,ba)
        # print('RGB')
        # print(max(np.unique(ra)))
        # print(min(np.unique(ga)))
        # print(max(np.unique(ba)))

        label_path = os.path.join(res_path, case_name + '_' + str(slice) + '.png')
        # print(img_path)
        make_img(ra, ga, ba, img_array[:, :, slice], label_path, raws)


if __name__ == '__main__':
    # pre_path='/storage/student1/sc/code/segcode/transformer/MISSFormer-main/MISSFormer-main/output/predictions'
    # pre_path=os.path.join(gen_path,'pre')
    # label_path=os.path.join(gen_path,'label')

    # raw_pre_path='/storage/student1/sc/node6/ppcode/Multi-Scale-Attention-master/results/test_de_CGNet/result/pred'
    # raw_label_path='/storage/student1/sc/node6/ppcode/Multi-Scale-Attention-master/results/test_de_CGNet/result/label'

    # os.rmdir(pre_path)
    # os.rmdir(label_path)
    # shutil.rmtree(pre_path)
    # shutil.rmtree(label_path)

    # if os.path.exists(pre_path)==False:
    #     os.mkdir(pre_path)

    # if os.path.exists(label_path)==False:
    #     os.mkdir(label_path)

    # RGB(raw_label_path,label_path)
    # RGB(raw_pre_path,pre_path)
    orgin_path = r'/root/mymodel/FSA_edges/model_out'

    # img_path = os.path.join(orgin_path,'predictions')

    pre_orgin_path = os.path.join(orgin_path, 'predictions')

    # pre_orgin_path = orgin_path

    predict_path = os.path.join(orgin_path, 'pre')
    label_path = os.path.join(orgin_path, 'label')
    image_path = os.path.join(orgin_path, 'img')

    # predict_o1_path=os.path.join(orgin_path,'pre_o1')
    # predict_o2_path=os.path.join(orgin_path,'pre_o2')
    # predict_o3_path=os.path.join(orgin_path,'pre_o3')
    # predict_o4_path=os.path.join(orgin_path,'pre_o4')

    # if os.path.exists(pre_orgin_path)==False:
    #     os.mkdir(pre_orgin_path)

    if os.path.exists(predict_path) == False:
        os.mkdir(predict_path)

    if os.path.exists(label_path) == False:
        os.mkdir(label_path)

    if os.path.exists(image_path) == False:
        os.mkdir(image_path)

    # if os.path.exists(predict_o1_path)==False:
    #     os.mkdir(predict_o1_path)

    # if os.path.exists(predict_o2_path)==False:
    #     os.mkdir(predict_o2_path)

    # if os.path.exists(predict_o3_path)==False:
    #     os.mkdir(predict_o3_path)

    # if os.path.exists(predict_o4_path)==False:
    #     os.mkdir(predict_o4_path)

    case_list = []
    for i in tqdm(sorted(os.listdir(pre_orgin_path))):
        case_name = i[0:8]
        if case_name in case_list:
            print('111')
            continue

        case_list.append(case_name)

        img = case_name + '_img.nii.gz'
        pre = case_name + '_pred.nii.gz'
        gt = case_name + '_gt.nii.gz'

        img_path = os.path.join(pre_orgin_path, img)
        pre_path = os.path.join(pre_orgin_path, pre)
        gt_path = os.path.join(pre_orgin_path, gt)

        # RGB_case(img_path,gt_path,label_path)
        # mk_case(img_path,gt_path,image_path)
        RGB_case(img_path, pre_path, predict_path)

        # pre_o1 = case_name+'_pred_o1.nii.gz'
        # pre_o2 = case_name+'_pred_02.nii.gz'
        # pre_o3 = case_name+'_pred_03.nii.gz'
        # pre_o4 = case_name+'_pred_04.nii.gz'

        # pre_o1_path = os.path.join(pre_orgin_path,pre_o1)
        # pre_o2_path = os.path.join(pre_orgin_path,pre_o2)
        # pre_o3_path = os.path.join(pre_orgin_path,pre_o3)
        # pre_o4_path = os.path.join(pre_orgin_path,pre_o4)

        # RGB_case(img_path,pre_o1_path,predict_o1_path)
        # RGB_case(img_path,pre_o2_path,predict_o2_path)
        # RGB_case(img_path,pre_o3_path,predict_o3_path)
        # RGB_case(img_path,pre_o4_path,predict_o4_path)
    os.system("shutdown")