%%-------------------------------------------------------------------------
% 作者：       赵敏琨  
% 日期：       2021年6月
% 说明：       数据集增广
% 软件版本：   MATLAB R2018a
%%-------------------------------------------------------------------------
%% 转三通道
clc, clear, close all

mkdir eigfaces_aug/train
mkdir eigfaces_aug/val
for ii = 1:360 
    imSrc_train = imread(['./eigfaces/train_set/' int2str(ii) '.bmp']);
    for ch = 1:3
        imDst_train(:,:,ch) = imSrc_train;
    end
    imwrite(imDst_train, ['./eigfaces_aug/train/' int2str(ii) '.jpg'])
end

for jj = 1:40 
    imSrc_val = imread(['./eigfaces/test_set/' int2str(jj) '.bmp']);
%     imSrc_val = imnoise(imSrc_val, 'salt & pepper', 0.03);   
    for ch = 1:3
        imDst_val(:,:,ch) = imSrc_val;
    end
    imwrite(imDst_val, ['./eigfaces_aug/val/' int2str(jj) '.jpg'])
%     imwrite(imDst_val, ['./eigfaces_aug/val/' int2str(jj) '_.jpg'])
end

%% 文件夹安排
for jj = 1:40 
    eval(['mkdir ./eigfaces_aug/train/Person' int2str(jj)])
    eval(['mkdir ./eigfaces_aug/val/Person' int2str(jj)])
end
for jj = 1:40
    for ii = 1:9
        eval(['movefile ./eigfaces_aug/train/' int2str(ii+(jj-1)*9) '.jpg ./eigfaces_aug/train/Person' int2str(jj)])
    end
    eval(['movefile ./eigfaces_aug/val/' int2str(jj) '.jpg ./eigfaces_aug/val/Person' int2str(jj)])    
%     eval(['movefile ./eigfaces_aug/val/' int2str(jj) '_.jpg ./eigfaces_aug/val/Person' int2str(jj)])    
end

%% 图像增广
for jj = 1:40
    for ii = 1:9
        path_img = ['./eigfaces_aug/train/Person' int2str(jj) '/' int2str(ii+(jj-1)*9) '.jpg'];
        path_folder = ['./eigfaces_aug/train/Person' int2str(jj) '/'];
        imSrc = imread(path_img);
        imDst_r45 = imrotate(imSrc, 45, 'bicubic', 'loose');
        %imshow(imDst_r45)
        imDst_sym = imSrc(:, end:-1:1, :);
        imDst_r45sym = imDst_r45(:, end:-1:1, :);
        %imshow(imDst_sym) 
        imwrite(imDst_r45, [path_folder int2str(ii+(jj-1)*9) '_r45.jpg'])
        imwrite(imDst_sym, [path_folder int2str(ii+(jj-1)*9) '_sym.jpg'])
        imwrite(imDst_r45sym, [path_folder int2str(ii+(jj-1)*9) '_r45sym.jpg'])
    end
end
