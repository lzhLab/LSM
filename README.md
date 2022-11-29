# LSM

liver segmentation augmentation

|    函数                                               |融合方式              | 代码|
|------------------------------------------------------|---------------------|-----------|
|copysmallobjects2_CaP                                 | CaP                 |aug_CaP.py|
|copysmallobjects2_Scale                               | Scale               | aug_CaP.py|
|copysmallobjects2_Rotation                            | Rotation            | aug_CaP.py|
|copysmallobjects2_Cir                                 | Cir                 | aug_Cir.py|
|copysmallobjects2_Rec                                 | Rec                 | aug_Rec.py|
|copysmallobjects2_Ply(dtype=Ply_nomix)                | Ply                 | aug_Ply.py|
|copysmallobjects2_Ply(dtype=Ply_Equal)                | Ply_Equal           | aug_Ply.py|
|copysmallobjects2_Ply(dtype=Ply_Gaussian)             | Ply_Gaussian        | aug_Ply_Gaussian.py|
|copysmallobjects2_Ply(dtype=Ply_Rev_Gaussian)         | Ply_Rev_Gaussian    | aug_Ply_Gaussian.py|

copysmallobjects2_Scale(),病灶直接放粘贴时病灶边界有过渡色，破坏病灶上下文信息，通过super参数裁减比病灶本身大1.5倍区域，对该区域放大后再粘贴原始病灶放大区域

copysmallobjects2_Rotation()，逐像素旋转后粘贴到新位置，旋转后可能出现的空洞，fill_img()函数对空洞填补
