# LSM

liver segmentation augmentation

|    Function                                          |Fusion method        | Code      |
|------------------------------------------------------|---------------------|-----------|
|copysmallobjects2_CaP                                 | CaP                 |aug_CaP.py |
|copysmallobjects2_Scale                               | Scale               | aug_CaP.py|
|copysmallobjects2_Rotation                            | Rotation            | aug_CaP.py|
|copysmallobjects2_Cir                                 | Cir                 | aug_Cir.py|
|copysmallobjects2_Rec                                 | Rec                 | aug_Rec.py|
|copysmallobjects2_Ply(dtype=Ply_nomix)                | Ply                 | aug_Ply.py|
|copysmallobjects2_Ply(dtype=Ply_Equal)                | Ply_Equal           | aug_Ply.py|
|copysmallobjects2_Ply(dtype=Ply_Gaussian)             | Ply_Gaussian        | aug_Ply_Gaussian.py|
|copysmallobjects2_Ply(dtype=Ply_Rev_Gaussian)         | Ply_Rev_Gaussian    | aug_Ply_Gaussian.py|

copysmallobjects2_CaP: copy-and-paste lesion to another place without context;
copysmallobjects2_Scale: scale-up/down lesion size to increase diversity;
copysmallobjects2_Rotation: rotate lesion with an arbitrary angle to increase diversity;
copysmallobjects2_Cir: circular context-aware augmentation;
copysmallobjects2_Rec: rectangular context-aware augmentation;
copysmallobjects2_Ply: polygonal context-aware augmentation.
