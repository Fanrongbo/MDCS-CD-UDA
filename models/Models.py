

from models.MDCS_CD_UDA import backbone,backbone2BN,backbone2BN_con,backbone2BN_con_ASSP,\
    backbone2BN_conSA,backbone2BN_con_PSP,backbone2BN_con_AT,backbone2BN_conATSA,backbone2BN_conATSA2,\
    backbone2BN_conATSA3,backbone2BN_conATSA4,backbone2BN_con2,backbone2BN_con3
from models.CD_compare import FCSiamDiff,FCSiamConc,UCDNet,DeepLabori
network_dict = {"backbone2BN_conATSA4":backbone2BN_conATSA4,
                "FCSiamDiff":FCSiamDiff,
                "FCSiamConc":FCSiamConc,
                "UCDNet":UCDNet,
                "backbone2BN":backbone2BN,
                "backbone2BN_con":backbone2BN_con,
                "DeepLabori":DeepLabori}




