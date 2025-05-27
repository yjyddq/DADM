from models.my_adapter import my_ViT_3modality_CDC_Adapter, MMDG, DADM


def get_model(model_name, args=None):
    model_dict = {
        # "vit_3modal": get_vit_3modal,

        "my_vit_2modality_cdc_adapter": get_my_vit_3modality_cdc_adapter,
        "mmdg": get_MMDG,
        "dadm": get_DADM

    }
    return model_dict[model_name](args)

def get_my_vit_3modality_cdc_adapter():
    return my_ViT_3modality_CDC_Adapter()

def get_MMDG():
    return MMDG()

def get_DADM(args):
    return DADM(missing_modality=args.train_missing_modality)