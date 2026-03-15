import timm


def create_model(model_name: str, num_classes: int, pretrained: bool = True):
    return timm.create_model(
        model_name=model_name, pretrained=pretrained, num_classes=num_classes
    )
