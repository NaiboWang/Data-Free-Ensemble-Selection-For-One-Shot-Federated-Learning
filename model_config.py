# from model.efficientnet_pytorch import EfficientNet
from model.effnetv2 import effnetv2_l
from model.SpinalNet import SpinalVGG
from model.dla import DLA
from model.densenet import DenseNet, Bottleneck
from model.resnet import ResNet50, ResNet18
def get_model(args):
    # print(args)
    if args.model == "SpinalNet":
        model = SpinalVGG(args.num_classes, args.input_channels).to(args.device)
    elif args.model == "effenetv2_l":
        model = effnetv2_l(args.num_classes).to(args.device)
    # elif args.model == "efficientnet-b7":
    #     model = EfficientNet.from_pretrained('efficientnet-b7', in_channels=args.input_channels,
    #                                           num_classes=args.num_classes).to(args.device)
    elif args.model == "dla":
        model = DLA(num_classes=args.num_classes).to(args.device)
    elif args.model == "resnet50":
        model = ResNet50(num_classes=args.num_classes, input_channels=args.input_channels).to(args.device)
    elif args.model == "resnet18":
        model = ResNet18(num_classes=args.num_classes, input_channels=args.input_channels).to(args.device)
    elif args.model == "densenet":
        model = DenseNet(Bottleneck, [6,12,24,16], growth_rate=12,num_classes=args.num_classes).to(args.device)
    return model

if __name__ == '__main__':
    from config import exp_config
    from commandline_config import Config
    config = Config(exp_config)
    config.model = "dla"
    model = get_model(config)
    print(config)
    for key in model.state_dict().keys():
        print(key)
    print(model.state_dict().keys())
