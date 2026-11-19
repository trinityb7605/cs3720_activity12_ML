from torchvision.models import ( mobilenet_v3_small, MobileNet_V3_Small_Weights );

mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
print("Model weights downloaded.")
