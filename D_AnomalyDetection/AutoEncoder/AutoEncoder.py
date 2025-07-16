from D_AnomalyDetection.Util.Util import *
from D_AnomalyDetection.Util.Draw_Graph import *

class AutoEncoder(nn.Module):
    def __init__(self, alpha=1.0, num_class=10):
        super().__init__()
        self.resize_size = (256, 256)
        self.transform_info = transforms.Compose([
            transforms.Resize(self.resize_size),
            transforms.ToTensor()
        ])

        self.network = []


    def create_network(self, resize_size):
        row = resize_size[0], col = resize_size[1]
        channel = 4
        while row > 20 and col > 20 :
            self.network.append(nn.Conv2d(3, channel, kernel_size=4, stride=2, padding=1))

        #while row == resize_size[0] and col == resize_size[1] :


    def forward(self, x):
        x = self.First_Step(x)
        x = self.Second_Step(x)
        x = self.Third_Step(x)
        return x

    def return_transform_info(self):
        return self.transform_info



def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, '..', 'Util', 'config.yaml')
    yaml_path = os.path.normpath(yaml_path)

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoEncoder(num_class=config['num_class']).to(device)

    graph = Draw_Graph(model=model, save_path=config['save_path'], patience=config['patience'])

    transform_info = model.return_transform_info()
    train_loader, validation_loader, test_loader = anomalydetection_data_loader(config['load_path'],
                                                                              config['batch_size'], transform_info)

    train_model(device=device, model=model, epochs=config['epoch'], patience=config['patience'], train_loader=train_loader,
                val_loader=validation_loader, test_loader=test_loader, lr=0.001, graph=graph)

if __name__ == "__main__":
    main()
