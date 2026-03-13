import torch
import torchvision
from dataset.run_datasets import VideoDataset
import h5py


def run_model(model_fun, dataset, batch_size, n_workers, callbacks):
    """
    Run the model on a given dataset

    @param model_fun: function that returns the model
    @param dataset: torch Dataset object
    @param batch_size: batch size for parallel computing
    @param n_workers: n° of workers for parallel process
    @param callbacks: list of callback function to execute after each item is processed
    @return:
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Running using device: ' + str(device))

    model = model_fun()
    # Setup the data loader
    if type(dataset) == VideoDataset:
        n_workers = 0
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers
    )

    # Make sure the model is set to evaluation mode
    model.eval()

    with torch.no_grad():
        for input, other in data_loader:
            input = input.to(device)
            predictions = model.predict(input)

            input = input.to('cpu')
            predictions = predictions.to('cpu')
            for i in range(input.shape[0]):
                for callback in callbacks:
                    callback(input[i], predictions[i], other[i])

def run_transforms(mean, std, size):
    """
    Run the necessary transformation for running the image in the net

    @param mean: mean for normalizing
    @param std: standard devation for normalizing
    @param size: resize input for the net
    """
    return torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(mean=mean,
                                                                            std=std),
                                           torchvision.transforms.Resize(size)
                                           ])  # normalize to (-1, 1)

#Louis new

import argparse
from models.CC import CrowdCounter
from config import cfg
from dataset.run_datasets import ImageDataset # 假設資料夾內是圖片
from callbacks import call_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Crowd Flow Detection')
    parser.add_argument('--path', type=str, required=True, help='Path to images/video')
    parser.add_argument('--callbacks', nargs='+', required=True, help='List of callbacks')
    args = parser.parse_args()

    # 1. 定義如何載入模型 (就像在 test_gpu.py 做的一樣)
    def load_my_model():
        cc = CrowdCounter()
        # 記得確認這裡的路徑是否正確
        cc.load('../exp/09-02_17-32_VisDrone_MobileCount_0.001__1080x1920_CROWD_COUNTING_BS4/all_ep_58_mae_23.9_rmse_30.0.pth')
        return cc

    # 2. 準備圖片資料集 (假設處理 images256 資料夾)
    # 這裡的參數請參考 config.py 裡面的平均值與標準差
    transform = run_transforms(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], size=(1080, 1920))
    dataset = ImageDataset(args.path, transform=transform)

    # 3. 轉換 callbacks 字串為實際的功能函數
    selected_callbacks = [call_dict[name] for name in args.callbacks]

    # 4. 正式啟動機器！
    run_model(model_fun=load_my_model, 
              dataset=dataset, 
              batch_size=1, 
              n_workers=0, 
              callbacks=selected_callbacks)
