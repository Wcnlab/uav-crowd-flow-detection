import torch
import torchvision
import argparse
import os
from PIL import Image
from models.CC import CrowdCounter
from config import cfg
from callbacks import call_dict

# --- 1. 定義圖片讀取器 (直接寫在這裡，不用 import) ---
class SimpleImageLoader(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # 只讀取常見的圖片格式
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if len(self.image_files) == 0:
            print(f"警告：在 {root_dir} 資料夾中找不到任何圖片！")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_name

# --- 2. 核心執行邏輯 ---
def run_model(model_fun, dataset, callbacks):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('正在使用設備: ' + str(device))

    model = model_fun()
    model.to(device)
    model.eval()

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for input, img_path in data_loader:
            input = input.to(device)
            # 呼叫模型的預測功能
            predictions = model.predict(input)

            input = input.to('cpu')
            predictions = predictions.to('cpu')
            
            # 執行您指定的 callback 功能 (例如顯示圖片或算人數)
            for callback in callbacks:
                # input[0] 是因為 batch_size 是 1
                callback(input[0], predictions[0], img_path[0])

def run_transforms(mean, std, size):
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std),
        torchvision.transforms.Resize(size)
    ])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--callbacks', nargs='+', required=True)
    args = parser.parse_args()

    print(f"--- 開始處理資料夾: {args.path} ---")

    # 💡 確保這裡的 def 與上面的 print 是垂直對齊的
    def my_model_loader():
        cc = CrowdCounter(gpus='0', model_name=cfg.NET) 
        cc.load(cfg.PRE_TRAINED)
        return cc

    # 💡 這裡的 transform 也必須與上面的 def 對齊
    transform = run_transforms(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], size=(1080, 1920))
    
    # 建立資料集
    if os.path.isdir(args.path):
        my_dataset = SimpleImageLoader(args.path, transform=transform)
        
        # 將指令中的 callback 名稱轉成實際的功能
        selected_callbacks = [call_dict[name] for name in args.callbacks]

        # 啟動！
        run_model(model_fun=my_model_loader, dataset=my_dataset, callbacks=selected_callbacks)
        print("--- 處理完成 ---")
    else:
        print(f"錯誤：找不到資料夾 '{args.path}'，請確認路徑是否正確。")