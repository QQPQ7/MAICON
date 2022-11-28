"""
Predict
"""
from datetime import datetime
from tqdm import tqdm
import numpy as np
import random, os, sys, torch, cv2, warnings
from glob import glob
from torch.utils.data import DataLoader

prj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(prj_dir)

from modules.utils import load_yaml, save_yaml, get_logger
from modules.scalers import get_image_scaler
from modules.datasets import SegDataset
from models.utils import get_model
warnings.filterwarnings('ignore')

if __name__ == '__main__':

    #개인적으로 추가해본 GPU 옵션
#    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'  # 본인이 사용하고 싶은 GPU 넘버를 써주면 됨
#    os.environ['MASTER_ADDR'] = 'localhost'
#    os.environ['MASTER_PORT'] = '29500'

    #! Load config
    config = load_yaml(os.path.join(prj_dir, 'config', 'predict.yaml'))  #프레딕트 야믈파일에 있는 옵션 변경해줘야함 학습시킨 결과가 담긴 폴더명으로 수정 ex)train_serial: '20221025_145511'(구버전) ->train_serial: '20221108_213153'(새로학습한거)
    train_config = load_yaml(os.path.join(prj_dir, 'results', 'train', config['train_serial'], 'train.yaml')) 
    
    #! Set predict serial 예측한후 폴더명 설정 - 완료시각으로 하는듯
    pred_serial = config['train_serial'] + '_' + datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set random seed, deterministic
    torch.cuda.manual_seed(train_config['seed'])
    torch.manual_seed(train_config['seed'])
    np.random.seed(train_config['seed'])
    random.seed(train_config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set device(GPU/CPU) 
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu_num'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create train result directory and set logger 결과생성하고 로그생성
    pred_result_dir = os.path.join(prj_dir, 'results', 'pred', pred_serial)
    pred_result_dir_mask = os.path.join(prj_dir, 'results', 'pred', pred_serial, 'mask') #결과저장경로 result폴더에 pred안에 마스크
    os.makedirs(pred_result_dir, exist_ok=True)
    os.makedirs(pred_result_dir_mask, exist_ok=True)

    # Set logger
    logging_level = 'debug' if config['verbose'] else 'info'
    logger = get_logger(name='train', #학습때 썼던거 들고옴
                        file_path=os.path.join(pred_result_dir, 'pred.log'),
                        level=logging_level)

    # Set data directory 데이터 불러오기
    test_dirs = os.path.join(prj_dir, 'data', 'test') #경로는 data폴더안에 있는 test폴더
    test_img_paths = glob(os.path.join(test_dirs, 'x', '*.png')) #거기에서 x란 폴더안에 .png로 끝나는거 다들고오기

    #! Load data & create dataset for train  데이터셋 설정, 데이터 로더 설정
    test_dataset = SegDataset(paths=test_img_paths,
                            input_size=[train_config['input_width'], train_config['input_height']],
                            scaler=get_image_scaler(train_config['scaler']),
                            mode='test',
                            logger=logger)

    # Create data loader
    test_dataloader = DataLoader(dataset=test_dataset,
                                batch_size=config['batch_size'],
                                num_workers=config['num_workers'],
                                shuffle=False,
                                drop_last=False)
    logger.info(f"Load test dataset: {len(test_dataset)}")

    # Load architecture 인공지능아키텍쳐 불러오기
    model = get_model(model_str=train_config['architecture']) #train config 파일에 있는 '아키텍쳐' 항목의 세팅을 읽어와서 get_model 이라는 함수를 사용하여 모델로 설정
    model = model(
                classes=train_config['n_classes'],
                encoder_name=train_config['encoder'],
                encoder_weights=train_config['encoder_weight'],
                activation=train_config['activation']).to(device)
    logger.info(f"Load model architecture: {train_config['architecture']}")

    #! Load weight 가중치 불러오기
    check_point_path = os.path.join(prj_dir, 'results', 'train', config['train_serial'], 'model.pt') #model.pt 라는 파일이 체크포인트파일임
    check_point = torch.load(check_point_path)
    model.load_state_dict(check_point['model']) #가중치(체크포인트)들고와서 적용
    logger.info(f"Load model weight, {check_point_path}")

    # Save config 학습한 설정표 저장
    save_yaml(os.path.join(pred_result_dir, 'train_config.yml'), train_config) 
    save_yaml(os.path.join(pred_result_dir, 'predict_config.yml'), config)
    
    # Predict 밑으로는 예측과정진행: 사실상 train이랑 똑같은건데 학습이 아니라 출력만 보는것
    logger.info(f"START PREDICTION")

    model.eval()

    with torch.no_grad():

        for batch_id, (x, orig_size, filename) in enumerate(tqdm(test_dataloader)): #test_dataloader 항목에서 batch_id, (x, orig_size, filename)을 추출하고 아래에서 사용할거임
            
            x = x.to(device, dtype=torch.float) #x는 이미지일거임
            y_pred = model(x)
            y_pred_argmax = y_pred.argmax(1).cpu().numpy().astype(np.uint8) #argmax(1)는 얘가 누구에 가까온 값인지 예측하는거임 파괴인지 신축인지 분류하는거
            orig_size = [(orig_size[0].tolist()[i], orig_size[1].tolist()[i]) for i in range(len(orig_size[0]))]
            # Save predict result 예측결과 저장
            for filename_, orig_size_, y_pred_ in zip(filename, orig_size, y_pred_argmax):
                resized_img = cv2.resize(y_pred_, [orig_size_[1], orig_size_[0]], interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(os.path.join(pred_result_dir_mask, filename_), resized_img)
    logger.info(f"END PREDICTION")