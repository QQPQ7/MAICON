"""Train
"""

from datetime import datetime  #날짜와 시간 다루는 모듈
from time import time
import numpy as np
import shutil, random, os, sys, torch
from glob import glob #데이터파일들의 리스트를 뽑을 때 사용하는데, 파일의 경로명을 관련 사용
from torch.utils.data import DataLoader #데이터타입으로 불러오는용
from sklearn.model_selection import train_test_split #데이터를 분할하는용임

prj_dir = os.path.dirname(os.path.abspath(__file__)) #os.path.abspath(__file__)는 현재프로젝트파일의 절대경로를 구하는것
sys.path.append(prj_dir) #시스템 경로에 프로젝트경로추가

#모듈폴더내에서 파일에서 함수 불러오기
from modules.utils import load_yaml, get_logger #utils 파일에 있는 함수 불러오기 https://www.inflearn.com/questions/16184이런 파일들 json, yaml 불러오는용, get_logger는 출력로그용임
from modules.metrics import get_metric_function #가중 스코어에 대한 가중치 추가하여 측정값 IoU로 계산
from modules.earlystoppers import EarlyStopper #예측 스코어가 더이상 안올라가면 에폭 다 안돌아도 미리 중단
from modules.losses import get_loss_function #뒤에 들어오는 거에 따라서 CCE나 GeneralizediceLoss사용 확인해보니 야믈파일에 MeanCCELoss로 설정해둠
from modules.optimizers import get_optimizer #일단 옵티마이저는 아담, SGD, 아담w 라는게 들어있으면 최신 옵티마이저 우리가 직접 추가해서 써도 될듯 혹은 변경해가면서 성능테스트
from modules.schedulers import get_scheduler #러닝스케쥴러관련함수 CosineAnnealingLR가 현재 기본설정값 config파일에서 수정
from modules.scalers import get_image_scaler #기본적인 이미지 전처리기 이미지 정규화랑 밝기 정규화밖에 없음 이것도 전처리따로 만들어서 학습량 강화하면댈듯
from modules.datasets import SegDataset #데이터셋 불러오는용
from modules.recorders import Recorder #레코더-> 대충 내용 저장하는용인듯 #가중치 저장는 187줄
from modules.trainer import Trainer #학습돌릴때 필요한 애들 들어있는듯
from models.utils import get_model #모델 들고오기 UNET, FPN, PAN, Psnet인가 다양하게 있음 FPN 괜춚을거같은디? 일단 기본은 Unet으로 설정되어 있음

if __name__ == '__main__':

    # Load config
    config_path = os.path.join(prj_dir, 'config', 'train.yaml') #config(구성)파일 불러오기 config 볼더 내에 train.yaml을 불러올거임
    config = load_yaml(config_path) #불러온 config파일 설정 이 밑으로 config달린거는 거이에 있는 설정 불러온다고 생각하면댐
    
    # Set train serial: ex) 20211004 //학습 시리얼폴더이름 설정-> 근데일단 자동으로 현재시간으로 해주는듯
    train_serial = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_serial = 'debug' if config['debug'] else train_serial

    # Set random seed, deterministic
    torch.cuda.manual_seed(config['seed'])
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set device(GPU/CPU)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu_num'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device) #그냥 한줄 추가해봤음 지금 돌아가는게 gpu인지 cpu인지 확인용 cuda면 gpu로 돌아가는거임

    # Create train result directory and set logger 결과저장경로랑 로그생성기
    train_result_dir = os.path.join(prj_dir, 'results', 'train', train_serial)
    os.makedirs(train_result_dir, exist_ok=True)

    # Set logger 로거설정
    logging_level = 'debug' if config['verbose'] else 'info'
    logger = get_logger(name='train',
                        file_path=os.path.join(train_result_dir, 'train.log'),
                        level=logging_level)


    # Set data directory 학습할 데이터경로설정
    train_dirs = os.path.join(prj_dir, 'data', 'train')

    # Load data and create dataset for train 
    # Load image scaler 이미지 불러오는거 데이터셋형태로 만들어서 가져옴
    train_img_paths = glob(os.path.join(train_dirs, 'x', '*.png'))
    train_img_paths, val_img_paths = train_test_split(train_img_paths, test_size=config['val_size'], random_state=config['seed'], shuffle=True) #학습할이미지(train)와 학습중 검증용(val)데이터경로 분리

    train_dataset = SegDataset(paths=train_img_paths,
                            input_size=[config['input_width'], config['input_height']], #원본이미지는 1508x754인데 인풋은 480x256 cv로 리사이즈해줘서그럼
                            scaler=get_image_scaler(config['scaler']),
                            logger=logger)
    val_dataset = SegDataset(paths=val_img_paths,
                            input_size=[config['input_width'], config['input_height']],
                            scaler=get_image_scaler(config['scaler']),
                            logger=logger)
    #분리된데이터를 데이터셋형태로 변경 segDataset함수는 datasets라는 파일에서 가져온거임
    
    # Create data loader
    train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=config['batch_size'], #배치사이즈
                                num_workers=config['num_workers'], #대기하는역할
                                shuffle=config['shuffle'],
                                drop_last=config['drop_last'])
                                
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=config['batch_size'],
                                num_workers=config['num_workers'], 
                                shuffle=False,
                                drop_last=config['drop_last'])

    logger.info(f"Load dataset, train: {len(train_dataset)}, val: {len(val_dataset)}")
    
    # Load model 모델불러오기
    model = get_model(model_str=config['architecture']) #config파일에서 설정한 아키텍쳐(인공지능) 불러오기
    model = model(classes=config['n_classes'], #구분할 클래스수(4개임)
                encoder_name=config['encoder'], #얘네 인공지능 형태가 한쪽에서는 축소시키고 다시 한쪽에서는 확대시키는 구조라서 인코더는 축소시키는 역할임
                encoder_weights=config['encoder_weight'],
                #psp_dropout = 0.2, #PSPnet용으로 추가한거 나중에 제거필요 ##################!!!!!!!!!!!!!!!!!!!!!!!-----------------------------------------
                activation=config['activation']).to(device) #활성화함수 : 각 요소별 학습연산도중 가중치값이 특정값 이하면 그 요소는 학습 안함
    logger.info(f"Load model architecture: {config['architecture']}")

    # Set optimizer 계산 옵션 설정
    optimizer = get_optimizer(optimizer_str=config['optimizer']['name'])
    optimizer = optimizer(model.parameters(), **config['optimizer']['args'])
    
    # Set Scheduler
    scheduler = get_scheduler(scheduler_str=config['scheduler']['name'])   #CosineAnnealingLR으로 설정되어있음
    scheduler = scheduler(optimizer=optimizer, **config['scheduler']['args'])  #스케줄러 설정 학습량, 보폭량을 변경이라고 생각
    """
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: 0.95 ** epoch,
                                        last_epoch=-1,
                                        verbose=False)
    원래는 이런 형태로 되어있는건데 함수로 설정하고 config파일에서 수정하게 만든거
    """
    

    # Set loss function
    loss_func = get_loss_function(loss_function_str=config['loss']['name'])  # MeanCCELoss로 일단 설정되어 있음
    loss_func = loss_func(**config['loss']['args'])

    # Set metric 측정기 설정
    metric_funcs = {metric_name:get_metric_function(metric_name) for metric_name in config['metrics']}
    logger.info(f"Load optimizer:{config['optimizer']['name']}, scheduler: {config['scheduler']['name']}, loss: {config['loss']['name']}, metric: {config['metrics']}")

    # Set trainer 트레이너 설정
    trainer = Trainer(model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss_func=loss_func,
                    metric_funcs=metric_funcs,
                    device=device,
                    logger=logger)
    logger.info(f"Load trainer")

    # Set early stopper 학습수준이 나아지지 않으면 여기서 중지
    early_stopper = EarlyStopper(patience=config['earlystopping_patience'],
                                logger=logger)
    # Set recorder 
    recorder = Recorder(record_dir=train_result_dir,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        logger=logger)
    logger.info("Load early stopper, recorder")

    # Recorder - save train config 저장하는놈
    shutil.copy(config_path, os.path.join(recorder.record_dir, 'train.yaml'))

    # Train 학습과정
    print("START TRAINING")
    logger.info("START TRAINING")
    for epoch_id in range(config['n_epochs']):
        
        # Initiate result row
        row = dict()
        row['epoch_id'] = epoch_id #에폭수 - 반복학습할수 보통 클수록 좋아지긴하는데 그만큼 시간이 더걸림
        row['train_serial'] = train_serial #이건 그냥 이름설정인데 현재날짜시간으로 되어있음
        row['lr'] = trainer.scheduler.get_last_lr() #학습보폭(걸음의 크기라고 생각하면댐) 스케쥴러에 있는 마지막lr을 가저옴

        # Train
        print(f"Epoch {epoch_id}/{config['n_epochs']} Train..") #학습출력내용 Epoch 13/100 Validation이런거 출력하는 애임
        logger.info(f"Epoch {epoch_id}/{config['n_epochs']} Train..")
        tic = time()
        trainer.train(dataloader=train_dataloader, epoch_index=epoch_id) #trainer파일안에 train함수를 통한 학습 자세한 코드는 그쪽봐야할듯
        toc = time()
        # Write tarin result to result row
        row['train_loss'] = trainer.loss  # Loss 오차값 : 실제답(y-변화지역)과의 인공지능이 예측한 변화지역 비교
        for metric_name, metric_score in trainer.scores.items():
            row[f'train_{metric_name}'] = metric_score

        row['train_elapsed_time'] = round(toc-tic, 1)
        # Clear
        trainer.clear_history()

        # Validation 검증용데이터결과: 훈련에는 사용되지 않는 데이터로 학습결과가 잘나오고 있는지 확인하는용
        print(f"Epoch {epoch_id}/{config['n_epochs']} Validation..")
        logger.info(f"Epoch {epoch_id}/{config['n_epochs']} Validation..")
        tic = time()
        trainer.validate(dataloader=val_dataloader, epoch_index=epoch_id)
        toc = time()
        row['val_loss'] = trainer.loss
        # row[f"val_{config['metric']}"] = trainer.score
        for metric_name, metric_score in trainer.scores.items():
            row[f'val_{metric_name}'] = metric_score
        row['val_elapsed_time'] = round(toc-tic, 1)
        trainer.clear_history()

        # Performance record - row
        recorder.add_row(row)
        
        # Performance record - plot
        recorder.save_plot(config['plot'])

        # Check early stopping
        early_stopper.check_early_stopping(row[config['earlystopping_target']])
        if early_stopper.patience_counter == 0:
            recorder.save_weight(epoch=epoch_id) #가중치 저장 result폴더안에train안에 날짜시간으로된 폴더의 model.pt
            
        if early_stopper.stop:
            print(f"Epoch {epoch_id}/{config['n_epochs']}, Stopped counter {early_stopper.patience_counter}/{config['earlystopping_patience']}")
            logger.info(f"Epoch {epoch_id}/{config['n_epochs']}, Stopped counter {early_stopper.patience_counter}/{config['earlystopping_patience']}")
            break

    print("END TRAINING")
    logger.info("END TRAINING")