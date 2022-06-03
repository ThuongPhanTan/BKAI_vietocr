from vietocr.tool.config import Cfg
from vietocr.model.train_new import Trainer



def main():
    config = Cfg.load_config_from_name('vgg_transformer')


    dataset_params = {
        'name': 'DATA',
        'data_root':'./data',
        'train_annotation':'train.txt',
        'valid_annotation':'test.txt'
    }

    params = {
        'print_every':200,
        'valid_every':2000,
        'iters':100000,
        'batch_size': 128,
        #'checkpoint':'./weights/vietocr.pth',    
        'export':'./weights/vietocr.pth',
        'metrics': 10000
    }

    config['trainer'].update(params)
    config['dataset'].update(dataset_params)
    config['device'] = 'cuda:2'
    config['vocab'] = 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~° '+ '̉'+ '̀' + '̃'+ '́'+ '̣'
    trainer = Trainer(config, pretrained=False)

    #trainer.config.save('my_config/vgg_transformer.yml')
    #trainer.load_checkpoint('./weights/vietocr.pth')
    #print('Weight loading')
    #trainer.load_weights('./weights/vietocr.pth')
    print('Start training')
    trainer.train()
    print(trainer.precision())
 
if __name__ == '__main__':
    main()

