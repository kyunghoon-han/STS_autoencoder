from STS import trainer
import argparse

parser = argparse.ArgumentParser(description='An STS autoencoder model')
parser.add_argument('--batch_size',type=int,help='number of batches')
parser.add_argument('--epochs',type=int,help='maximum number of training epochs')
parser.add_argument('--num_test',type=int,help='number of test data')
parser.add_argument('--lr_stt',type=float,help='learning rate of STT parts')
parser.add_argument('--lr_tts',type=float,help='learning rate of TTS parts')
parser.add_argument('--step_lrtts',type=int,help='scheduler step size for the TTS parts')
parser.add_argument('--step_lrstt',type=int,help='scheduler step size for the STT parts')
parser.add_argument('--STS_threshold',type=int,help='the epoch to start combining TTS and STT parts to STS and TTT models' )
parser.add_argument('--verbose',type=int,help='model verbosity')
parser.add_argument('--data_dir',help='directory where the preprocessed data are stpred')
parser.add_argument('--encoded_txt_pkl',help='a pickle file with text-to-one-hot dictionary')
parser.add_argument('--log_dir',help='directory to store the log file')
parser.add_argument('--log_filename',help='name of the log file')
parser.add_argument('--model_save_dir',help='directory to save the models')
parser.add_argument('--device',help='device to run the model on')
parser.add_argument('--device_auto',type=bool,help='an option to let the program to choose the model automatically')


args = parser.parse_args()

trainer(batch_size=args.batch_size, 
        epochs=args.epochs,
        num_test=args.num_test,
        lr_stt=args.lr_stt, 
        lr_tts=args.lr_tts, 
        step_lrtts=args.step_lrtts,
        step_lrstt=args.step_lrstt, 
        STS_threshold=args.STS_threshold, 
        verbose=args.verbose,
        data_dir=args.data_dir,
        encoded_txt_pkl=args.encoded_txt_pkl,
        log_dir=args.log_dir, 
        log_filename=args.log_filename,
        model_save_dir=args.model_save_dir,
        device=args.device,
        device_auto=args.device_auto)
