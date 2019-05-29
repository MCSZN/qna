from argparse import ArgumentParser
parser =  ArgumentParser()

# Get arguments to launch use commands
# User can call train/predict and specify wether to use GPU or CPU
# User should specify where to fetch data
parser.add_argument('-E','--env', type=str, default='local',
    help='script is run on colab or locally')
parser.add_argument('-P','--path', type=str, default='',
    help='default path to json data')
parser.add_argument('-M','--model_path', type=str, default='',
    help='path to model that should be loaded')
parser.add_argument('-S','--sentence', type=str, default='',
    help='sentence to predict with the LSTM IA')
parser.add_argument('-I','--model_id', type=str, default='',
    help='id if there are multiple models')
parser.add_argument('-ME','--meta_data', nargs='+', type=str,
    default="0,0,0,0,0,0,0,0",
    help='input the metadata to feed the model for prediction. eg. "0,2,4,4,1,5,6,8"')

parser.add_argument('-U','--use_gpu', action='store_true',
    help='call if want GPU otherwise dont')
parser.add_argument('-T','--train', action='store_true',
    help='call if want training otherwise dont')
parser.add_argument('-L','--load', action='store_true',
    help='call if want to load a model otherwise dont')
parser.add_argument('-PR','--predict', action='store_true',
    help='call if want to predict the sentence given otherwise dont')

args = parser.parse_args()



if __name__=='__main__':
    from QnA.QnA import execution
    meta_data = [int(item) for item in args.meta_data.split(',')]
    execution(
        env=args.env,
        path=args.path,
        use_gpu=args.use_gpu,
        train=args.train,
        load=args.load,
        predict=args.predict,
        model_path=args.model_path,
        sentence=args.sentence,
        model_id=args.model_id,
        meta_data=meta_data)