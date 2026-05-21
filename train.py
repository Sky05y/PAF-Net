import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description='Data Postprocess')
    parser.add_argument('--model', type=str, default=None, help='load the model')
    parser.add_argument('--data_dir', type=str, default=None, help='the dir to data')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--batch', type=int, default=16, help='batch size for training')
    parser.add_argument('--dataset_config', type=str, required=True, help='path to dataset config file')
    parser.add_argument('--output_dir', type=str, default='runs/train', help='directory to save training results')
    parser.add_argument('--run_name', type=str, default='exp', help='name of the training run')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(f"Model: {args.model}")
    print(f"Data Directory: {args.data_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch}")
    print(f"Dataset Config: {args.dataset_config}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Run Name: {args.run_name}")
    
    model = YOLO(args.model)
    model.train(data=args.dataset_config, epochs=args.epochs, batch=args.batch, project=args.output_dir, name=args.run_name)

if __name__ == '__main__':
    main()
