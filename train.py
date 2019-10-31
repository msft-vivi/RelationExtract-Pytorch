"""Train and evaluate the model"""
import argparse
import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import trange

import tools.utils as utils
import model.net as net
from tools.data_loader import DataLoader
from evaluate import evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/SemEval2010_task8', help="Directory containing the dataset")
parser.add_argument('--embedding_file', default='data/embeddings/vector_50d.txt', help="Path to embeddings file.")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--gpu', default=-1, help="GPU device number, 0 by default, -1 means CPU.")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before training")


def train(model, data_iterator, optimizer, scheduler, params, steps_num):
    """Train the model on `steps_num` batches"""
    # set model to training mode
    model.train()
    # scheduler.step()
    # a running average object for loss
    loss_avg = utils.RunningAverage()
    
    # Use tqdm for progress bar
    t = trange(steps_num)
    for _ in t:
        # fetch the next training batch
        batch_data, batch_labels = next(data_iterator)

        # compute model output and loss
        batch_output = model(batch_data)
        loss = model.loss(batch_output, batch_labels)

        # clear previous gradients, compute gradients of all variables wrt loss
        model.zero_grad()
        # optimizer.zero_grad()
        loss.backward()

        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), params.clip_grad)

        # performs updates using calculated gradients
        optimizer.step()

        # update the average loss
        loss_avg.update(loss.item())
        t.set_postfix(loss='{:05.3f}'.format(loss_avg()))

    return loss_avg()
    

def train_and_evaluate(model, train_data, val_data, optimizer, scheduler, params, metric_labels, model_dir, restore_file=None):
    """Train the model and evaluate every epoch."""
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)
        
    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, params.epoch_num + 1):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch, params.epoch_num))

        # Compute number of batches in one epoch
        train_steps_num = params.train_size // params.batch_size
        val_steps_num = params.val_size // params.batch_size

        # data iterator for training
        train_data_iterator = data_loader.data_iterator(train_data, params.batch_size, shuffle='True')
        # Train for one epoch on training set
        train_loss = train(model, train_data_iterator, optimizer, scheduler, params, train_steps_num)

        # data iterator for training and validation
        train_data_iterator = data_loader.data_iterator(train_data, params.batch_size)
        val_data_iterator = data_loader.data_iterator(val_data, params.batch_size)

        # Evaluate for one epoch on training set and validation set
        train_metrics = evaluate(model, train_data_iterator, train_steps_num, metric_labels)
        train_metrics['loss'] = train_loss
        train_metrics_str = "; ".join("{}: {:05.2f}".format(k, v) for k, v in train_metrics.items())
        logging.info("- Train metrics: " + train_metrics_str)
        
        val_metrics = evaluate(model, val_data_iterator, val_steps_num, metric_labels)
        val_metrics_str = "; ".join("{}: {:05.2f}".format(k, v) for k, v in val_metrics.items())
        logging.info("- Eval metrics: " + val_metrics_str)
        
        val_f1 = val_metrics['f1']
        improve_f1 = val_f1 - best_val_f1

        # Save weights ot the network
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()}, 
                               is_best=improve_f1>0,
                               checkpoint=model_dir)
        if improve_f1 > 0:
            logging.info("- Found new best F1")
            best_val_f1 = val_f1
            if improve_f1 < params.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping and logging best f1
        if (patience_counter >= params.patience_num and epoch > params.min_epoch_num) or epoch == params.epoch_num:
            logging.info("best val f1: {:05.2f}".format(best_val_f1))
            break
        

def CNN(data_loader,params):
    # Define the model and optimizer
    model = net.CNN(data_loader, params)
    if params.optim_method == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=params.weight_decay)
    elif params.optim_method == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=params.learning_rate, betas=(0.9, 0.999), weight_decay=params.weight_decay)
    elif params.optim_method == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, weight_decay=params.weight_decay)
    else:
        raise ValueError("Unknown optimizer, must be one of 'sgd'/'adam'/'adadelta'.")

    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(1 + 0.05*epoch)) # 动态改变学习率

    # Train and evaluate the model
    logging.info("Starting training for {} epoch(s)".format(params.epoch_num))
    train_and_evaluate(model=model,
                       train_data=train_data,
                       val_data=val_data,
                       optimizer=optimizer,
                       scheduler=scheduler,
                       params=params,
                       metric_labels=metric_labels,
                       model_dir=args.model_dir,
                       restore_file=args.restore_file)

def BiLSTM_Att(data_loader,params):
    # Define the model and optimizer
    model = net.BiLSTM_Att(data_loader, params)
    if params.optim_method == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=params.weight_decay)
    elif params.optim_method == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=params.learning_rate, betas=(0.9, 0.999), weight_decay=params.weight_decay)
    elif params.optim_method == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, weight_decay=params.weight_decay)
    else:
        raise ValueError("Unknown optimizer, must be one of 'sgd'/'adam'.")

    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(1 + 0.05*epoch)) # 动态改变学习率

    # Train and evaluate the model
    logging.info("Starting training for {}  epoch(s)".format(params.epoch_num))
    train_and_evaluate(model=model,
                       train_data=train_data,
                       val_data=val_data,
                       optimizer=optimizer,
                       scheduler=scheduler,
                       # scheduler=None,
                       params=params,
                       metric_labels=metric_labels,
                       model_dir=args.model_dir,
                       restore_file=args.restore_file)


def BiLSTM_MaxPooling(data_loader,params):
    # Define the model and optimizer
    model = net.BiLSTM_MaxPooling(data_loader, params)
    if params.optim_method == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=params.weight_decay)
    elif params.optim_method == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=params.learning_rate, betas=(0.9, 0.999), weight_decay=params.weight_decay)
    elif params.optim_method == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, weight_decay=params.weight_decay)
    else:
        raise ValueError("Unknown optimizer, must be one of 'sgd'/'adam'.")

    # scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(1 + 0.05*epoch)) # 动态改变学习率

    # Train and evaluate the model
    logging.info("Starting training for {}  epoch(s)".format(params.epoch_num))
    train_and_evaluate(model=model,
                       train_data=train_data,
                       val_data=val_data,
                       optimizer=optimizer,
                       # scheduler=scheduler,
                       scheduler=None,
                       params=params,
                       metric_labels=metric_labels,
                       model_dir=args.model_dir,
                       restore_file=args.restore_file)

if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Use GPU if available
    if torch.cuda.is_available():
        params.gpu = args.gpu
    else:
        params.gpu = -1
    
    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.gpu >= 0:
        torch.cuda.set_device(params.gpu)
        torch.cuda.manual_seed(230)
    
    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")
    
    # Initialize the DataLoader
    data_loader = DataLoader(data_dir=args.data_dir,
                             embedding_file=args.embedding_file,
                             word_emb_dim=params.word_emb_dim,
                             max_len=params.max_len,
                             pos_dis_limit=params.pos_dis_limit,
                             pad_word='<pad>',
                             unk_word='<unk>',
                             other_label='Other',
                             gpu=params.gpu)
    # Load word embdding
    data_loader.load_embeddings_from_file_and_unique_words(emb_path=args.embedding_file,
                                                           emb_delimiter=' ',
                                                           verbose=True)
    metric_labels = data_loader.metric_labels  # relation labels to be evaluated
    
    # Load data
    train_data = data_loader.load_data('train')
    # Due to the small dataset, the test data is used as validation data!
    val_data = data_loader.load_data('test')

    # Specify the train and val dataset sizes
    params.train_size = train_data['size']
    params.val_size = val_data['size']

    logging.info("- done.")

    # train with CNN model
    CNN(data_loader,params)

    # train with BiLSTM + attention
    """
        train:Adadelta 动态调整学习率
        batch_size : 10
        1、动态max_len
        - Eval metrics: precison: 79.13; recall: 82.29; f1: 80.68
        2、固定max_len = 98
    """
    # BiLSTM_Att(data_loader,params)

    # BiLSTM_MaxPooling(data_loader,params)

