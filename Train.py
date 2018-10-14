import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn.utils as utils


def train(train_iter, test_iter, dev_iter, model, args):
    # torch.manual_seed(0)

    best_dev = 0
    best_test = 0

    if args.cuda:
        model.to('cuda')

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, weight_decay=args.l2)

    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        print("Current epoch:", epoch)
        for batch in train_iter:
            feature, target = batch.text, batch.label
            feature.data.t_(), target.data.sub_(1)  # batch first, index align
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)

            # print('logit vector', logit.size())
            # print('target vector', target.size())
            loss = F.cross_entropy(logit, target)
            loss.backward()

            utils.clip_grad_norm(model.parameters(), max_norm=10)

            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * float(corrects) / batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                             loss.data[0],
                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size))
            if steps % args.test_interval == 0:
                dev_acc = eval(dev_iter, model, args)
                test_acc = eval(test_iter, model, args)

                print("Dev accuracy", dev_acc)
                print("Test accuracy", test_acc)

                if dev_acc > best_dev:
                    best_dev = dev_acc
                    best_test = test_acc
                print("Best dev acc is %.4f and corresponding test acc is %.4f."
                      % (best_dev, best_test))

                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
            elif steps % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', steps)

        save_param(args.save_dir, accuracy, args)


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=True)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * float(corrects) / size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    model.train()
    return accuracy


def predict(text, model, text_field, label_feild, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = text_field.tensor_type(text)
    x = autograd.Variable(x, volatile=True)
    if cuda_flag:
        x = x.cuda()
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    # return label_feild.vocab.itos[predicted.data[0][0]+1]
    return label_feild.vocab.itos[predicted.data[0] + 1]


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)


def save_param(save_dir, accuracy, args):
    # Record the adapted parameters and accuracy in txt file.
    # print("\n\nSave parameters entered through command line.")
    with open(save_dir + '/parameters.txt', 'w') as f:
        for attr, value in args.__dict__.items():
            f.write('%-15s = %-s\n' % (attr, value))
        f.write('\nAccuracy\t\t\t{}%'.format(accuracy))