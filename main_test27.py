from tqdm import tqdm
import torch.nn as nn
from config import *
from MST import *
import matplotlib.pyplot as plt
print("updated main_24")

"""
train : python main_.py -bsz=16 -dr=0.2 -lr=0.001 -gate=True -ds=multiwoz -an='model'
test : python test5.py -bsz=16 -dr=0.2 -lr=0.001 -gate=True -ds=multiwoz -path=model_path
"""
if args['dataset']=='multiwoz':
    from utils_multiWOZ_DST import *
    early_stop = None
elif args['dataset']=='kvr':
    from utils_Ent_kvr import *
elif args['dataset']=='babi':
    from utils_babi import *
else:
    print("exit")
avg_best, cnt, acc = 0.0, 0, 0.0
train, dev, test, test_special, lang, SLOTS_LIST, gating_dict, max_word = prepare_data_seq(True, args['task'], False, batch_size=int(args['batch']))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


model = globals()['MST'](
    hidden_size=400,
    lang=lang,
    path=args['path'],
    task=args['task'],
    lr=float(args['learn']),
    dropout=float(args['drop']),
    slots=SLOTS_LIST,   #  self.slots = slots[0]   self.slot_temp = slots[2]
    gating_dict=gating_dict,
    nb_train_vocab=max_word)

print("====================")
print("model parameters : ", count_parameters(model))
print("====================")



loss = []
loss_gate = []
loss_ptr = []

count = 0
epochs = 50
loss = []
loss_gate = []
loss_ptr = []
count = 0

directory = './save/' + str(args['addName']+'/')
if not os.path.exists(directory):
    os.makedirs(directory)


acc = model.evaluate(dev, avg_best, SLOTS_LIST[2], early_stop)
for epoch in range(epochs):
    print("Epoch:{}".format(epoch))
    # Run the train function
    pbar = tqdm(enumerate(train),total=len(train))

    for i, data in pbar:
        # SLOTS_LIST[1] = slot_temp // SLOTS_LIST = [ALL_SLOTS, slot_train, slot_dev, slot_test]
        # print("Train data length : ", len(data[0]))
        model.train_batch(data, int(args['clip']), SLOTS_LIST[1], i, reset=(i==0)) # encode and decode

        model.optimize(args['clip']) ### 파라미터 업데이트 하는 부분
        pbar.set_description(model.print_loss())

        if i%1000 == 0:
            count += 1; loss.append(model.loss); loss_gate.append(model.loss_gate); loss_ptr.append(model.loss_ptr);
            fig, ax = plt.subplots(1, 1, figsize=(8, 12))
            ax.plot(list(range(count)), loss, label='Total Loss')
            ax.plot(list(range(count)), loss_ptr, label='Loss ptr')
            ax.plot(list(range(count)), loss_gate, label='Loss gate')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend(loc='upper right')
            plt.savefig(directory+'result.png')
            plt.cla()
            plt.close(fig)
        # print(data)
        # exit(1)
    if((epoch+1) % int(args['evalp']) == 0):
        acc = model.evaluate(dev, avg_best, SLOTS_LIST[2], early_stop, epoch)
        model.scheduler.step(acc)
        # acc = model.evaluate(test, avg_best, SLOTS_LIST[3], early_stop)
        # model.scheduler.step(acc)

        if(acc >= avg_best):
            avg_best = acc
            cnt=0
            best_model = model
        else:
            cnt+=1

    directory = './save/' + str(args['addName'])
    with open(directory+"/loss.txt", 'w') as pfile:
        loss_ = np.array(loss)
        pfile.write(str(loss_))
    with open(directory+"/loss_gate.txt", 'w') as pfile:
        loss_ = np.array(loss_gate)
        pfile.write(str(loss_))
    with open(directory+"/loss_ptr.txt", 'w') as pfile:
        loss_ = np.array(loss_ptr)
        pfile.write(str(loss_))




