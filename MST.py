from torch.optim import lr_scheduler
from torch import optim
import matplotlib.pyplot as plt
import seaborn as sns
import random
from enc_dec import *
# from transformer_encoder2 import *
from masked_cross_entropy import *
from config import *

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.enabled = False
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class MST(nn.Module):
    def __init__(self, hidden_size, lang, path, task, lr, dropout, slots, gating_dict, nb_train_vocab=0):
        print("attn test")
        super(MST, self).__init__()
        self.name = "MST"
        self.task = task
        self.hidden_size = hidden_size
        self.lang = lang[0]
        self.mem_lang = lang[1]
        self.lr = lr
        self.dropout = dropout
        self.slots = slots[0]
        self.slot_list = slots[1]
        # self.slot_temp = slots[2]
        self.gating_dict = gating_dict
        self.nb_gate = len(gating_dict)
        self.cross_entorpy = nn.CrossEntropyLoss()
        self.countFig = 0

        # self.encoder = PositionAwareRNN(self.dropout, self.lang.n_words, self.hidden_size, num_layer=1, pe_dim=30, attn_dim=200)
        self.encoder = EncoderRNN(self.lang.n_words, hidden_size, self.dropout, self.lang)
        # generator에서 slot에 대한 임베딩 self.Slot_emb = nn.Embedding(len(self.slot_w2i), hidden_size)
        self.decoder = Generator(self.lang, self.encoder.embedding, self.lang.n_words, hidden_size, self.dropout,
                                 self.slots, self.nb_gate)

        if path:
            if USE_CUDA:
                print("MODEL {} LOADED".format(str(path)))
                trained_encoder = torch.load(str(path) + '/enc.th')
                trained_decoder = torch.load(str(path) + '/dec.th')
            else:
                print("MODEL {} LOADED".format(str(path)))
                trained_encoder = torch.load(str(path) + '/enc.th', lambda storage, loc: storage)
                trained_decoder = torch.load(str(path) + '/dec.th', lambda storage, loc: storage)


            self.encoder.load_state_dict(trained_encoder.state_dict())
            self.decoder.load_state_dict(trained_decoder.state_dict())

        # Initialize optimizers and criterion

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=1,
                                                        min_lr=0.0001, verbose=True)
        if path:
            checkpoint = torch.load(str(path) + '/checkpoint.pth.tar')
            print("loading checkpoint ......!!! ")
            self.cuda()
            self.optimizer = optim.Adam(self.parameters())
            # self.scheduler = lr_scheduler.ReduceLROnPlateau(self.parameters())
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            # for state in self.scheduler.state.values():
            #     for k, v in state.items():
            #         if torch.is_tensor(v):
            #             state[k] = v.cuda()

        self.reset()
        if USE_CUDA:
            self.encoder.cuda()
            self.decoder.cuda()

    def print_loss(self):
        print_loss_avg = self.loss / self.print_every
        print_loss_ptr = self.loss_ptr / self.print_every
        print_loss_gate = self.loss_gate / self.print_every
        # print_loss_class = self.loss_class / self.print_every
        # # print_loss_domain = self.loss_domain / self.print_every
        self.print_every += 1
        return 'L:{:.2f},LP:{:.2f},LG:{:.2f}'.format(print_loss_avg, print_loss_ptr, print_loss_gate)

    def save_model(self, dec_type):
        # directory = 'save/MST-'+args["addName"]+args['dataset']+str(self.task)+'/'+'HDD'+str(self.hidden_size)+'BSZ'+str(args['batch'])+'DR'+str(self.dropout)+str(dec_type)
        directory = '/content/gdrive/My Drive/BSZ' + str(args['addName']) + str(args['batch']) + 'DR' + str(
            self.dropout) + str(dec_type)  # google colab
        directory = './save/' + str(args['addName'])
        if not os.path.exists(directory):
            os.makedirs(directory)

        state = {'state_dict': self.state_dict(),
             'optimizer': self.optimizer.state_dict(), 'scheduler' : self.scheduler.state_dict() }
        torch.save(state, directory + '/checkpoint.pth.tar')
        torch.save(self.encoder, directory + '/enc.th')
        torch.save(self.decoder, directory + '/dec.th')
        print("MODEL SAVED")

    def reset(self):
        self.loss, self.print_every, self.loss_ptr, self.loss_gate, self.loss_class = 0, 1, 0, 0, 0

    def train_batch(self, data, clip, slot_temp, iter, reset=0):


        if reset: self.reset()
        # Zero gradients of both optimizers
        self.optimizer.zero_grad()

        use_teacher_forcing = random.random() < args["teacher_forcing_ratio"]

        all_point_outputs, gates, words_point_out, words_class_out= self.encode_and_decode(data, use_teacher_forcing,
                                                                                            slot_temp)


        #print(all_point_outputs.shape, data["generate_y"].shape) # orch.Size([30, 16, 4, 18311]) torch.Size([16, 30, 4])


        loss_ptr = masked_cross_entropy_for_value(
            all_point_outputs.transpose(0, 1).contiguous(),
            data["generate_y"].contiguous(),  # [:,:len(self.point_slots)].contiguous(),
            data["y_lengths"])  # [:,:len(self.point_slots)])


        loss_gate = self.cross_entorpy(gates.transpose(0, 1).contiguous().view(-1, gates.size(-1)),
                                       data["gating_label"].contiguous().view(-1))

        if args["use_gate"]:
            loss = loss_ptr + loss_gate
        else:
            loss = loss_ptr


        self.loss_grad = loss
        self.loss += loss.data
        self.loss_ptr += loss_ptr.item()
        self.loss_gate += loss_gate.item()

    def optimize(self, clip):
        self.loss_grad.backward(retain_graph=True)
        # clip_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
        self.optimizer.step()

    def encode_and_decode(self, data, use_teacher_forcing, slot_temp):


        # Build unknown mask for memory to encourage generalization
        # print("story shape : ", data['context'].shape)
        # story = self.masking(data, 'context')
        # pos = self.masking(data, 'pos_context')

        if args['unk_mask'] and self.decoder.training:
            story_size = data['context'].size()
            rand_mask = np.ones(story_size)
            bi_mask = np.random.binomial([np.ones((story_size[0], story_size[1]))], 1 - self.dropout)[0]
            rand_mask = rand_mask * bi_mask
            rand_mask = torch.Tensor(rand_mask)
            if USE_CUDA:
                rand_mask = rand_mask.cuda()
            story = data['context'] * rand_mask.long()
            # pos = data['pos_context'] * rand_mask.long()
        else:
            story = data['context']

            # pos = data['pos_context']

        encoded_outputs, encoded_hidden, attention_score = self.encoder(story.transpose(0, 1).long(),
                                                                        data['context_len'])  # torch.Size([329, 32])

        """--- self-attention visualization ---"""
        condata = data['context'][0]
        condata = condata.view(condata.size(0), 1)
        attn = attention_score[0]
        self.countFig += 1
        with torch.no_grad():
            if self.countFig == 1000:
                self.visualize_attend(attn, condata)
                self.countFig = 0

        """ --- attention matrix test --- """
        # i = 0
        # j = 0
        # input_words = [self.lang.index2word[w_idx.item()] for w_idx in condata]
        # print(input_words)
        # print("=============")
        # for row in attn:
        #     for col in row:
        #         if col > 0.5:
        #             print(input_words[i], input_words[j], " >>> ", col)
        #         j += 1
        #     j = 0
        #     i += 1

        batch_size = len(data['context_len'])
        self.copy_list = data['context_plain']
        max_res_len = data['generate_y'].size(2) if self.encoder.training else 10

        all_point_outputs, all_gate_outputs, words_point_out, words_class_out = self.decoder.forward(batch_size, \
                                                                                                     encoded_hidden,
                                                                                                     encoded_outputs,
                                                                                                     data['context_len'],
                                                                                                     story, max_res_len,
                                                                                                     data['generate_y'],
                                                                                                     use_teacher_forcing,
                                                                                                     slot_temp)

        return all_point_outputs, all_gate_outputs, words_point_out, words_class_out

    def evaluate(self, dev, matric_best, slot_temp, early_stop=None):
        self.encoder.train(False)
        self.decoder.train(False)
        print("STARTING EVALUATION")

        all_prediction = {}
        inverse_unpoint_slot = dict([(v, k) for k, v in self.gating_dict.items()])
        pbar = tqdm(enumerate(dev), total=len(dev))
        for j, data_dev in pbar:
            # Encode and Decode
            batch_size = len(data_dev['context_len'])
            print("batch_size : ", batch_size)
            _, gates, words, class_words = self.encode_and_decode(data_dev, False, slot_temp)

            for bi in range(batch_size):
                if data_dev["ID"][bi] not in all_prediction.keys():
                    all_prediction[data_dev["ID"][bi]] = {}
                all_prediction[data_dev["ID"][bi]][data_dev["turn_id"][bi]] = {
                    "turn_belief": data_dev["turn_belief"][bi]}
                predict_belief_bsz_ptr, predict_belief_bsz_class = [], []
                gate = torch.argmax(gates.transpose(0, 1)[bi], dim=1)
                #
                # context = [self.lang.index2word[w_idx.item()] for w_idx in data_dev['context'][bi]]
                # print(context)
                # print("================")
                # print(gate)
                # print("===================")
                # pointer-generator results
                ptr_slots = []
                if args["use_gate"]:
                    for si, sg in enumerate(
                        gate):  # slot에 대한 gate에 따라 value 부여 - ptr이면 generated value 넣고 none 이면 값 안넣음
                        if sg == self.gating_dict["none"]:
                            continue
                        elif sg == self.gating_dict["ptr"]:
                            pred = np.transpose(words[si])[bi]
                            st = []
                            for e in pred:
                                if e == 'EOS':
                                    break
                                else:
                                    st.append(e)
                            st = " ".join(st)
                            if st == "none":
                                continue
                            else:
                                predict_belief_bsz_ptr.append(slot_temp[si] + "-" + str(st))
                                ptr_slots.append(slot_temp[si])
                        else:
                            predict_belief_bsz_ptr.append(slot_temp[si] + "-" + inverse_unpoint_slot[sg.item()])

                    # print(ptr_slots)
                else:
                    for si, _ in enumerate(gate):
                        pred = np.transpose(words[si])[bi]
                        st = []
                        for e in pred:
                            if e == 'EOS':
                                break
                            else:
                                st.append(e)
                        st = " ".join(st)
                        if st == "none":
                            continue
                        else:
                            predict_belief_bsz_ptr.append(slot_temp[si] + "-" + str(st))

                all_prediction[data_dev["ID"][bi]][data_dev["turn_id"][bi]]["pred_bs_ptr"] = predict_belief_bsz_ptr

                if set(data_dev["turn_belief"][bi]) != set(predict_belief_bsz_ptr) and args["genSample"]:
                    print("True", set(data_dev["turn_belief"][bi]))
                    print("Pred", set(predict_belief_bsz_ptr), "\n")

        json.dump(all_prediction, open("all_prediction_{}.json".format(self.name), 'w'), indent=4)

        print(all_prediction)
        joint_acc_score_ptr, F1_score_ptr, turn_acc_score_ptr = self.evaluate_metrics(all_prediction, "pred_bs_ptr",
                                                                                      slot_temp)

        evaluation_metrics = {"Joint Acc": joint_acc_score_ptr, "Turn Acc": turn_acc_score_ptr,
                              "Joint F1": F1_score_ptr}
        print(evaluation_metrics)

        self.encoder.train(True)
        self.decoder.train(True)

        joint_acc_score = joint_acc_score_ptr  # (joint_acc_score_ptr + joint_acc_score_class)/2
        F1_score = F1_score_ptr
        self.save_model('ENTF1-{:.4f}'.format(F1_score))
        if (early_stop == 'F1'):
            if (F1_score >= matric_best):
                self.save_model('ENTF1-{:.4f}'.format(F1_score))
                print("MODEL SAVED")
            return F1_score
        else:
            if (joint_acc_score >= matric_best):
                self.save_model('ACC-{:.4f}'.format(joint_acc_score))
                print("MODEL SAVED")
            return joint_acc_score

    def evaluate_metrics(self, all_prediction, from_which, slot_temp):
        total, turn_acc, joint_acc, F1_pred, F1_count = 0, 0, 0, 0, 0
        for d, v in all_prediction.items():
            for t in range(len(v)):
                cv = v[t]
                if set(cv["turn_belief"]) == set(cv[from_which]):
                    joint_acc += 1
                total += 1

                temp_acc = self.compute_acc(set(cv["turn_belief"]), set(cv[from_which]), slot_temp)
                turn_acc += temp_acc

                temp_f1, temp_r, temp_p, count = self.compute_prf(set(cv["turn_belief"]), set(cv[from_which]))
                F1_pred += temp_f1
                F1_count += count

        joint_acc_score = joint_acc / float(total) if total != 0 else 0
        turn_acc_score = turn_acc / float(total) if total != 0 else 0
        F1_score = F1_pred / float(F1_count) if F1_count != 0 else 0

        string = str(joint_acc_score) + ' / ' + str(turn_acc_score) + ' / ' + str(F1_score) + '\n'
        directory = './save/' + str(args['addName'] + '/')

        if not os.path.exists(directory):
            os.makedirs(directory)
        f = open(directory + str(random.random()) + 'result.txt', 'w')
        f.write(string)
        f.close()
        return joint_acc_score, F1_score, turn_acc_score

    def compute_acc(self, gold, pred, slot_temp):

        # print("====compute_acc=====================")
        # print("gold : ", list(gold))
        # print("pred : ", list(pred))
        miss_gold = 0
        miss_slot = []
        for g in gold:
            if g not in pred:
                miss_gold += 1
                miss_slot.append(g.rsplit("-", 1)[0])
        wrong_pred = 0
        for p in pred:
            if p not in gold and p.rsplit("-", 1)[0] not in miss_slot:  # slot만은 찾았는지 확인
                wrong_pred += 1
        ACC_TOTAL = len(slot_temp)
        ACC = len(slot_temp) - miss_gold - wrong_pred
        ACC = ACC / float(ACC_TOTAL)
        return ACC

    def compute_prf(self, gold, pred):

        # print("====compute_prf=====================")
        # print("gold : ", list(gold))
        # print("pred : ", list(pred))

        TP, FP, FN = 0, 0, 0
        if len(gold) != 0:
            count = 1
            for g in gold:
                if g in pred:
                    TP += 1
                else:
                    FN += 1
            for p in pred:
                if p not in gold:
                    FP += 1
            precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
            recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
            F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
        else:
            if len(pred) == 0:
                precision, recall, F1, count = 1, 1, 1, 1
            else:
                precision, recall, F1, count = 0, 0, 0, 1
        return F1, recall, precision, count

    def visualize_attend(self, prob, input_seq):
        directory = './save/' + str(args['addName'] + '/')
        if not os.path.exists(directory):
            os.makedirs(directory)
        input_words = [self.lang.index2word[w_idx.item()] for w_idx in input_seq]
        sns.set()
        plt.clf()
        input_words = input_words[:20]
        plt.figure(figsize=(len(input_words), len(input_words)))
        # prob = prob.cpu().numpy()
        ax = sns.heatmap(prob[:len(input_words) + 2, : len(input_words) + 2],
                         xticklabels=input_words,
                         yticklabels=input_words)

        ax.invert_yaxis()
        plt.savefig(directory+'attention' + str(random.random()) + '.png')
        plt.cla()
        plt.close()


class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))
