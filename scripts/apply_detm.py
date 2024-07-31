import logging
import gzip
import math
import json
import argparse
import numpy as np
import numpy
import torch
from detm_copy import DETM
import os.path

logger = logging.getLogger("apply_detm")


def split_doc(tokens, max_length):
    num_subdocs = math.ceil(len(tokens) / max_length)
    retval = []
    for i in range(num_subdocs):
        retval.append(tokens[i * max_length : (i + 1) * max_length])
    return retval

def get_eta(model, rnn_inp, device):
    inp = model.q_eta_map(rnn_inp).unsqueeze(1)
    hidden = model.init_hidden()
    output, _ = model.q_eta(inp, hidden)
    output = output.squeeze()
    etas = torch.zeros(model.num_times, model.num_topics).to(device)
    inp_0 = torch.cat([output[0], torch.zeros(model.num_topics,).to(device)], dim=0)
    etas[0] = model.mu_q_eta(inp_0)
    for t in range(1, model.num_times):
        inp_t = torch.cat([output[t], etas[t-1]], dim=0)
        etas[t] = model.mu_q_eta(inp_t)
    return etas

def get_theta(eta, bows):
    inp = torch.cat([bows, eta], dim=1)
    q_theta = model.q_theta(inp)
    mu_theta = model.mu_q_theta(q_theta)
    theta = torch.nn.functional.softmax(mu_theta, dim=-1)
    return theta

def apply(model, data, norm_data, times, rnn_input, num_subdocs, device):
    alpha = model.mu_q_alpha
    acc_loss = 0.0
    cnt = 0
    val_batch_size = 1000

    eta = get_eta(model, rnn_input, device)


    eta_td = eta[times.type('torch.LongTensor')]
    theta = get_theta(eta_td, norm_data)
    alpha_td = alpha[:, times.type('torch.LongTensor'), :]
    beta = model.get_beta(alpha_td).permute(1, 0, 2)

    lik = theta.unsqueeze(2) * beta

    return lik.to("cpu")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", dest="input", help="Data file")
    parser.add_argument("--model", dest="model", help="Model file")
    parser.add_argument("--output", dest="output", help="File to save model to", required=True)
    parser.add_argument("--max_subdoc_length", dest="max_subdoc_length", type=int, default=200, help="Documents will be split into subdocuments of at most this number of tokens")
    parser.add_argument('--device') #, choices=["cpu", "cuda"], help='')
    parser.add_argument('--batch_size', type=int, default=100, help='')
    #parser.add_argument("--")
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


    if not args.device:
       args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    elif args.device == "cuda" and not torch.cuda.is_available():
       logger.warning("Setting device to CPU because CUDA isn't available")
       args.device = "cpu"

    with gzip.open(args.model, "rb") as ifd:
        model = torch.load(ifd, map_location=torch.device(args.device))



    token2id = {v : k for k, v in model.id2token.items()}
    id2token = model.id2token

    all_subdocs = []
    model.eval()
    with torch.no_grad():
        with gzip.open(args.input, "rt") as ifd:
            for line in ifd:

                j = json.loads(line)
                time = int(j["year"])
                window = int((time - model.min_time) / model.window_size)
                title = j["title"]
                author = j.get("author", 0)
                doc = []
                split_content = j["content"].split()
                for x in split_content:
                    doc.append(x)
                subdocs = []
                subdoc_texts = []
                for subdoc_num, subdoc in enumerate(split_doc(doc, args.max_subdoc_length)):
                    subdoc_counts = {}
                    subdoc_text = []
                    for t in subdoc:
                        if t in token2id:
                            tid = token2id[t]
                            subdoc_counts[tid] = subdoc_counts.get(tid, 0) + 1
                            subdoc_text.append((t, tid))
                        else:
                            subdoc_text.append((t, None))
                    if len(subdoc_counts) > 0:
                        all_subdocs.append(
                            {
                                "text" : subdoc_text,
                                "counts" : subdoc_counts,
                                "author" : author,
                                "title" : title,
                                "time" : time,
                                "window" : window,
                                "num" : subdoc_num,
                                "id" : j["htid"] + "_" + str(subdoc_num),
                            }
                        )
            
        logger.info("Found %d sub-docs", len(all_subdocs))

        indices = torch.arange(0, len(all_subdocs), dtype=torch.int)
        indices = torch.split(indices, args.batch_size) 
        rnn_input = torch.zeros(model.num_times, len(model.id2token)).to(args.device)
        cnt = torch.ones(model.num_times, ).to(args.device)
        for idx, ind in enumerate(indices):
            batch_size = len(ind)
            data_batch = np.zeros((batch_size, len(id2token)))
            times_batch = np.zeros((batch_size, ))
            for i, doc_id in enumerate(ind):
                #timestamp, yr, doc, title, author, _ =
                subdoc = all_subdocs[doc_id]
                times_batch[i] = subdoc["window"] #timestamp
                for k, v in subdoc["counts"].items():
                    data_batch[i, k] = v
            data_batch = torch.from_numpy(data_batch).float().to(args.device)
            times_batch = torch.from_numpy(times_batch).to(args.device)
            for t in range(model.num_times):
                tmp = (times_batch == t).nonzero()
                docs = data_batch[tmp].squeeze().sum(0)
                rnn_input[t] += docs
                cnt[t] += len(tmp)
        rnn_input = rnn_input / cnt.unsqueeze(1)
        
        
        indices = torch.arange(0, len(all_subdocs), dtype=torch.int)
        indices = torch.split(indices, args.batch_size)
        for idx, ind in enumerate(indices):
            batch_size = len(ind)
            data_batch = np.zeros((batch_size, len(model.id2token)))
            times_batch = np.zeros((batch_size, ))
            subdocs_batch = []
            for i, doc_id in enumerate(ind):
                subdoc = all_subdocs[doc_id]
                times_batch[i] = subdoc["window"] #timestamp
                for k, v in subdoc["counts"].items():
                    data_batch[i, k] = v
            data_batch = torch.from_numpy(data_batch).float().to(args.device)
            times_batch = torch.from_numpy(times_batch).to(args.device)
            sums = data_batch.sum(1).unsqueeze(1)
            normalized_data_batch = data_batch / sums
            out = apply(
                model,
                data_batch,
                normalized_data_batch,
                times_batch,
                rnn_input,
                len(all_subdocs),
                args.device
            )
            for lik, i in zip(out, ind):
                lik = lik.argmax(0)
                all_subdocs[i]["text"] = [(tok, lik[token2id[tok]].item()) if tok in token2id else (tok, None) for tok, tid in all_subdocs[i]["text"]]
                del all_subdocs[i]["counts"]

        with gzip.open(args.output, "wt") as ofd:
            for sd in all_subdocs:
                ofd.write(json.dumps(sd) + "\n")


	## visualize
"""
        model.eval()
        with torch.no_grad():
            alpha = model.mu_q_alpha
            beta = model.get_beta(alpha) 
            print('beta: ', beta.size())
            print('\n')
            print('#'*100)
            print('Visualize topics...')
            times = range(beta.shape[1])
            topics_words = []
            #print(beta.shape)
            for k in range(beta.shape[0]):
                for t in times:
                    gamma = beta[k, t, :]
                    print(gamma.shape)
                    top_words = list(gamma.cpu().numpy().argsort()[-beta.shape[0]+1:][::-1])
                    topic_words = [model.id2token[a] for a in top_words]
                    topics_words.append(' '.join(topic_words))
                    print('Topic {} .. Time: {} ===> {}'.format(k, t, topic_words)) 
"""
