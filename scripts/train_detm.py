import os
import json
import gzip
import argparse
import math
import random
import logging
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy as np
import numpy
import torch
import copy
from detm import DETM

logger = logging.getLogger("split_docs")

def split_doc(tokens, max_len):
	num_subdocs = math.ceil(len(tokens) / max_len)
	subdocs = []
	for i in range(num_subdocs):
		subdocs.append(tokens[i * max_len : (i + 1) * max_len])
	return subdocs

def get_theta(model, eta, bows):
	model.eval()
	with torch.no_grad():
		inp = torch.cat([bows, eta], dim=1)
		q_theta = model.q_theta(inp)
		mu_theta = model.mu_q_theta(q_theta)
		theta = torch.nn.functional.softmax(mu_theta, dim=-1)
		return theta

def _eta_helper(rnn_inp, device):
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


def get_eta(model, rnn_inp, device):
	model.eval()
	with torch.no_grad():
		return _eta_helper(rnn_inp, device)

def get_completion_ppl(model, val_subdocs, val_rnn_inp, id2token, device):
	"""Returns document completion perplexity.
	"""

	model.eval()
	with torch.no_grad():
		alpha = model.mu_q_alpha
		acc_loss = 0.0
		cnt = 0
		val_batch_size = 1000
		eta = get_eta(model, val_rnn_inp, device)
		indices = torch.split(torch.tensor(range(len(val_subdocs))), val_batch_size)
		for idx, ind in enumerate(indices):
			batch_size = len(ind)
			data_batch = np.zeros((batch_size, len(id2token)))
			times_batch = np.zeros((batch_size, ))
			for i, doc_id in enumerate(ind):
				subdoc = val_subdocs[doc_id]
				times_batch[i] = subdoc["window"] #timestamp
				for k, v in subdoc["counts"].items():
					data_batch[i, k] = v
			data_batch = torch.from_numpy(data_batch).float().to(args.device)
			times_batch = torch.from_numpy(times_batch).to(args.device)

			sums = data_batch.sum(1).unsqueeze(1)
			normalized_data_batch = data_batch / sums

			eta_td = eta[times_batch.type('torch.LongTensor')]
			theta = get_theta(model, eta_td, normalized_data_batch)
			alpha_td = alpha[:, times_batch.type('torch.LongTensor'), :]
			beta = model.get_beta(alpha_td).permute(1, 0, 2)
			loglik = theta.unsqueeze(2) * beta
			loglik = loglik.sum(1)
			loglik = torch.log(loglik)
			nll = -loglik * data_batch
			nll = nll.sum(-1)
			loss = nll / sums.squeeze()
			loss = loss.mean().item()
			acc_loss += loss
			cnt += 1
		cur_loss = acc_loss / cnt
		ppl_all = round(math.exp(cur_loss), 1)
	return ppl_all

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--input", dest="input", help="Input file") # concert_programs.json.gz
	parser.add_argument("--max_subdoc_len", dest="max_subdoc_len", type=int, default=200,\
		help="Documents will be split into subdocuments of at most this number of tokens")
	parser.add_argument('--train_proportion', type=float, default=0.7, help='')
	parser.add_argument("--min_word_occurrence", dest="min_word_occurrence", type=int, default=0, \
		help="Words occuring less than this number of times throughout the entire dataset will be ignored")
	parser.add_argument("--max_word_proportion", dest="max_word_proportion", type=float, default=1.0, \
		help="Words occurring in more than this proportion of documents will be ignored (probably conjunctions, etc)")
	parser.add_argument("--window_size", dest="window_size", type=int, default=20, help="")
	parser.add_argument("--embeddings", dest="embeddings", help="Embeddings file")
	#parser.add_argument("--output_directory", dest="output_directory", help="Directory for output files") # concert_programs_split

	arser.add_argument("--top_words", dest="top_words", type=int, default=10, help="Number of words to show for each topic in the summary file")
	parser.add_argument("--epochs", dest="epochs", type=int, default=400, help="How long to train")
	parser.add_argument("--random_seed", dest="random_seed", type=int, default=None, help="Specify a random seed (for repeatability)")

	parser.add_argument('--num_words', type=int, default=20, help='number of words for topic viz')
	parser.add_argument('--log_interval', type=int, default=10, help='when to log training')
	parser.add_argument('--visualize_every', type=int, default=1, help='when to visualize results')
	parser.add_argument('--eval_batch_size', type=int, default=1000, help='input batch size for evaluation')
	parser.add_argument('--load_from', type=str, default='', help='the name of the ckpt to eval from')
	parser.add_argument('--tc', type=int, default=0, help='whether to compute tc or not')

	parser.add_argument('--learning_rate', dest="learning_rate", type=float, default=0.0001, help='learning rate')
	parser.add_argument('--lr_factor', type=float, default=2.0, help='divide learning rate by this')
	parser.add_argument('--mode', type=str, default='train', help='train or eval model')
	parser.add_argument('--device') #, choices=["cpu", "cuda"], help='')
	parser.add_argument('--optimizer', type=str, default='adam', help='choice of optimizer')
	parser.add_argument('--seed', type=int, default=2019, help='random seed (default: 1)')
	parser.add_argument('--enc_drop', type=float, default=0.0, help='dropout rate on encoder')
	parser.add_argument('--eta_dropout', type=float, default=0.0, help='dropout rate on rnn for eta')
	parser.add_argument('--clip', type=float, default=2.0, help='gradient clipping')
	parser.add_argument('--nonmono', type=int, default=10, help='number of bad hits allowed')
	parser.add_argument('--wdecay', type=float, default=1.2e-6, help='some l2 regularization')
	parser.add_argument('--anneal_lr', type=int, default=0, help='whether to anneal the learning rate or not')
	sparser.add_argument('--bow_norm', type=int, default=1, help='normalize the bows or not')



	#parser.add_argument('--limit_docs', type=int, help='')
	parser.add_argument('--batch_size', type=int, default=100, help='')
	parser.add_argument('--num_topics', type=int, default=50, help='number of topics')
	parser.add_argument('--rho_size', type=int, default=300, help='dimension of rho')
	parser.add_argument('--emb_size', type=int, default=300, help='dimension of embeddings')
	parser.add_argument('--t_hidden_size', type=int, default=800, help='dimension of hidden space of q(theta)')
	parser.add_argument('--theta_act', type=str, default='relu', help='tanh, softplus, relu, rrelu, leakyrelu, elu, selu, glu)')
	parser.add_argument('--train_embeddings', default=False, action="store_true", help='whether to fix rho or train it')
	parser.add_argument('--eta_nlayers', type=int, default=3, help='number of layers for eta')
	parser.add_argument('--eta_hidden_size', type=int, default=200, help='number of hidden units for rnn')
	parser.add_argument('--delta', type=float, default=0.005, help='prior variance')
	parser.add_argument('--train_proportion', type=float, default=0.7, help='')
	parser.add_argument("--min_time", type=int, default=0)
	parser.add_argument("--max_time", type=int, default=0)


	args = parser.parse_args()

	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


	if args.random_seed:
		random.seed(args.random_seed)
		np.random.seed(args.seed)
		torch.backends.cudnn.deterministic = True
		torch.manual_seed(args.seed)

	if not args.device:
		args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	elif args.device == "cuda" and not torch.cuda.is_available():
		logger.warning("Setting device to CPU because CUDA isn't available")
		args.device = "cpu"

	unique_times = set()
	total_subdocs = 0
	token_subdoc_count = {}
	data = {}
	token_id_mapping = {}

	with gzip.open(os.path.expanduser("~/corpora/" + args.input), "rt") as ifd:
		all_subdocs = []
		for doc in ifd:
			# unload each document (line) into a dictionary
			j = json.loads(doc)

			time = int(j["year"])
			unique_times.add(time)

			title = j["title"]
			#print("time: " + str(time) + ", title: " + title + ", pub: " + publisher)
			author = j.get("author", "")
			publisher = j.get("publisher", "")

			#print("time: " + str(time) + ", title: " + title + ", pub: " + publisher + ", " + "author: " + author + ", " + j["htid"])

			full_text_words = j["content"].split()

			# split into subdocs and iterate over them
			for subdoc_num, subdoc in enumerate(split_doc(full_text_words, args.max_subdoc_len)):
				total_subdocs += 1
				utokens = set()

				# get unique words in a subdoc and update counts
				for t in subdoc:
					utokens.add(t)
				for t in utokens:
					token_subdoc_count[t] = token_subdoc_count.get(t, 0) + 1

				local_dict = {\
						"time" : time, \
						"tokens" : subdoc, \
						"title" : title, \
						"id" : j["htid"] + "_" + str(subdoc_num), \
						"author" : author, \
						"publisher" : publisher, \
						"subdoc_number" : subdoc_num \
					}
				all_subdocs.append(local_dict)

		random.shuffle(all_subdocs)
		num_training = math.ceil(args.train_proportion * len(all_subdocs))

		# training/validation split
		data["train"] = all_subdocs[:num_training]
		data["val"] = all_subdocs[num_training:]

		with gzip.open(os.path.expanduser("~/corpora/concert_programs_split_train.json.gz"), "wt") as ofd_train:
			for line in data["train"]:
				ofd_train.write(json.dumps(line) + "\n")
		with gzip.open(os.path.expanduser("~/corpora/concert_programs_split_val.json.gz"), "wt") as ofd_val:
			for line in data["val"]:
				ofd_val.write(json.dumps(line) + "\n")

	vocab_kept = set()
	for t, count in token_subdoc_count.items():
		if count >= args.min_word_occurrence and (count / total_subdocs) <= args.max_word_proportion:
			vocab_kept.add(t)

	logger.info("keeping %d words from a vocabulary of %d", len(vocab_kept), len(token_subdoc_count))

	sorted_times = list(sorted(unique_times))

	min_time = sorted_times[0] - 1
	max_time = sorted_times[-1] + 1
	span = max_time - min_time

	curr_min = min_time
	curr_max = min_time

	# dictionary of window : (number of unique times in window)
	window_counts = {}

	# dictionary mapping each time to a window
	time_window_mapping = {}

	j = 0
	for i in range(math.ceil(span / args.window_size)):
		curr_max += args.window_size
		print((curr_min, curr_max))
		while j < len(sorted_times) and sorted_times[j] < curr_max:
			time_window_mapping[sorted_times[j]] = i
			j += 1
			window_counts[i] = window_counts.get(i, 0) + 1
		curr_min = curr_max

	logger.info("Found %d sub-docs, min time = %d, max time = %d, window count = %d", \
		sum([len(v) for v in data.values()]), min_time, max_time, len(window_counts))

	# dict of lists (1 for train, 1 for val) of subdocs including counts for each token
	subdoc_counts = {}
	# dict of dicts (1 dict for train, 1 for val) of window : (num subdocs in window)
	window_counts = {}

	for name, vs in data.items():
		subdoc_counts[name] = []
		window_counts[name] = {}
		for subdoc in data[name]:
			window = time_window_mapping[subdoc["time"]]
			# dict of counts for each token in subdoc
			subdoc["counts"] = {}
			subdoc["window"] = window
			for t in subdoc["tokens"]:
				if t in vocab_kept:
					tid = token_id_mapping.setdefault(t, len(token_id_mapping))
					# update counts for given token
					subdoc["counts"][tid] = subdoc["counts"].get(tid, 0) + 1
			if len(subdoc["counts"]) > 0:
				subdoc_counts[name].append(subdoc)
				window_counts[name][window] = window_counts[name].get(window, 0) + 1

	# keep only windows present in both training and validation
	windows_kept = set([w for w in window_counts["train"].keys() if all([w in v for v in window_counts.values()])])

	window_transform = {w : i for i, w in enumerate(sorted(windows_kept))}

	for name in list(subdoc_counts.keys()):
		# list of dicts for each subdoc in subdoc_counts
		subdoc_counts[name] = [dict([(k, v) for k, v in s.items() if k != "window"] +\
			[("window", window_transform[s["window"]])]) for s in subdoc_counts[name] if s["window"] in windows_to_keep]
		window_counts[name] = {window_transform[k] : v for k, v in window_counts[name].items() if k in windows_to_keep}

	# dict mapping id to token
	id2token = {v : k for k, v in token_id_mapping.items()}

	if args.embeddings:
        	if args.embeddings.endswith("txt"):
			wv = {}
			with open(os.path.expanduser(""~/corpora/" + args.embeddings), "rt") as ifd:
				for line in ifd:
					toks = line.split()
					wv[toks[0]] = list(map(float, toks[1:]))
			embeddings = torch.tensor([wv[id2token[i]] for i in range(len(id2token))])

		else:
			w2v = Word2Vec.load(args.embeddings)

			test_model = KeyedVectors.load(args.embeddings)

			words = list(test_model.wv.key_to_index.keys())

			embeddings = torch.tensor(numpy.array([w2v.wv[id2token[i]] for i in range(len(id2token))]))

	args.embeddings_dim = embeddings.size()
	args.num_times = len(window_counts["train"])
 	args.vocab_size = len(id2token)
	args.train_embeddings = 0

	model = DETM( \
		args, \
		id2token, \
		min_time, \
		num_topics=args.num_topics, \
		windows=window_counts["train"], \
		embeddings=embeddings, \
		t_hidden_size=args.t_hidden_size, \
		eta_hidden_size=args.eta_hidden_size, \
		eta_drop=args.eta_dropout, \
		rho_size=args.rho_size, \
		emb_size=args.emb_size, \
		enc_drop=args.enc_drop, \
		eta_nlayers=args.eta_nlayers, \
		delta=args.delta, \
		theta_act=args.theta_act, \
		device=args.device, \
		adapt_embeddings=False \
		)

	print("initialized model")
	model.to(args.device)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.wdecay)

	rnn_input = {}
	for name in ["train", "val"]:
		indices = torch.arange(0, len(subdoc_counts[name]), dtype=torch.int)
		indices = torch.split(indices, args.batch_size) 
		rnn_input[name] = torch.zeros(len(window_counts["train"]), len(id2token)).to(args.device)
		cnt = torch.zeros(len(window_counts["train"]), ).to(args.device)
		for idx, ind in enumerate(indices):
			batch_size = len(ind)
			data_batch = np.zeros((batch_size, len(id2token)))
			times_batch = np.zeros((batch_size, ))
			for i, doc_id in enumerate(ind):
				subdoc = subdoc_counts[name][doc_id]
				times_batch[i] = subdoc["window"] #timestamp
				for k, v in subdoc["counts"].items():
					data_batch[i, k] = v
			data_batch = torch.from_numpy(data_batch).float().to(args.device)
			times_batch = torch.from_numpy(times_batch).to(args.device)
			for t in range(len(window_counts["train"])):
				tmp = (times_batch == t).nonzero()
				docs = data_batch[tmp].squeeze().sum(0)
				rnn_input[name][t] += docs
				cnt[t] += len(tmp)
		rnn_input[name] = rnn_input[name] / cnt.unsqueeze(1)

	best_state = None
	best_val_ppl = None
	since_annealing = 0
	since_improvement = 0
	for epoch in range(1, args.epochs + 1):
		logger.info("Starting epoch %d", epoch)
		model.train()
		acc_loss = 0
		acc_nll = 0
		acc_kl_theta_loss = 0
		acc_kl_eta_loss = 0
		acc_kl_alpha_loss = 0
		cnt = 0
		indices = torch.randperm(len(subdoc_counts["train"]))
		indices = torch.split(indices, args.batch_size)
		for idx, ind in enumerate(indices):
			optimizer.zero_grad()
			model.zero_grad()
			batch_size = len(ind)
			data_batch = np.zeros((batch_size, len(id2token)))
			times_batch = np.zeros((batch_size, ))
			for i, doc_id in enumerate(ind):
				subdoc = subdoc_counts["train"][doc_id]
				times_batch[i] = subdoc["window"] #timestamp
				for k, v in subdoc["counts"].items():
					data_batch[i, k] = v

			data_batch = torch.from_numpy(data_batch).float().to(args.device)
			times_batch = torch.from_numpy(times_batch).to(args.device)
			sums = data_batch.sum(1).unsqueeze(1)
			normalized_data_batch = data_batch / sums
			loss, nll, kl_alpha, kl_eta, kl_theta = model( \
				data_batch, \
				normalized_data_batch, \
				times_batch, \
				rnn_input["train"], \
				len(subdoc_counts["train"]) )

			if not torch.any(torch.isnan(loss)):
				loss.backward()
				if args.clip > 0:
					torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
				optimizer.step()

				acc_loss += torch.sum(loss).item()
				acc_nll += torch.sum(nll).item()
				acc_kl_theta_loss += torch.sum(kl_theta).item()
				acc_kl_eta_loss += torch.sum(kl_eta).item()
				acc_kl_alpha_loss += torch.sum(kl_alpha).item()
			cnt += 1

		cur_loss = round(acc_loss / cnt, 2) 
		cur_nll = round(acc_nll / cnt, 2) 
		cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
		cur_kl_eta = round(acc_kl_eta_loss / cnt, 2) 
		cur_kl_alpha = round(acc_kl_alpha_loss / cnt, 2) 
		lr = optimizer.param_groups[0]['lr']
		logger.info("Computing perplexity...")
		val_ppl = get_completion_ppl(model, subdoc_counts["val"], rnn_input["val"], id2token, args.device)
		logger.info( \
			'{}: LR: {}, KL_theta: {}, KL_eta: {}, KL_alpha: {}, Rec_loss: {}, NELBO: {}, PPL: {}'.format( \
			epoch, \
			lr, \
			cur_kl_theta, \
			cur_kl_eta, \
			cur_kl_alpha, \
			cur_nll, \
			cur_loss, \
			val_ppl \
            		) \
        		)

		if best_val_ppl == None or val_ppl < best_val_ppl:
			logger.info("Copying new best model...")
			best_val_ppl = val_ppl
			best_state = copy.deepcopy(model.state_dict())
			since_improvement = 0
			logger.info("Copied.")
		else:
			since_improvement += 1
		since_annealing += 1
		if since_improvement > 5 and since_annealing > 5 and since_improvement < 10:
			optimizer.param_groups[0]['lr'] /= args.lr_factor
			model.load_state_dict(best_state)
			since_annealing = 0
		elif since_improvement >= 10:
			break

	model.load_state_dict(best_state)
	with gzip.open(args.output, "wb") as ofd:
		torch.save(model, ofd)
