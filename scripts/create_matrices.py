import argparse
import math
import gzip
import json
import logging
import pickle
import numpy

logger = logging.getLogger("create_matrices")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--topic_annotations", dest="topic_annotations", help="Input file")    
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--min_time", dest="min_time", default=-200, type=int)
    parser.add_argument("--window_size", dest="window_size", default=20, type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
        
    # text author title time window num
    docs = {}
    doc2title, doc2author, doc2year = {}, {}, {}
    unique_times = set()
    unique_words = set()
    unique_topics = set()
    unique_authors = set()
    with gzip.open(args.topic_annotations, "rt") as ifd:
        for i, line in enumerate(ifd):
            j = json.loads(line)
            title = j["title"]
            author = j["author"]
            year = j["time"]
            num = j["num"]
            id = j["id"].split("_")[0]
            doc2title[id] = title
            doc2author[id] = author
            doc2year[id] = year
            docs[id] = docs.get(id, [])
            docs[id].append(j)
            unique_times.add(year)
            unique_authors.add(author)
            for w, t in j["text"]:
                if t != None:
                    unique_words.add(w)
                    unique_topics.add(t)

    sorted_times = list(sorted(unique_times))

    min_time = args.min_time #sorted_times[0]
    max_time = sorted_times[-1]

    min_time = sorted_times[0]
    max_time = sorted_times[-1]

    time2window = {}
    cur_min_time = min_time
    cur_max_time = min_time
    unique_windows = set()
    for i in range(math.ceil((max_time - min_time + 1) / args.window_size)):
        cur_max_time += args.window_size
        j = 0
        while j < len(sorted_times) and sorted_times[j] < cur_max_time:
            time2window[sorted_times[j]] = i
            j += 1
            key = (cur_min_time, cur_max_time)
        sorted_times = sorted_times[j:]
        cur_min_time = cur_max_time

    nwins = len(set(time2window.values()))
    nwords = len(unique_words)
    ntopics = len(unique_topics)
    nauths = len(unique_authors)
    ndocs = len(docs)
    word2id = {w : i for i, w in enumerate(unique_words)}
    id2word = {i : w for w, i in word2id.items()}
    author2id = {a : i for i, a in enumerate(unique_authors)}
    id2author = {i : a for a, i in author2id.items()}
    levy2id = {d : i for i, d in enumerate(docs.keys())}
    id2levy = {i : d for d, i in levy2id.items()}
    
    logger.info(
        "Found %d windows, %d unique words, %d unique topics, %d unique documents, and %d unique authors",
        nwins,
        nwords,
        ntopics,
        ndocs,
        nauths
    )

    words_wins_topics = numpy.zeros(shape=(nwords, nwins, ntopics))
    auths_wins_topics = numpy.zeros(shape=(nauths, nwins, ntopics))
    levy_wins_topics = numpy.zeros(shape=(ndocs, nwins, ntopics))
    
    for levy, subdocs in docs.items():
        title = doc2title[levy]
        author = doc2author[levy]
        year = doc2year[levy]
        aid = author2id[author]
        win = time2window[year]
        did = levy2id[levy]
        
        for subdoc in subdocs:
            for word, topic in subdoc["text"]:
                if topic != None:
                    wid = word2id[word]
                    words_wins_topics[wid, win, topic] += 1
                    auths_wins_topics[aid, win, topic] += 1
                    levy_wins_topics[did, win, topic] += 1
    
    matrices = {
        "start" : min_time,
        "window_size" : args.window_size,
        "id2author" : id2author,
        "id2word" : id2word,
        "id2levy" : id2levy,
        "doc2title" : doc2title,
        "doc2author" : doc2author,
        "doc2year" : doc2year,
        "wwt" : words_wins_topics,
        "awt" : auths_wins_topics,
        "lwt" : levy_wins_topics
    }

    with gzip.open(args.output, "wb") as ofd:
        ofd.write(pickle.dumps(matrices))
