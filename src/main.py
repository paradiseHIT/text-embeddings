import json
import time
import ssl
import sys, getopt
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from flask import Flask, redirect, url_for
app = Flask(__name__)

# Use tensorflow 1 behavior to match the Universal Sentence Encoder
# examples (https://tfhub.dev/google/universal-sentence-encoder/2).
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub

##### INDEXING #####

def index_data():
    print("Creating the 'posts' index.")
    client.indices.delete(index=INDEX_NAME, ignore=[404])

    with open(INDEX_FILE) as index_file:
        source = index_file.read().strip()
        client.indices.create(index=INDEX_NAME, body=source)

    docs = []
    count = 0

    with open(DATA_FILE) as data_file:
        for line in data_file:
            line = line.strip()

            doc = json.loads(line)
            if doc["type"] != "question":
                continue

            docs.append(doc)
            count += 1

            if count % BATCH_SIZE == 0:
                index_batch(docs)
                docs = []
                print("Indexed {} documents.".format(count))

        if docs:
            index_batch(docs)
            print("Indexed {} documents.".format(count))

    client.indices.refresh(index=INDEX_NAME)
    print("Done indexing.")

def index_batch(docs):
    titles = [doc["title"] for doc in docs]
    title_vectors = embed_text(titles)

    requests = []
    for i, doc in enumerate(docs):
        request = doc
        request["_op_type"] = "index"
        request["_index"] = INDEX_NAME
        request["title_vector"] = title_vectors[i]
        requests.append(request)
    bulk(client, requests)

##### SEARCHING #####

def run_query_loop():
    while True:
        try:
            handle_query()
        except KeyboardInterrupt:
            return

@app.route('/query/<query>')
def handle_query2(query):
    embedding_start = time.time()
    query_vector = embed_text([query])[0]
    embedding_time = time.time() - embedding_start

    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'title_vector') + 1.0",
                "params": {"query_vector": query_vector}
            }
        }
    }

    search_start = time.time()
    body={
        "size": SEARCH_SIZE,
        "query": script_query,
        "_source": {"includes": ["title"]}
    }
    response = client.search(
        index=INDEX_NAME,
        body=body
    )
    search_time = time.time() - search_start

    print()
    print("{} total hits.".format(response["hits"]["total"]["value"]))
    print("embedding time: {:.2f} ms".format(embedding_time * 1000))
    print("search time: {:.2f} ms".format(search_time * 1000))
    ret_str=""
    for hit in response["hits"]["hits"]:
        print("id: {}, score: {}".format(hit["_id"], hit["_score"]))
        print(hit["_source"])
        ret_str = ret_str + " " + hit["_source"]["title"]
        print()
    return ret_str
def handle_query():
    query = input("Enter query: ")

    embedding_start = time.time()
    query_vector = embed_text([query])[0]
    embedding_time = time.time() - embedding_start

    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'title_vector') + 1.0",
                "params": {"query_vector": query_vector}
            }
        }
    }

    search_start = time.time()
    body={
        "size": SEARCH_SIZE,
        "query": script_query,
        "_source": {"includes": ["title"]}
    }
    response = client.search(
        index=INDEX_NAME,
        body=body
    )
    search_time = time.time() - search_start

    print()
    print("{} total hits.".format(response["hits"]["total"]["value"]))
    print("embedding time: {:.2f} ms".format(embedding_time * 1000))
    print("search time: {:.2f} ms".format(search_time * 1000))
    for hit in response["hits"]["hits"]:
        print("id: {}, score: {}".format(hit["_id"], hit["_score"]))
        print(hit["_source"])
        print()

##### EMBEDDING #####

def embed_text(text):
    vectors = session.run(embeddings, feed_dict={text_ph: text})
    return [vector.tolist() for vector in vectors]

##### MAIN SCRIPT #####

if __name__ == '__main__':
    INDEX_NAME = "posts"
    INDEX_FILE = "data/posts/index.json"

    DATA_FILE = "data/posts/posts.json"
    BATCH_SIZE = 1000

    SEARCH_SIZE = 5

    GPU_LIMIT = 0.5
    model_dir = None
    ca_cert_file = None
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hm:c:",["model_dir=","ca_cert_file="])
    except getopt.GetoptError:
        print('main.py -m <model_dir> -c <ca_cert_file>')
        sys.exit(-1)
    for opt, arg in opts:
        if opt == '-h':
            print('main.py -m <model_dir> -c <ca_cert_file>')
            sys.exit(0)
        elif opt in ("-m", "--model_dir"):
            model_dir = arg
        elif opt in ("-c", "--ca_cert_file"):
            ca_cert_file = arg
    if model_dir == None:
        print("model_dir is None")
        sys.exit(-1)
    if ca_cert_file == None:
        print("ca_cert_file is None")
        sys.exit(-1)
    print('model_dir=%s' % str(model_dir))
    print('ca_cert_file=%s' % str(ca_cert_file))

    #print("Downloading pre-trained embeddings from tensorflow hub...")
    #embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
    embed = hub.Module(model_dir)
    text_ph = tf.placeholder(tf.string)
    embeddings = embed(text_ph)

    print("Creating tensorflow session...")
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = GPU_LIMIT
    session = tf.Session(config=config)
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    print("Done.")

    ES=["localhost:9200"]
    context = ssl._create_unverified_context()
    client = Elasticsearch(ES,
        ca_certs=ca_cert_file,
        scheme="https",
        ssl_context=context,
        http_auth=('elastic', 'zLD*uPqtDNoybExIkEgt'))

    #index_data()
    #run_query_loop()
    app.run(host="0.0.0.0", port=8000)

    print("Closing tensorflow session...")
    session.close()
    print("Done.")
