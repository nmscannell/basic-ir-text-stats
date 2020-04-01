import os
import math
import random

# vocab: {term:{docID:term frequency}}
unigrams = {}
bigrams = {}
# docs: filenames in index ID-1
docs = []
dir_path = os.path.dirname(os.path.realpath(__file__))


def process_corpus(corpus, n=1):
    """
    process_corpus takes a corpus (directory of files or file containing local filenames) and an optional integer labeling the test iteration.
    For the given corpus, store a mapping of document IDs and filenames. Produce the following files:

    Create the following 2 files, with one converted token or bigram per line followed by each statistic as a labelled pair, e.g. (cf cvalue) (df dvalue) (idf ivalue):

        HW1.testn.unigram.stats

        HW1.testn.bigram.stats

    which,  for  a term  "apple" occurring 850 times over 3 documents when  N= 1000  contains:

        apple (cf 850) (df 3) (idf 2.52)

    Create the following 2 files, with one converted token or bigram per line followed by its postings list:

        HW1.testn.unigram.postings

        HW1.testn.bigram.postings

    which, if "apple" occurred in documents 18, 50 and 700, HW1.testn.unigram.postings contains:

        apple 18 50 700
    """
    unigrams.clear()
    bigrams.clear()
    docs.clear()
    doc_id = 1
    files = get_files(os.path.join(dir_path, corpus))
    for filename in files:
        filename = os.path.join(dir_path, filename)
        with open(filename) as f:
            prev_word = ''
            for line in f.readlines():
                for word in line.split():
                    word = word.lower().strip('.,:()')
                    # store each unigram and bigram with their postings list and term frequency for each document
                    if word not in unigrams:
                        unigrams[word] = {doc_id: 1}
                    else:
                        if doc_id in unigrams[word]:
                            unigrams[word][doc_id] = unigrams[word][doc_id] + 1
                        else:
                            unigrams[word][doc_id] = 1
                    if prev_word != '':
                        bigram = prev_word + ' ' + word
                        if bigram not in bigrams:
                            bigrams[bigram] = {doc_id: 1}
                        else:
                            if doc_id in bigrams[bigram]:
                                bigrams[bigram][doc_id] = bigrams[bigram][doc_id] + 1
                            else:
                                bigrams[bigram][doc_id] = 1
                    prev_word = word
        doc_id += 1
    # create a file with each unigram, collection f, doc f, log(len(docs)/doc f) and a separate file containing the postings lists for unigrams
    file1 = "files/HW1.test"+str(n)+".unigram.stats"
    file1 = os.path.join(dir_path, file1)
    open(file1, 'w').close()
    file2 = "files/HW1.test"+str(n)+".unigram.postings"
    file2 = os.path.join(dir_path, file2)
    open(file2, 'w').close()
    with open(file1, 'a') as file:
        for u in unigrams:
            with open(file2, 'a') as f:
                s = u
                for k in unigrams[u]:
                    s += " " + str(k)
                f.write(s+'\n')
            df = len(unigrams[u])
            cf = sum(unigrams[u].values())
            num_docs = len(docs)
            out = u + " " + "(cf {0}) (df {1}) (idf {2:.2})\n".format(cf, df, math.log10(num_docs/df))
            file.write(out)
    # create a file with each unigram, collection f, doc f, log(len(docs)/doc f) and a separate file containing the postings lists for bigrams
    file1 = "files/HW1.test"+str(n)+".bigram.stats"
    file1 = os.path.join(dir_path, file1)
    open(file1, 'w').close()
    file2 = "files/HW1.test"+str(n)+".bigram.postings"
    file2 = os.path.join(dir_path, file2)
    open(file2, 'w').close()
    with open(file1, 'a') as file:
        for b in bigrams:
            with open(file2, 'a') as f:
                s = b
                for k in bigrams[b]:
                    s += " " + str(k)
                f.write(s+'\n')
            df = len(bigrams[b])
            cf = sum(bigrams[b].values())
            num_docs = len(docs)
            out = b + " " + "(cf {0}) (df {1}) (idf {2:.2})\n".format(cf, df, math.log10(num_docs/df))
            file.write(out)


def get_files(corpus):
    """
    get_files is a helper method for process_corpus. It takes in a directory or a filename of a file containing the names of files
    in the local directory that are a part of the corpus. get_files stores a mapping of document IDs to filenames and returns a
    list of the filenames to process
    """
    files = []
    if os.path.isdir(corpus):
        for filename in sorted(os.walk(corpus).__next__()[2]):
            docs.append(filename)
            files.append(corpus+'/'+filename)
    else:
        with open(corpus) as f:
            files = f.read().split()
            for file in files:
                docs.append(file)
    return files


def get_fname(doc_id):
    """Given a document ID as an integer, return the filename associated with the given document ID"""
    return docs[doc_id-1]


def query(item, max_docs, doc_id=True):
    """
    item: unigram or bigram to do query on
    max_docs: maximum number of documents to return
    doc_id: if true, return document IDs, otherwise return filenames

    Based on the current corpus processed, return the top max_docs documents that contain the item ordered by the tf*idf weight.
    """
    result = []
    item = item.lower().strip('.,:')
    if " " in item:
        ds = bigrams[item]
    else:
        ds = unigrams[item]
    for d in ds:
        result.append([d, math.log10(len(docs)/ds[d])])
    result.sort(reverse=True, key=lambda e: e[1])  # must be descending order to choose top max_docs
    if doc_id:
        return [r[0] for r in result][:max_docs]
    else:
        return [docs[r[0]-1] for r in result][:max_docs]


def top_unigrams(doc, num):
    """
    doc: a document ID as integer or string, or filename
    num: maximum number of unigrams to return

    Return the top unigrams in the given document, ordered by decreasing tf*idf value.
    """
    if type(doc) == str:
        if not doc.isdigit():
            doc = docs.index(doc) + 1
        else:
            doc = int(doc)
    result = []
    for term in unigrams:
        if doc in unigrams[term]:
            df = len(unigrams[term])  # number of docs term appears in
            tf = unigrams[term][doc]  # term frequency for term in doc
            tf_idf = tf*math.log10(len(docs)/df)
            result.append([term, tf_idf])
    result.sort(reverse=True, key=lambda e: e[1])  # must be descending order to choose top num
    return [r[0] for r in result][:num]


def top_bigrams(doc, num):
    """
    doc: a document ID as integer or string, or filename
    num: maximum number of bigrams to return

    Return the top bigrams in the given document, ordered by decreasing tf*idf value.
    """
    if type(doc) == str:
        if not doc.isdigit():
            doc = docs.index(doc) + 1
        else:
            doc = int(doc)
    result = []
    for term in bigrams:
        if doc in bigrams[term]:
            df = len(bigrams[term])  # number of docs term appears in
            tf = bigrams[term][doc]  # term frequency for term in doc
            tf_idf = tf*math.log10(len(docs)/df)
            result.append([term, tf_idf])
    result.sort(reverse=True, key=lambda e: e[1])  # must be descending order to choose top num
    return [r[0] for r in result][:num]


def test_all():
    """Processes multiple corpuses and tests the above functions. Results are written to a file in 'files'."""
    process_corpus('documents/1', 1)
    file = "files/HW1.testAll.1"
    file = os.path.join(dir_path, file)
    open(file, 'w').close()
    with open(file, 'a') as f:
        f.write("Processed corpus 1 given a directory path: documents/1\n")
    write_test_file(file, '1')

    process_corpus('files1.txt', 2)
    file = "files/HW1.testAll.2"
    file = os.path.join(dir_path, file)
    open(file, 'w').close()
    with open(file, 'a') as f:
        f.write("Processed corpus 1 given a file containing filenames in the local directory: files1.txt.\n")
    write_test_file(file, '1')

    process_corpus('documents/2', 3)
    file = "files/HW1.testAll.3"
    file = os.path.join(dir_path, file)
    open(file, 'w').close()
    with open(file, 'a') as f:
        f.write("Processed corpus 2 given a directory path: documents/2.\n")
    write_test_file(file, '2')

    process_corpus('files2.txt', 4)
    file = "files/HW1.testAll.4"
    file = os.path.join(dir_path, file)
    open(file, 'w').close()
    with open(file, 'a') as f:
        f.write("Processed corpus 2 given a file containing filenames in the local directory: files2.txt.\n")
    write_test_file(file, '2')


def write_test_file(file, test):
    """Helper for testAll that tests each function for a corpus and writes the results in the given file."""
    with open(file, 'a') as f:
        f.write("\nDocuments processed: ")
        f.write(str(docs) + '\n')
        f.write("\nTesting get_fname for each of the documents: \n")
        for i in range(1, len(docs)+1):
            f.write(str(i) + ": " + get_fname(i) + '\n')
        f.write("\nChoosing 10 random unigrams in corpus...\n")
        f.write("\nTesting query for unigrams in corpus " + test + " and returning doc IDs: \n")
        items = random.sample(list(unigrams.keys()), 10)
        for i in items:
            f.write(i + ": ")
            for j in query(i, 5):
                f.write(' ' + str(j))
            f.write('\n')
        f.write("\nTesting query for unigrams in corpus " + test + " and returning filenames: \n")
        for i in items:
            f.write(i + ": ")
            for j in query(i, 5, False):
                f.write(' ' + str(j))
            f.write('\n')
        f.write("\nChoosing 10 random bigrams in corpus...\n")
        f.write("\nTesting query for bigrams in corpus " + test + " and returning doc IDs: \n")
        items = random.sample(list(bigrams.keys()), 10)
        for i in items:
            f.write(i + ": ")
            for j in query(i, 5):
                f.write(' ' + str(j))
            f.write('\n')
        f.write("\nTesting query for bigrams in corpus " + test + " and returning filenames: \n")
        for i in items:
            f.write(i + ": ")
            for j in query(i, 5, False):
                f.write(' ' + str(j))
            f.write('\n')
        if test == '1':
            file = 'p4.txt'
        else:
            file = 'nasa4.txt'
        f.write("\nRetrieving top 10 unigrams in document 4 (using doc ID as integer): \n")
        for i in top_unigrams(4, 10):
            f.write(i + '\n')
        f.write("\nRetrieving top 10 unigrams in document 4 (using doc ID as string): \n")
        for i in top_unigrams('4', 10):
            f.write(i + '\n')
        f.write("\nRetrieving top 10 unigrams in document " + file + " (using filename): \n")
        for i in top_unigrams(file, 10):
            f.write(i + '\n')
        if test == '1':
            file = 'p2.txt'
        else:
            file = 'nasa2.txt'
        f.write("\nRetrieving top 10 bigrams in document 2 (using doc ID as integer): \n")
        for i in top_bigrams(2, 10):
            f.write(i + '\n')
        f.write("\nRetrieving top 10 bigrams in document 2 (using doc ID as string): \n")
        for i in top_bigrams('2', 10):
            f.write(i + '\n')
        f.write("\nRetrieving top 10 bigrams in document " + file + " (using filename): \n")
        for i in top_bigrams(file, 10):
            f.write(i + '\n')


test_all()
