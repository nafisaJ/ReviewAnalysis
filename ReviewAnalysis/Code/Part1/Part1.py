import glob;
import nltk;
import random;
import ntpath
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords;
from nltk.tokenize import RegexpTokenizer
from nltk import ne_chunk
from nltk.corpus import conll2000
from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC


from nltk.classify import ClassifierI
from statistics import mode



class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

path='../issue_sent_labeled_data/router/*.txt'

neg_word_file = open('../issue_sent_labeled_data/negative-words.txt');
pos_word_file = open('../issue_sent_labeled_data/positive-words.txt');

pos_words = pos_word_file.read();
neg_words = neg_word_file.read();

positive = pos_words.split("\n");
negative = neg_words.split("\n");


# f = open(path, 'r');
strCorpora = '';
# info = os.stat(f);        headAspect = [];
# print info.st_mtime       files=glob.glob(path);
strCorpora = '';
headAspect = [];
files=glob.glob(path);
for file in files:
    ftemp = open(file, 'r');
    filenameTemp = file.split('/')[-1];        
    headAspect.append(filenameTemp.split('.')[0]);
    strCorpora = strCorpora + ftemp.read();
'''

# info = os.stat(f);
# print info.st_mtime
fname=ntpath.basename(path);
k = fname.split('.')
headAspect=k[0]
# print(headAspect);
strCorpora = f.read();
'''
stop_words = set(stopwords.words('english'));
strSent = sent_tokenize(strCorpora);
classSent = [];
issueSent = [];
nonIssueSent = [];
# print(strSent)
for sent in strSent:
	sent = sent.replace("\n","");
	for x in range(0, 3):
		start = sent.find( '[' );
		end = sent.find( ']' );
		if start != -1 and end != -1:
			sent = sent[0:start]+sent[end+1:len(sent)]
	issueBool = sent.find("<i>");
	if issueBool > -1:
		sent = sent.replace("<i>","");
		classSent= classSent+[(sent,'I')];
		issueSent = issueSent + [(sent,'I')];
	else:
		classSent= classSent+[(sent,'NI')];
		nonIssueSent = nonIssueSent + [(sent,'NI')];


def getSuffix(word):
    for suffix in ['ing', 'ly', 'ed', 'ious', 'ies', 'ive', 'es', 's', 'ment', 'able', 'ible', 'al', 'ial', 'en', 'er', 'est', 'ful', 'ic', 'ation', 'ition', 'tion', 'ion', 'ity', 'ty', 'ative', 'itive', 'ive', 'less', 'ment', 'ness', 'eous', 'ious', 'ous', 'y']:
        if word.endswith(suffix):
            return (suffix)
    return 'AA'

def getPrefix(word):
    for prefix in ['anti','inter','de','dis','en','em','fore','in','im','il','ir','mid','mis','non','over','pre','re','semi','sub','super','trans','under','un']:
        if word.startswith(prefix):
            return (prefix)
    return 'AA'

def getPolarity(word):
    polarity = 'Neutral';
    if word.lower() in positive:
        polarity = 'Positive';
    elif word.lower() in negative:
            polarity = 'Negative';
    return polarity

def find_features(sent):
    tokenizer = RegexpTokenizer(r'\w+')
    words=tokenizer.tokenize(sent);
    # print(words)
    position=-1
    for (i, subword) in enumerate(words):
        if subword in headAspect:
            # print(i)
            position=i
    wrds_pos=nltk.pos_tag(words);
    chunkGram = """
    NP: {<NNP>*}
    {<DT>?<JJ>?<NNS>}
    {<NN><NN>}
    {<DT|PP\$>?<JJ>*<NN>}
    {<NNP>+}
    {<NN>+}
    PP: {<TO><IN>}
    VP: {<RB><MD><VB>}
    ADVP: {<RB>}
    ADJP: {<CC><RB><JJ>}
    SBAR: {<IN>}
    PRT: {<RP>}
    INTJ: {<UH>}
    PNP: {<PP><NP>}
    """;
    chunkParser= nltk.RegexpParser(chunkGram)
    chunked = chunkParser.parse(wrds_pos);
    # phrasechnktgs = nltk.chunk.util.tree2conlltags(chunked);

    

    if position!=-1:
        features = {};
        if position-4>=0:
            features["w-4"]=words[position-4]
            features["pos-4"]=wrds_pos[position-4][1]
            features["polarity-4"]=getPolarity(words[position-4])
            # features["phchnk-4"]=phrasechnktgs[position-4][2]
            # features["pre-4"]=getPrefix(words[position-4])
            # features["suf-4"]=getSuffix(words[position-4])
        else:
            features["w-4"]='X';
            features["pos-4"]='Y';
            features["polarity-4"]='Z';
            # features["phchnk-4"]='A'
            # features["pre-4"]='W'
            # features["suf-4"]='V'
        if position-3>=0:
            features["w-3"]=words[position-3]
            features["pos-3"]=wrds_pos[position-3][1]
            features["polarity-3"]=getPolarity(words[position-3])
            # features["phchnk-3"]=phrasechnktgs[position-3][2]
            # features["pre-3"]=getPrefix(words[position-3])
            # features["suf-3"]=getSuffix(words[position-3])
        else:
            features["w-3"]='X';
            features["pos-3"]='Y';
            features["polarity-3"]='Z';
            # features["phchnk-3"]='A'
            # features["pre-3"]='W'
            # features["suf-3"]='V'
        if position-2>=0:
            features["w-2"]=words[position-2]
            features["pos-2"]=wrds_pos[position-2][1]
            features["polarity-2"]=getPolarity(words[position-2])
            # features["phchnk-2"]=phrasechnktgs[position-2][2]
            # features["pre-2"]=getPrefix(words[position-2])
            # features["suf-2"]=getSuffix(words[position-2])
        else:
            features["w-2"]='X';
            features["pos-2"]='Y';
            features["polarity-2"]='Z';
            # features["phchnk-2"]='A'
            # features["pre-2"]='W'
            # features["suf-2"]='V'
        if position-1>=0:
            features["w-1"]=words[position-1]
            features["pos-1"]=wrds_pos[position-1][1]
            features["polarity-1"]=getPolarity(words[position-1])
            # features["phchnk-1"]=phrasechnktgs[position-1][2]
            # features["pre-1"]=getPrefix(words[position-1])
            # features["suf-1"]=getSuffix(words[position-1])
        else:
            features["w-1"]='X';
            features["pos-1"]='Y';
            features["polarity-1"]='Z';
            # features["phchnk-1"]='A'
            # features["pre-1"]='W'
            # features["suf-1"]='V'

        features["w"]=words[position]
        features["pos"]=wrds_pos[position][1]
        features["polarity"]=getPolarity(words[position])
        # features["phchnk"]=phrasechnktgs[position][2]
        # features["pre"]=getPrefix(words[position])
        # features["suf"]=getSuffix(words[position])

        if position+1<len(words):
            features["w+1"]=words[position+1]
            features["pos+1"]=wrds_pos[position+1][1]
            features["polarity+1"]=getPolarity(words[position+1])
            # features["phchnk+1"]=phrasechnktgs[position+1][2]
            # features["pre+1"]=getPrefix(words[position+1])
            # features["suf+1"]=getSuffix(words[position+1])
        else:
            features["w+1"]='X'
            features["pos+1"]='Y';
            features["polarity+1"]='Z';
            # features["phchnk+1"]='A'
            # features["pre+1"]='W'
            # features["suf+1"]='V'
        if position+2<len(words):
            features["w+2"]=words[position+2]
            features["pos+2"]=wrds_pos[position+2][1]
            features["polarity+2"]=getPolarity(words[position+2])
            # features["phchnk+2"]=phrasechnktgs[position+2][2]
            # features["pre+2"]=getPrefix(words[position+2])
            # features["suf+2"]=getSuffix(words[position+2])
        else:
            features["w+2"]='X'
            features["pos+2"]='Y';
            features["polarity+2"]='Z';
            # features["phchnk+2"]='A'
            # features["pre+2"]='W'
            # features["suf+2"]='V'
        if position+3<len(words):
            features["w+3"]=words[position+3]
            features["pos+3"]=wrds_pos[position+3][1]
            features["polarity+3"]=getPolarity(words[position+3])
            # features["phchnk+3"]=phrasechnktgs[position+3][2]
            # features["pre+3"]=getPrefix(words[position+3])
            # features["suf+3"]=getSuffix(words[position+3])
        else:
            features["w+3"]='X'
            features["pos+3"]='Y';
            features["polarity+3"]='Z';
            # features["phchnk+3"]='A'
            # features["pre+3"]='W'
            # features["suf+3"]='V'
        if position+4<len(words):
            features["w+4"]=words[position+4]
            features["pos+4"]=wrds_pos[position+4][1]
            features["polarity+4"]=getPolarity(words[position+4])
            # features["phchnk+4"]=phrasechnktgs[position+4][2]
            # features["pre+4"]=getPrefix(words[position+4])
            # features["suf+4"]=getSuffix(words[position+4])
        else:
            features["w+4"]='X'
            features["pos+4"]='Y';
            features["polarity+4"]='Z';
            # features["phchnk+4"]='A'
            # features["pre+4"]='W'
            # features["suf+4"]='V'

        # features["polarity-4"]
        # features["polarity-3"]
        # features["polarity-2"]
        # features["polarity-1"]
        # features["polarity"]
        # features["polarity+4"]
        # features["polarity+3"]
        # features["polarity+2"]
        # features["polarity+1"]
        # features["suf-4"]
        # features["suf-3"]
        # features["suf-2"]
        # features["suf-1"]
        # features["suf"]
        # features["suf+4"]
        # features["suf+3"]
        # features["suf+2"]
        # features["suf+1"]
     #    for w in word_features:
     #        features[w] = (w in words);
	return features;
# print(issueSent)



issueFeaturesets = [(find_features(sent), category) for (sent, category) in issueSent];
# print('Non issue start')
# print(nonIssueSent)
nonIssueSent=nonIssueSent[:len(nonIssueSent)-1]
# print(nonIssueSent)
nonIssueFeaturesets = [(find_features(rev), category) for (rev, category) in nonIssueSent];
# random.shuffle(issueFeaturesets);
# random.shuffle(nonIssueFeaturesets);
# issueBreak1 = int(len(issueFeaturesets)/3);
# nonIssueBreak1 = int(len(nonIssueFeaturesets)/3);
num_folds = 10

subset_size_issue = len(issueFeaturesets)/num_folds
subset_size_nonissue = len(nonIssueFeaturesets)/num_folds
classifieravg=0
LinearSVC_classifieravg=0
MNB_classifieravg=0
BernoulliNB_classifieravg=0
LogisticRegression_classifieravg=0
votedavg=0 
for i in range(num_folds):
    print 'Cross Validation',i+1,'/',num_folds
    testing_this_round_issue = issueFeaturesets[i*subset_size_issue:][:subset_size_issue]
    testing_this_round_nonissue = nonIssueFeaturesets[i*subset_size_nonissue:][:subset_size_nonissue]
    testing_set=testing_this_round_issue+testing_this_round_nonissue
    training_this_round_issue = issueFeaturesets[:i*subset_size_issue] + issueFeaturesets[(i+1)*subset_size_issue:]
    training_this_round_nonissue = nonIssueFeaturesets[:i*subset_size_nonissue] + nonIssueFeaturesets[(i+1)*subset_size_nonissue:]
    training_set=training_this_round_issue+training_this_round_nonissue



    # testing_set = issueFeaturesets[:issueBreak1] + nonIssueFeaturesets[:nonIssueBreak1];
    # training_set = issueFeaturesets[issueBreak1:] + nonIssueFeaturesets[nonIssueBreak1:];
    # print(training_set)
    # print('Removing')




    training_set = [x for x in training_set if x[0] is not None];
    # print(training_set)
    testing_set = [x for x in testing_set if x[0] is not None];
    classifier = nltk.NaiveBayesClassifier.train(training_set);

    print("Original Naive Bayes Algo accuracy percent using high frequency words:", (nltk.classify.accuracy(classifier, testing_set))*100);
    classifieravg=classifieravg+(nltk.classify.accuracy(classifier, testing_set))*100
    #classifier = nltk.NaiveBayesClassifier.train(testing_set);
    # print(issueSent[0][0])
    # issueFeaturesets=find_features(issueSent[0][0])
    # issueFeaturesets=find_features(issueSent(0)), category


    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(training_set)
    print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)
    MNB_classifieravg=MNB_classifieravg+(nltk.classify.accuracy(MNB_classifier, testing_set))*100

    BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
    BernoulliNB_classifier.train(training_set)
    print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)
    BernoulliNB_classifieravg=BernoulliNB_classifieravg+(nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100
    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    LogisticRegression_classifier.train(training_set)
    print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)
    LogisticRegression_classifieravg=LogisticRegression_classifieravg+(nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100

    # # less for earphone check for others
    # SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
    # SGDClassifier_classifier.train(training_set)
    # print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

    # less for earphone
    # SVC_classifier = SklearnClassifier(SVC())
    # SVC_classifier.train(training_set)
    # print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

    LinearSVC_classifier = SklearnClassifier(LinearSVC())
    LinearSVC_classifier.train(training_set)
    print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)
    LinearSVC_classifieravg=LinearSVC_classifieravg+(nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100



    voted_classifier = VoteClassifier(classifier,
                                      LinearSVC_classifier,
                                      MNB_classifier,
                                      BernoulliNB_classifier,
                                      LogisticRegression_classifier)

    print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)
    votedavg=votedavg+(nltk.classify.accuracy(voted_classifier, testing_set))*100;



print("Average Original Naive Bayes Algo accuracy percent using high frequency words:", classifieravg/num_folds);
print("Average MNB_classifier accuracy percent:", MNB_classifieravg/num_folds)
print("Average BernoulliNB_classifier accuracy percent:", BernoulliNB_classifieravg/num_folds)
print("Average LogisticRegression_classifier accuracy percent:",LogisticRegression_classifieravg/num_folds)
print("Average LinearSVC_classifier accuracy percent:", LinearSVC_classifieravg/num_folds)
print("Average voted_classifier accuracy percent:", votedavg/num_folds)


#mouse error
# NuSVC_classifier = SklearnClassifier(NuSVC())
# NuSVC_classifier.train(training_set)
# print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

