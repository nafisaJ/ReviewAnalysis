import glob;
import nltk;
import random;
from nltk import ngrams;
from nltk.corpus import stopwords;
from nltk import Tree;
from nltk.corpus import brown
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.corpus import conll2000
from nltk.corpus import sentiwordnet as swn


neg_word_file = open('../issue_sent_labeled_data/negative-words.txt');
pos_word_file = open('../issue_sent_labeled_data/positive-words.txt');

pos_words = pos_word_file.read();
neg_words = neg_word_file.read();

positive = pos_words.split("\n");
negative = neg_words.split("\n");



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

def getPOSFeatures(sent):
    sent = sent.replace("<i>","");
    tagged = [];
    words = nltk.word_tokenize(sent);
    #print(nltk.pos_tag(words));
    tagged.append(nltk.pos_tag(words));
    return tagged




path='../issue_sent_labeled_data/earphone/*.txt';
strCorpora = '';
headAspect = [];
files=glob.glob(path);



for file in files:
        ftemp = open(file, 'r');
        filenameTemp = file.split('\\')[1];        
        headAspect.append(filenameTemp.split('.')[0]);
        strCorpora = strCorpora + ftemp.read();

issueSent = '';
nonIssueSent = '';

stop_words = set(stopwords.words('english'));


from nltk.tokenize import sent_tokenize, word_tokenize


strSent = sent_tokenize(strCorpora);
ppF = [];

classSent = [];

 
all_pos_words = [];
processedStr = '';

outputFileName = 'trainEarphone.txt';
f1 = open(outputFileName, 'w');

def duplicates(lst, item1, item2, item3):
    start = -1;
    end = -1;
    for i in range(0,len(lst)-1):
        if lst[i] == item1 and lst[i+1] == item2 and lst[i+2] == item3:
            start = i;
            break;
    for j in range(start+1,len(lst)-1):
        if lst[j] == item1 and lst[j+1] == item2 and lst[j+2] == item3:
            end = j-3;
            break;
    return [start,end];

for sent in strSent:
    flat_tree = '';
    sent = sent.replace("\n","");
    
    for x in range(0, 3):
        start = sent.find( '[' );
        end = sent.find( ']' );
       
        if start != -1 and end != -1:
          sent = sent[0:start]+sent[end+1:len(sent)]
    issueBool =  sent.find("<i>");
    issueStartEnd = [];
    words = nltk.word_tokenize(sent);
    if issueBool > -1:
        
        issueStartEnd = duplicates(words,'<','i','>');
        
    sent = sent.replace("<i>","");
    temp = getPOSFeatures(sent);
    words = nltk.word_tokenize(sent);
    
    tagged = nltk.pos_tag(words)
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
    """;
    
    chunkParser = nltk.RegexpParser(chunkGram);
    chunked = chunkParser.parse(tagged);
    
    temp = nltk.chunk.util.tree2conllstr(chunked);
    tupArr = temp.split("\n");
    issuePos = 'O';
    indexIssue = 0;
    for tup in tupArr:
        wordTemp = tup.split(" ")[0];
        if wordTemp!='ï':
            
            suffix = getSuffix(wordTemp);
            prefix = getPrefix(wordTemp);
            polarity = 'Neutral';

           
            if wordTemp.lower() in positive:
                polarity = 'Positive';
            elif wordTemp.lower() in negative:
                polarity = 'Negative';

            
            if issueBool > -1:
                if indexIssue == issueStartEnd[0]:
                    issuePos = 'B';
                elif indexIssue > issueStartEnd[0] and indexIssue < issueStartEnd[1]:
                    issuePos = 'I';
                else:
                    issuePos = 'O';
                    
                
            f1.write(str(tup) + ' ' + suffix + ' ' + prefix + ' ' + polarity + ' ' + issuePos + '\n');       
            indexIssue = indexIssue+1;
    f1.write('\n');
       

f1.close();


with open(outputFileName, 'r') as fin:
    data = fin.read().splitlines(True)
with open(outputFileName, 'w') as fout:
    fout.writelines(data[:len(data)-2])
   
print(outputFileName + ' generated to be used input for CRF Suite');
