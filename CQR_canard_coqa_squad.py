import numpy as np 
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, BertTokenizer 
from transformers import BertForQuestionAnswering
from tqdm.auto import tqdm 

from numpy import dot
from numpy.linalg import norm
from transformers import BertTokenizer, BertModel

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

import nltk
nltk.download('averaged_perceptron_tagger')




#Squad Training

from datasets import load_dataset
raw = load_dataset('quac')
raw['train'][2]

raw['train'][1]['answers']['texts'][7]
len(raw['train'][1]['answers']['texts'])
tokenizer_ajai = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")



def add_end_idx(answers, context):
    answer = []
    for answer, context in tqdm(zip(answers, context)):
        for j in range(len(answer['texts'])):
            gold_text = answer['texts'][j]
            start_idx = answer['answer_starts'][j]
            answer['answer_end'][j]= start_idx + len(gold_text))
    return answer

def prep_data(dataset):
    questions = dataset['questions']
    contexts = dataset['context']
    answers = add_end_idx(
        dataset['answers'],
        contexts
    )
    return {
        'question': questions,
        'context': contexts,
        'answers': answers
    }



dataset = prep_data(raw['train'][:10000])

train = tokenizer_ajai(dataset['context'], dataset['question'],
                  truncation=True, padding='max_length',
                  max_length=512, return_tensors='pt')


def add_token_positions(encodings, answers):
    # initialize lists to contain the token indices of answer start/end
    start_positions = []
    end_positions = []
    for i in tqdm(range(len(answers))):
        # append start/end token position using char_to_token method
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer_ajai.model_max_length
        # end position cannot be found, char_to_token found space, so shift position until found
        shift = 1
        while end_positions[-1] is None:
            end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] - shift)
            shift += 1
    # update our encodings object with the new token-based start/end positions
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

    add_token_positions(train, dataset['answers'])

    train.keys()

    train['start_positions'][:5], train['end_positions'][:5]

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

# build datasets for both our training data
train_dataset = SquadDataset(train)

loader = torch.utils.data.DataLoader(train_dataset,
                                     batch_size=16,
                                     shuffle=True)

m = BertForQuestionAnswering.from_pretrained('/Users/ajai_devanathan/Desktop/project_ir/covidbert_last_layer_training/')

from transformers import AdamW

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
m.to(device)
m.train()
optim = AdamW(m.parameters(), lr=5e-5)

for epoch in range(1):
    loop = tqdm(loader)
    for batch in loop:
        optim.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)

        outputs = m(input_ids, attention_mask=attention_mask,
                        start_positions=start_positions,
                        end_positions=end_positions)
        
        loss = outputs[0]
        loss.backward()
        optim.step()

        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())


m.save_pretrained('/Users/ajai_devanathan/Desktop/project_ir/covbert_squad_10000/')

# saving the first step

covbert_squad_10000=BertForQuestionAnswering.from_pretrained('/Users/ajai_devanathan/Desktop/project_ir/covbert_squad_10000/')


#paragraphs for testing the bot

paragraph1 = '''Coronavirus disease (COVID-19) is an infectious disease caused by the SARS-CoV-2 virus.Most people infected with the virus will experience mild to moderate respiratory illness and recover without requiring special treatment. However, some will become seriously ill and require medical attention. Older people and those with underlying medical conditions like cardiovascular disease, diabetes, chronic respiratory disease, or cancer are more likely to develop serious illness. Anyone can get sick with COVID-19 and become seriously ill or die at any age. The best way to prevent and slow down transmission is to be well informed about the disease and how the virus spreads. Protect yourself and others from infection by staying at least 1 metre apart from others, wearing a properly fitted mask, and washing your hands or using an alcohol-based rub frequently. Get vaccinated when it’s your turn and follow local guidance.The virus can spread from an infected person’s mouth or nose in small liquid particles when they cough, sneeze, speak, sing or breathe. These particles range from larger respiratory droplets to smaller aerosols. It is important to practice respiratory etiquette, for example by coughing into a flexed elbow, and to stay home and self-isolate until you recover if you feel unwell.To prevent infection and to slow transmission of COVID-19, do the following: Get vaccinated when a vaccine is available to you.Stay at least 1 metre apart from others, even if they don’t appear to be sick.Wear a properly fitted mask when physical distancing is not possible or when in poorly ventilated settings.Choose open, well-ventilated spaces over closed ones. Open a window if indoors.Wash your hands regularly with soap and water. clean them with alcohol-based hand rub.Cover your mouth and nose when coughing or sneezing.If you feel unwell, stay home and self-isolate until you recover.'''


paragraph2 = '''Coronavirus disease (COVID-19) is an infectious disease caused by the SARS-CoV-2 virus.Most people infected with the virus will experience mild to moderate respiratory illness and recover without requiring special treatment. However, some will become seriously ill and require medical attention. Older people and those with underlying medical conditions like cardiovascular disease, diabetes, chronic respiratory disease, or cancer are more likely to develop serious illness. Anyone can get sick with COVID-19 and become seriously ill or die at any age. The best way to prevent and slow down transmission is to be well informed about the disease and how the virus spreads. Protect yourself and others from infection by staying at least 1 metre apart from others, wearing a properly fitted mask, and washing your hands or using an alcohol-based rub frequently. Get vaccinated when it’s your turn and follow local guidance.The virus can spread from an infected person’s mouth or nose in small liquid particles when they cough, sneeze, speak, sing or breathe. These particles range from larger respiratory droplets to smaller aerosols. It is important to practice respiratory etiquette, for example by coughing into a flexed elbow, and to stay home and self-isolate until you recover if you feel unwell.To prevent infection and to slow transmission of COVID-19, do the following: Get vaccinated when a vaccine is available to you.Stay at least 1 metre apart from others, even if they don’t appear to be sick.Wear a properly fitted mask when physical distancing is not possible or when in poorly ventilated settings.Choose open, well-ventilated spaces over closed ones. Open a window if indoors.Wash your hands regularly with soap and water. clean them with alcohol-based hand rub.Cover your mouth and nose when coughing or sneezing.If you feel unwell, stay home and self-isolate until you recover.'''

paragraph = '''Coronavirus disease (COVID-19) is an infectious disease caused by the SARS-CoV-2 virus.Most people infected with the virus will experience mild to moderate respiratory illness and recover without requiring special treatment. However, some will become seriously ill and require medical attention. Older people and those with underlying medical conditions like cardiovascular disease, diabetes, chronic respiratory disease, or cancer are more likely to develop serious illness. Anyone can get sick with COVID-19 and become seriously ill or die at any age. The best way to prevent and slow down transmission is to be well informed about the disease and how the virus spreads. Protect yourself and others from infection by staying at least 1 metre apart from others, wearing a properly fitted mask, and washing your hands or using an alcohol-based rub frequently. Get vaccinated when it’s your turn and follow local guidance.The virus can spread from an infected person’s mouth or nose in small liquid particles when they cough, sneeze, speak, sing or breathe. These particles range from larger respiratory droplets to smaller aerosols. It is important to practice respiratory etiquette, for example by coughing into a flexed elbow, and to stay home and self-isolate until you recover if you feel unwell.To prevent infection and to slow transmission of COVID-19, do the following: Get vaccinated when a vaccine is available to you.Stay at least 1 metre apart from others, even if they don’t appear to be sick.Wear a properly fitted mask when physical distancing is not possible or when in poorly ventilated settings.Choose open, well-ventilated spaces over closed ones. Open a window if indoors.Wash your hands regularly with soap and water. clean them with alcohol-based hand rub.Cover your mouth and nose when coughing or sneezing.If you feel unwell, stay home and self-isolate until you recover.'''

# THE NEXT STEP 10,000-20,000

dataset1 = prep_data(raw_datasets['train'][10001:20000])
train1 = tokenizer_ajai(dataset1['context'], dataset1['question'],
                  truncation=True, padding='max_length',
                  max_length=512, return_tensors='pt')
train_dataset1 = SquadDataset(train1)
loader1 = torch.utils.data.DataLoader(train_dataset1,
                                     batch_size=16,
                                     shuffle=True)
m1 = BertForQuestionAnswering.from_pretrained('/Users/ajai_devanathan/Desktop/project_ir/covbert_squad_10000')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
m1.to(device)
m1.train()
optim = AdamW(m.parameters(), lr=5e-5)

for epoch in range(1):
    loop = tqdm(loader)
    for batch in loop:
        optim.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)

        outputs = m1(input_ids, attention_mask=attention_mask,
                        start_positions=start_positions,
                        end_positions=end_positions)
        
        loss = outputs[0]
        loss.backward()
        optim.step()

        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())


m1.save_pretrained('/Users/ajai_devanathan/Desktop/project_ir/covbert_squad_10000-20000/')
covbert_squad_10000_20000=BertForQuestionAnswering.from_pretrained('/Users/ajai_devanathan/Desktop/project_ir/covbert_squad_10000-20000/')

#This function will use the above run model

def queryme(text,question):
    encoding = tokenizer_ajai.encode_plus(text=query,text_pair=text)
    inputs = encoding['input_ids']  #Token embeddings
    sentence_embedding = encoding['token_type_ids']  #Segment embeddings
    tokens = tokenizer_ajai.convert_ids_to_tokens(inputs)
    start_scores = covbert_squad_10000_20000(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))[0]
    end_scores =  covbert_squad_10000_20000(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))[1]
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)
    answer = ' '.join(tokens[start_index:end_index+1])
    corrected_answer = ''

    for word in answer.split():
     
    #If it's a subword token
        if word[0:2] == '##':
            corrected_answer += word[2:]
        else:
            corrected_answer += ' ' + word

    return corrected_answer

#Test run

text = 'corona virus is a infectious disease. It can kill anybody. Mostly old peope die because of this. please wash hands with soap and water'
query = 'hands can be should be washed with what?'
queryme(text,query)

query = 'who die in covid?'
queryme(text,query)

text = 'Coronavirus disease (COVID-19) is an infectious disease caused by the SARS-CoV-2 virus.Most people infected with the virus will experience mild to moderate respiratory illness and recover without requiring special treatment. However, some will become seriously ill and require medical attention. Older people and those with underlying medical conditions like cardiovascular disease, diabetes, chronic respiratory disease, or cancer are more likely to develop serious illness. Anyone can get sick with COVID-19 and become seriously ill or die at any age. The best way to prevent and slow down transmission is to be well informed about the disease and how the virus spreads. Protect yourself and others from infection by staying at least 1 metre apart from others, wearing a properly fitted mask, and washing your hands or using an alcohol-based rub frequently. Get vaccinated when it’s your turn and follow local guidance.The virus can spread from an infected person’s mouth or nose in small liquid particles when they cough, sneeze, speak, sing or breathe. These particles range from larger respiratory droplets to smaller aerosols. It is important to practice respiratory etiquette, for example by coughing into a flexed elbow, and to stay home and self-isolate until you recover if you feel unwell.To prevent infection and to slow transmission of COVID-19, do the following: Get vaccinated when a vaccine is available to you.Stay at least 1 metre apart from others, even if they don’t appear to be sick.Wear a properly fitted mask when physical distancing is not possible or when in poorly ventilated settings.Choose open, well-ventilated spaces over closed ones. Open a window if indoors.Wash your hands regularly with soap and water. clean them with alcohol-based hand rub.Cover your mouth and nose when coughing or sneezing.If you feel unwell, stay home and self-isolate until you recover.'
query = 'hands can be washed with what?'
queryme(text,query)

query = 'what will people experience?'
queryme(text,query)

query = 'what age can we die?'

text = "The Coronavirus (CoV) is a large family of viruses known to cause illnesses ranging from the common cold to acute respiratory tract infection. The severity of the infection may be visible as pneumonia, acute respiratory syndrome, and even death. Until the outbreak of SARS, this group of viruses was greatly overlooked. However, since the SARS and MERS outbreaks, these viruses have been studied in greater detail, propelling the vaccine research. On December 31, 2019, mysterious cases of pneumonia were detected in the city of Wuhan in China's Hubei Province. On January 7, 2020, the causative agent was identified as a new coronavirus (2019-nCoV), and the disease was later named as COVID-19 by the WHO. The virus spread extensively in the Wuhan region of China and has gained entry to over 210 countries and territories."
query = 'where was pneumonia detected?'
queryme(text,query)


text ="Though experts suspected that the virus is transmitted from animals to humans, there are mixed reports on the origin of the virus. There are no treatment options available for the virus as such, limited to the use of anti-HIV drugs and/or other antivirals such as Remdesivir and Galidesivir. For the containment of the virus, it is recommended to quarantine the infected and to follow good hygiene practices. The virus has had a significant socio-economic impact globally. Economically, China is likely to experience a greater setback than other countries from the pandemic due to added trade war pressure, which have been discussed in this paper."
query = 'is there any treatment option?'
queryme(text,query)


text ="Coronaviridae is a family of viruses with a positive-sense RNA that possess an outer viral coat. When looked at with the help of an electron microscope, there appears to be a unique corona around it. This family of viruses mainly cause respiratory diseases in humans, in the forms of common cold or pneumonia as well as respiratory infections. These viruses can infect animals as well (1, 2). Up until the year 2003, coronavirus (CoV) had attracted limited interest from researchers. However, after the SARS (severe acute respiratory syndrome) outbreak caused by the SARS-CoV, the coronavirus was looked at with renewed interest (3, 4). This also happened to be the first epidemic of the 21st century originating in the Guangdong province of China. Almost 10 years later, there was a MERS (Middle East respiratory syndrome) outbreak in 2012, which was caused by the MERS-CoV (5, 6). Both SARS and MERS have a zoonotic origin and originated from bats. A unique feature of these viruses is the ability to mutate rapidly and adapt to a new host. The zoonotic origin of these viruses allows them to jump from host to host. Coronaviruses are known to use the angiotensin-converting enzyme-2 (ACE-2) receptor or the dipeptidyl peptidase IV (DPP-4) protein to gain entry into cells for replication (7–10).In December 2019, almost seven years after the MERS 2012 outbreak, a novel Coronavirus (2019-nCoV) surfaced in Wuhan in the Hubei region of China. The outbreak rapidly grew and spread to neighboring countries. However, rapid communication of information and the increasing scale of events led to quick quarantine and screening of travelers, thus containing the spread of the infection. The major part of the infection was restricted to China, and a second cluster was found on a cruise ship called the Diamond Princess docked in Japan (11, 12)."
query = 'what is the origin of the virus?'
queryme(text,query)

query = 'what did we find in the electron microscope?'
queryme(text,query)


text ="Coronaviridae is a family of viruses with a positive-sense RNA that possess an outer viral coat, The german variant can kill young people.When looked at with the help of an electron microscope, there appears to be a unique corona around it. This family of viruses mainly cause respiratory diseases in humans, in the forms of common cold or pneumonia as well as respiratory infections. These viruses can infect animals as well (1, 2). Up until the year 2003, coronavirus (CoV) had attracted limited interest from researchers. However, after the SARS (severe acute respiratory syndrome) outbreak caused by the SARS-CoV, the coronavirus was looked at with renewed interest (3, 4). This also happened to be the first epidemic of the 21st century originating in the Guangdong province of China. Almost 10 years later, there was a MERS (Middle East respiratory syndrome) outbreak in 2012, which was caused by the MERS-CoV (5, 6). Both SARS and MERS have a zoonotic origin and originated from bats. A unique feature of these viruses is the ability to mutate rapidly and adapt to a new host. The zoonotic origin of these viruses allows them to jump from host to host. Coronaviruses are known to use the angiotensin-converting enzyme-2 (ACE-2) receptor or the dipeptidyl peptidase IV (DPP-4) protein to gain entry into cells for replication (7–10).In December 2019, almost seven years after the MERS 2012 outbreak, a novel Coronavirus (2019-nCoV) surfaced in Wuhan in the Hubei region of China. The outbreak rapidly grew and spread to neighboring countries. However, rapid communication of information and the increasing scale of events led to quick quarantine and screening of travelers, thus containing the spread of the infection. The major part of the infection was restricted to China, and a second cluster was found on a cruise ship called the Diamond Princess docked in Japan (11, 12)."
query='what can the german variant do'
queryme(text,query)

######Topics Extraction

example=['what are the symptoms of covid virus', 'what should we do to protect from this virus','how many countries are infected with this', 'where did this originate']
tfidf = TfidfVectorizer()
svd = TruncatedSVD(n_components = 4)
pipeline = Pipeline([
    ('tfidf', tfidf),
    ('svd', svd)
])
X_lsa = pipeline.fit_transform(example)
tfidf = pipeline.named_steps['tfidf']
vocab = tfidf.get_feature_names()
df=pd.DataFrame(svd.components_, index=['Topic1','Topic2','Topic3','Topic4'],columns = vocab)

terms = tfidf.get_feature_names()
for i, comp in enumerate(svd.components_):
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:7]
    print("\nTopic "+str(i)+": ")
    for t in sorted_terms:
        print(t[0], end =" ")

df1=pd.DataFrame(X_lsa, columns=['Topic0','Topic1','Topic2','Topic3'],index = example)

X_normed = normalize(X_lsa, axis = 1)
similarity = X_normed @ X_normed.T
df2=pd.DataFrame(similarity, index = example, columns = example)

df2.T @ df2
####

#Historical Embeddings

# QUAC model trained by us oin COLAB after converting the QUAC type to SQUAD with only 1 epoch , 10**-5 learning rate and only 5000 samples

model_quac = AutoModelForQuestionAnswering.from_pretrained('/Users/ajai_devanathan/Desktop/project_ir/quac_covbert_5000')

#A similar Hugging face model , open source available which is trained for QUAC converted to SQUAD type

tokenizer = AutoTokenizer.from_pretrained("ixa-ehu/SciBERT-SQuAD-QuAC")
model_ = AutoModelForQuestionAnswering.from_pretrained("ixa-ehu/SciBERT-SQuAD-QuAC")

# general huffing face for SQUAD only

model = AutoModelForQuestionAnswering.from_pretrained("dmis-lab/biobert-base-cased-v1.1-squad")

#only SQUAD generic from Hugging face , gives mostly factoid answers
def queryme1(text,question):
    encoding = tokenizer_ajai.encode_plus(text=question,text_pair=text)
    inputs = encoding['input_ids']  #Token embeddings
    sentence_embedding = encoding['token_type_ids']  #Segment embeddings
    tokens = tokenizer_ajai.convert_ids_to_tokens(inputs)
    start_scores = model(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))[0]
    end_scores =  model(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))[1]
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)
    answer = ' '.join(tokens[start_index:end_index+1])
    corrected_answer = ''

    for word in answer.split():
     
    #If it's a subword token
        if word[0:2] == '##':
            corrected_answer += word[2:]
        else:
            corrected_answer += ' ' + word

    return corrected_answer

# testing by adding the history back to the context

text = 'The novel corona virus originated in the city of wuhan,china.It was reported that intially over a 1000 people were infected with the virus which started from a  small market in that city. It is claimed that the virus was transmitted through bats.The WHO sounded an alarm after 2 months of the report of the fist case of the infection.Meanwhile, the infection had spread to more than 30 countries.The main symptom of the virus remains to be sore throat, headache and fever. The only way to save yourself is by regularly washing hands with soap and water and wearing a mask always. Today more than 140 countries have been infected with this virus'
query = 'what are the symptoms of the virus'
a = queryme1(text,query)
a 


text = text + query  + a 
text 

query = 'how to save yourself?'
b = queryme1(text,query)
b


text = text + query +  b 
text

query = 'how many countries have been infected by it?'
c = queryme1(text,query)
c

text = text + query +  c  

query = 'where did it originate?'
d = queryme1(text,query)
d 
text = text +query + d 

query = 'When did WHO sound an alarm about it?'
queryme1(text,query)

query = ' How did this transmit'
queryme1(text,query)
text 

query= ' does god exist'
queryme1(text,query)

################################################################
##Getting the cosine similarity between previous queries 

model_sim =   BertModel.from_pretrained('dmis-lab/biobert-v1.1') 
def get_bert_similarity(sentence_pairs):
    df = pd.DataFrame(columns=['queries','cosine similarity'])
    queries=[]
    cosine = []
    for index in range(len(sentence_pairs)-1):
        inputs_1 = tokenizer_ajai(sentence_pairs[index], return_tensors='pt')
        inputs_2 = tokenizer_ajai(sentence_pairs[-1], return_tensors='pt')
        sent_1_embed = model_sim(**inputs_1).last_hidden_state[0][0].detach().numpy()
        sent_2_embed = model_sim(**inputs_2).last_hidden_state[0][0].detach().numpy()
        distance = dot(sent_1_embed, sent_2_embed)/(norm(sent_1_embed)* norm(sent_2_embed)) # computes the average of all the tokens' last_hidden_state
        queries.append(sentence_pairs[index] + '  AND  ' + sentence_pairs[-1])
        cosine.append(distance)
    df['queries'] = queries
    df['cosine similarity'] = cosine
    return df

sentence_pairs = ['What are the symptoms of the corona virus', 
                                 'how to save yourself','how many countries have been infected by it',
                                 'where did it originate','When did WHO sound an alarmn about it']


get_bert_similarity(sentence_pairs)
    

def seek_info(t,q):
    a = queryme1(t,query)
    print(a)
    t = t + query + a
    q = input('Ask another one ? or Type 0 to Exit')
    b = queryme1(t,q)
    print(b)
    #text1 = text1 + q1 +answer1
    #return answer1
seek_info(text)
text 


################################################################

# This query uses generic QUAC converted to SQUAD from HFace

def queryme2(text,question):
    encoding = tokenizer.encode_plus(text=question,text_pair=text)
    inputs = encoding['input_ids']  #Token embeddings
    sentence_embedding = encoding['token_type_ids']  #Segment embeddings
    tokens = tokenizer.convert_ids_to_tokens(inputs)
    start_scores = model_(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))[0]
    end_scores =  model_(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))[1]
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)
    answer = ' '.join(tokens[start_index:end_index+1])
    corrected_answer = ''

    for word in answer.split():
     
    #If it's a subword token
        if word[0:2] == '##':
            corrected_answer += word[2:]
        else:
            corrected_answer += ' ' + word

    return corrected_answer

#Testing the bot

text = 'Medical News Today is a web-based outlet for medical information and news, targeted at both the general public and physicians. All posted content is available online (>250,000 articles as of January 2014), and the earliest available article dates from May 2003. The website has been owned by Healthline Media since 2016.The business office for the site is located in Brighton, East Sussex, and a second office is maintained near Manchester. As of September 2019, it was the third most visited health site in the United States. As of October 2019, it had a global ranking by Alexa of 869 and a United States ranking of 440'
query = 'What is Medical News Today'
a = queryme2(text,query)
a 


text = text + query  + a 

query = 'what is the earliest available article date?'
b = queryme2(text,query)
b


text = text + query +  b 


query = 'where is its office located?'

c = queryme2(text,query)
c 

text = text + query +  c  

query = 'who is the owner of this?'
d = queryme2(text,query)
d 
text = text +query + d 

query = 'How often is it visited?'
e = queryme1(text,query)
e 
text = text + query +e 


query = ' what is the ranking'
queryme2(text,query)
text 

query= ' does god exist'
queryme2(text,query)

def find_noun(q):
    s =  nltk.sent_tokenize(q)
    s = [nltk.word_tokenize(i) for i in s]
    s = [nltk.pos_tag(i) for i in s]
    s = s[0]
    noun =''
    for j in range(len(s)):
        if s[j][1] in ['NN', 'NNP']:
            noun = noun + ' ' + s[j][0]
    return noun 

def find_PRP(q):
    s =  nltk.sent_tokenize(q)
    s = [nltk.word_tokenize(i) for i in s]
    s = [nltk.pos_tag(i) for i in s]
    s = s[0]
    PRP =''
    for j in range(len(s)):
        if s[j][1] in ['PRP', 'DT']:
            PRP = PRP + ' ' + s[j][0]
        PRP = PRP.replace('the','')
    return PRP

find_noun('what is the news today')

text = 'Medical News Today is a web-based outlet for medical information and news, targeted at both the general public and physicians. All posted content is available online (>250,000 articles as of January 2014), and the earliest available article dates from May 2003. The website has been owned by Healthline Media since 2016.The business office for the site is located in Brighton, East Sussex, and a second office is maintained near Manchester. As of September 2019, it was the third most visited health site in the United States.[2] As of October 2019, it had a global ranking by Alexa of 869 and a United States ranking of 440 ||| what is medical news today ||| what is the earliest article date ||| where is the office located || who is the owner of this ||| how often is it visited'
query1 = 'What is Medical News Today'
a = queryme2(text,query1)
a 


query2 = 'what is the earliest available article date?'
b = queryme2(text,query2)
b



query3 = 'where is the office located?'
query3 = query3.replace(find_PRP(query3), find_noun(query1))
c = queryme2(text,query3)
c 
query3 = 'where is the Medical News Today office located'
 

query4 = 'who is the owner of this?'
query4 = query4.replace(find_PRP(query4), find_noun(query3))
d = queryme2(text,query4)
d 

query5 = 'How often is it visited?'
query5 = query5.replace(find_PRP(query5), find_noun(query4))
e = queryme1(text,query5)
if e == '[CLS]':
    query5 = query5.replace(find_PRP(query5), find_noun(query3))
query5 
e 
text = text + query +e 


query6 = ' what is its ranking'
queryme1(text,query6)

query= ' ''
queryme2(text,query)

# 3rd experiment ( span for quac is higher ,no results are more and contexts work better)
# Squad will always give an answer right or wrong

text = 'Narendra Damodardas Modi born 17 September 1950) is an Indian politician serving as the 14th and current prime minister of India since 2014. Modi was the chief minister of Gujarat from 2001 to 2014 and is the Member of Parliament from Varanasi. He is a member of the Bharatiya Janata Party (BJP) and of the Rashtriya Swayamsevak Sangh (RSS), a right-wing Hindu nationalist paramilitary volunteer organisation. He is the longest serving prime minister from outside the Indian National Congress.Modi was born and raised in Vadnagar in northeastern Gujarat, where he completed his secondary education. He was introduced to the RSS at age eight. He has reminisced about helping out after school at his fathers tea stall at the Vadnagar railway station. At age 18, Modi was married to Jashodaben Chimanlal Modi, whom he abandoned soon after. He first publicly acknowledged her as his wife more than four decades later when required to do so by Indian law, but has made no contact with her since. Modi has asserted he had travelled in northern India for two years after leaving his parental home, visiting a number of religious centres, but few details of his travels have emerged. Upon his return to Gujarat in 1971, he became a full-time worker for the RSS. After the state of emergency was declared by prime minister Indira Gandhi in 1975, Modi went into hiding. The RSS assigned him to the BJP in 1985 and he held several positions within the party hierarchy until 2001, rising to the rank of general secretary.'





query1 = ' when was modi born'
a = queryme2(text,query1)
a 


query2 = 'when was he chief minister'
b = queryme2(text,query2)
b



query3 = 'what was his wifes name?'
c = queryme2(text,query3)
c 

query4 = 'when did he  return to gujrat'
d = queryme2(text,query4)
d 

query5 ='who declared emergency'
e = queryme1(text,query5)
e 


query6 = ' where was she born'
f = queryme2(text,query6)

f 

query= ''
queryme2(text,query)

#  SQUAD Trained on QUAC and then on CoQA datasets

model_coqa = AutoModelForQuestionAnswering.from_pretrained('/Users/ajai_devanathan/Desktop/project_ir/squad_quac_coqa')
tokenizer_coqa = AutoTokenizer.from_pretrained("ixa-ehu/SciBERT-SQuAD-QuAC")

from transformers import AutoTokenizer, AutoModelForQuestionAnswering

tokenizer_coqa = AutoTokenizer.from_pretrained("peggyhuang/SciBERT-CoQA")

model_coqa = AutoModelForQuestionAnswering.from_pretrained("peggyhuang/SciBERT-CoQA")

def queryme3(text,question):
    encoding = tokenizer_coqa.encode_plus(text=question,text_pair=text)
    inputs = encoding['input_ids']  #Token embeddings
    sentence_embedding = encoding['token_type_ids']  #Segment embeddings
    tokens = tokenizer_coqa.convert_ids_to_tokens(inputs)
    start_scores = model_coqa(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))[0]
    end_scores =  model_coqa(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))[1]
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)
    answer = ' '.join(tokens[start_index:end_index+1])
    corrected_answer = ''

    for word in answer.split():
     
    #If it's a subword token
        if word[0:2] == '##':
            corrected_answer += word[2:]
        else:
            corrected_answer += ' ' + word

    return corrected_answer

text = 'Narendra Damodardas Modi born 17 September 1950) is an Indian politician serving as the 14th and current prime minister of India since 2014. Modi was the chief minister of Gujarat from 2001 to 2014 and is the Member of Parliament from Varanasi. He is a member of the Bharatiya Janata Party (BJP) and of the Rashtriya Swayamsevak Sangh (RSS), a right-wing Hindu nationalist paramilitary volunteer organisation. He is the longest serving prime minister from outside the Indian National Congress.Modi was born and raised in Vadnagar in northeastern Gujarat, where he completed his secondary education. He was introduced to the RSS at age eight. He has reminisced about helping out after school at his fathers tea stall at the Vadnagar railway station. At age 18, Modi was married to Jashodaben Chimanlal Modi, whom he abandoned soon after. He first publicly acknowledged her as his wife more than four decades later when required to do so by Indian law, but has made no contact with her since. Modi has asserted he had travelled in northern India for two years after leaving his parental home, visiting a number of religious centres, but few details of his travels have emerged. Upon his return to Gujarat in 1971, he became a full-time worker for the RSS. After the state of emergency was declared by prime minister Indira Gandhi in 1975, Modi went into hiding. The RSS assigned him to the BJP in 1985 and he held several positions within the party hierarchy until 2001, rising to the rank of general secretary.'



query1 = ' when was modi born'
a = queryme3(text,query1)
a 

text = text + query1 + a

query2 = 'when was he chief minister'
b = queryme3(text,query2)
b

text = text + query2 + b


query3 = 'what was his wifes name?'
c = queryme3(text,query3)
c 

text = text + query + c

query4 = 'when did he  return to gujrat'
d = queryme3(text,query4)
d 

query5 ='who declared emergency'
e = queryme3(text,query5)
e 


query6 = ' where was she born'
f = queryme3(text,query6)

f 

query= ''
queryme3(text,query)


text ='In late December 2019, an outbreak of a mysterious pneumonia characterized by fever, dry cough, and fatigue, and occasional gastrointestinal symptoms happened in a seafood wholesale wet market, the Huanan Seafood Wholesale Market, in Wuhan, Hubei, China.1 The initial outbreak was reported in the market in December 2019 and involved about 66% of the staff there. The market was shut down on January 1, 2020, after the announcement of an epidemiologic alert by the local health authority on December 31, 2019. However, in the following month (January) thousands of people in China, including many provinces (such as Hubei, Zhejiang, Guangdong, Henan, Hunan, etc.) and cities (Beijing and Shanghai) were attacked by the rampant spreading of the disease.Furthermore, the disease traveled to other countries, such as Thailand, Japan, Republic of Korea, Viet Nam, Germany, United States, and Singapore. The first case reported in our country was on January 21, 2019. As of February 6, 2020, a total of 28,276 confirmed cases with 565 deaths globally were documented by WHO, involving at least 25 countries.The pathogen of the outbreak was later identified as a novel beta-coronavirus, named 2019 novel coronavirus (2019-nCoV) and recalled to our mind the terrible memory of the severe acute respiratory syndrome (SARS-2003, caused by another beta-coronavirus) that occurred 17 years ago.In 2003, a new coronavirus, the etiology of a mysterious pneumonia, also originated from southeast China, especially Guangdong province, and was named SARS coronavirus that fulfilled the Koch’s postulate.The mortality rate caused by the virus was around 10%–15%.Through the years, the medical facilities have been improved; nevertheless, no proper treatment or vaccine is available for the SARS.The emergence of another outbreak in 2012 of novel coronavirus in Middle East shared similar features with the outbreak in 2003.Both were caused by coronavirus but the intermediate host for MERS is thought to be the dromedary camel and the mortality can be up to 37%.5'

query1 = 'what is SARS'
a = queryme3(text,query1)
a 
queryme1(text,query1)
queryme2(text,query1)
queryme3(text,query1)
queryme4(text,query1) 
text = text +query1+a

query2 = 'what are its symptoms'
b = queryme3(text,query2)
b
queryme1(text,query2)
queryme2(text,query2)
queryme3(text,query2)
queryme4(text,query2) 
text = text + query2 + b

query3 = 'what is its origin'
c = queryme3(text,query3)
c
queryme1(text,query3)
queryme2(text,query3)
queryme3(text,query3)
queryme4(text,query3)
text = text + query2 +c

query4 = 'where did it spread'
d = queryme3(text,query4)
d
queryme1(text,query4)
queryme2(text,query4)
queryme3(text,query4)
queryme4(text,query4)
text = text + query4 + d

query5 = 'what was done to stop this'
e = queryme4(text,query5) 
e
queryme1(text,query5)
queryme2(text,query5)
queryme3(text,query5)
queryme4(text,query5)
text = text + query5 + e

query6 = 'have they been able to find a vaccine'
f = queryme4(text,query6)
f
queryme1(text,query6)
queryme2(text,query6)
queryme3(text,query6)
queryme4(text,query6)

text = text + query5 + f


#queryme 4 this is based on the hugging face quac trained with history

#hugging face trained on quac with history

tokenizer_quac_history = AutoTokenizer.from_pretrained("Jellevdl/bert-base-uncased-finetuned-quac-2QA-History-v2")

model_quac_history = AutoModelForQuestionAnswering.from_pretrained("Jellevdl/bert-base-uncased-finetuned-quac-2QA-History-v2")

def queryme4(text,question):
    encoding = tokenizer_quac_history.encode_plus(text=question,text_pair=text)
    inputs = encoding['input_ids']  #Token embeddings
    sentence_embedding = encoding['token_type_ids']  #Segment embeddings
    tokens = tokenizer_quac_history.convert_ids_to_tokens(inputs)
    start_scores = model_quac_history(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))[0]
    end_scores =  model_quac_history(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))[1]
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)
    answer = ' '.join(tokens[start_index:end_index+1])
    corrected_answer = ''

    for word in answer.split():
     
    #If it's a subword token
        if word[0:2] == '##':
            corrected_answer += word[2:]
        else:
            corrected_answer += ' ' + word

    return corrected_answer

text ='In late December 2019, an outbreak of a mysterious pneumonia characterized by fever, dry cough, and fatigue, and occasional gastrointestinal symptoms happened in a seafood wholesale wet market, the Huanan Seafood Wholesale Market, in Wuhan, Hubei, China.1 The initial outbreak was reported in the market in December 2019 and involved about 66% of the staff there. The market was shut down on January 1, 2020, after the announcement of an epidemiologic alert by the local health authority on December 31, 2019. However, in the following month (January) thousands of people in China, including many provinces (such as Hubei, Zhejiang, Guangdong, Henan, Hunan, etc.) and cities (Beijing and Shanghai) were attacked by the rampant spreading of the disease.Furthermore, the disease traveled to other countries, such as Thailand, Japan, Republic of Korea, Viet Nam, Germany, United States, and Singapore. The first case reported in our country was on January 21, 2019. As of February 6, 2020, a total of 28,276 confirmed cases with 565 deaths globally were documented by WHO, involving at least 25 countries.The pathogen of the outbreak was later identified as a novel beta-coronavirus, named 2019 novel coronavirus (2019-nCoV) and recalled to our mind the terrible memory of the severe acute respiratory syndrome (SARS-2003, caused by another beta-coronavirus) that occurred 17 years ago.In 2003, a new coronavirus, the etiology of a mysterious pneumonia, also originated from southeast China, especially Guangdong province, and was named SARS coronavirus that fulfilled the Koch’s postulate.The mortality rate caused by the virus was around 10%–15%.Through the years, the medical facilities have been improved; nevertheless, no proper treatment or vaccine is available for the SARS.The emergence of another outbreak in 2012 of novel coronavirus in Middle East shared similar features with the outbreak in 2003.Both were caused by coronavirus but the intermediate host for MERS is thought to be the dromedary camel and the mortality can be up to 37%.5 ||| what are the symptoms of corona virus '

query1 = 'what happened in 2019'
a = queryme2(text,query1)
a 
text = text +query1+a

query2 = 'what was this pneumonia'
b = queryme2(text,query2)
b

text = text + query2 + b
query3 = 'where did this happen'
c = queryme2(text,query3)
c

text = text + query2 +c

query4 = 'where did it spread'
d = queryme2(text,query4)
d

text = text + query4 + d

query5 = 'how did the outbreak happen'
e = queryme2(text,query5)
e

text = text + query5 + e

query6 = 'what was the treatment '
f = queryme2(text,query6)
f


text = text + query5 + f

################################################################
# Context Query Rewriting


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer_castorini = AutoTokenizer.from_pretrained("castorini/t5-base-canard")

model_castorini = AutoModelForSeq2SeqLM.from_pretrained("castorini/t5-base-canard")

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#tokenizer_qr = AutoTokenizer.from_pretrained("peggyhuang/t5-base-canard")

#model_qr = AutoModelForSeq2SeqLM.from_pretrained("peggyhuang/t5-base-canard")

#from transformers import AutoTokenizer, AutoModelForQuestionAnswering

#tokenizer= AutoTokenizer.from_pretrained("peggyhuang/roberta-canard")

#model= AutoModelForQuestionAnswering.from_pretrained("peggyhuang/roberta-canard")

#model_qr 

#from transformers import AutoTokenizer, AutoModelForCausalLM

#tokenizer = AutoTokenizer.from_pretrained("peggyhuang/gpt2-canard")

#model = AutoModelForCausalLM.from_pretrained("peggyhuang/gpt2-canard")

def rewrite(x):
    encoded_input = tokenizer_castorini(x,max_length=512,truncation=True,return_tensors="pt")
    encoder_output = model_castorini.generate(input_ids=encoded_input["input_ids"])
    output = tokenizer_castorini.decode(encoder_output[0],skip_special_tokens=True)
    return output

orig = 'The novel corona virus originated in the city of wuhan,china.It was reported that intially over a 1000 people were infected with the virus which started from a  small market in that city. It is claimed that the virus was transmitted through bats.The WHO sounded an alarm after 2 months of the report of the fist case of the infection.Meanwhile, the infection had spread to more than 30 countries.The main symptom of the virus remains to be sore throat, headache and fever. The only way to save yourself is by regularly washing hands with soap and water and wearing a mask always. Today more than 140 countries have been infected with this virus'
text = orig 
q1 = 'what are the symptoms of the virus'
a = queryme1(text,q1)
a 
text = text + '||| ' + q1

q2 = 'how to save yourself?'
text = text + '||| ' + q2
q2 = rewrite(text)
b = queryme1(text,q2)
b
 
text
q3 = 'how many countries have been infected by it?'
text = text + '||| ' + q3
q3 = rewrite(text)
q3
c = queryme1(orig,q3)
c
text  
q3 = 'where did it originate?'
text = text + '||| ' + q3
q3 = rewrite(text)
q3
d = queryme1(orig,q3)
d 
text 
q4 = 'When did WHO sound an alarm about it?'
text = text + '||| ' + q4
q4 = rewrite(text)
e = queryme1(orig,q4)
e
text

q5 = ' How did this transmit'
text = text + '||| ' + q5
q5 = rewrite(text)
e = queryme1(orig,q5)
e


