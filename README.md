## A Commonsense-Infused Language-Agnostic Learning Framework for Enhancing Prediction of Political Polarity in Multilingual News Headlines

## Abstract

>Predicting the political polarity of news headlines is a challenging task as they are inherently short, catchy, appealing, context-deficient, and contain only subtle bias clues. It becomes even more challenging in a multilingual setting involving low-resource languages. Our research hypothesis is that the use of additional knowledge, such as commonsense knowledge can compensate for a lack of adequate context. However, in a multilingual setting, it becomes ineffective as the majority of the underlying knowledge sources are available only in high-resource languages, such as English. To overcome this barrier, we propose to utilise the Inferential Commonsense Knowledge (IC_Knwl) via a Translate-Retrieve-Translate strategy to introduce a learning framework for the prediction of political polarity in multilingual news headlines. To evaluate the effectiveness of our framework, we present a dataset of multilingual news headlines.<br/>

For More details refer our paper (Coming Soon!!)

## Requirements

We recommend Conda with Python3. Use requirements.yml to create the necessary environment.  

## Dataset
    
>The dataset and its generation scripts are stored in the data folder.<br/>
Follow https://github.com/allenai/comet-atomic-2020/ to retrieve the Inferential Commonsense Knowledge (IC_Knwl).<br/>
Use https://cloud.google.com/translate for translations.</br>

To replicated the reported dataset run:
```
python3 main.py eventRegistry_apiKey
```
To generate custom dataset, pass the deseired values in the commandline arguments.
For example, to retrieve events in the 'Business' category in the 'Slovene' language reported by 'Delo' run:
```
python3 main.py eventRegistry_apiKey --lang slv --category news/Business --source delo.si
```

## To generate the predictions use one of the following files: 

For headlines only
```
python3 Headline.py 
```

For IC_Knwl only
```
python3 IC_Knwl.py 
```

For headlines with IC_Knwl
```
python3 Headline+IC_Knwl.py 
```

For headlines with attended IC_Knwl
```
python3 Headline+Attn(IC_Knwl).py 
```
