# AMAN-Adaptive Multi-attention Network
Implementation of the AMAN model for Duplicate Question Detection task.

This repository contains an implementation of the sequential model presented in the paper "Adaptive Multi-Attention Network Incorporating Answer Information for Duplicate QuestionDetection" 

* CQADupStack: 
    CQADupStack is a benchmark dataset for community question-answering (cQA) research. It contains threads from twelve StackExchange1 subforums, annotated with duplicate question information and comes with pre-defined training, development, and test splits, both for retrieval and classification experiments.The script below can be used to make these splits, and provides easy access to the different questions, answers, comments, users, and different meta data provided in the set. It also has several evaluation metrics implemented.Some initial baseline results were published at the Australasion Document Computing Symposium (ADCS) 2015, and an analysis of the quality of the set was published at the WebQA Workshop at SIGIR 2016.
    
    http://stackexchange.com/
    
    data download:[url](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/cqadupstack.tar.gz)
    
    data_utils_github : [github](https://github.com/D1Doris/CQADupStack)
* QQP:
    Where else but Quora can a physicist help a chef with a math problem and get cooking tips in return? Quora is a place to gain and share knowledge—about anything. It’s a platform to ask questions and connect with people who contribute unique insights and quality answers. This empowers people to learn from each other and to better understand the world.
    
    Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term.
    
    Currently, Quora uses a Random Forest model to identify duplicate questions. In this competition, Kagglers are challenged to tackle this natural language processing problem by applying advanced techniques to classify whether question pairs are duplicates or not. Doing so will make it easier to find high quality answers to questions resulting in an improved experience for Quora writers, seekers, and readers.
    
    kaggle:[url](https://www.kaggle.com/c/quora-question-pairs)
    
    dataset: [url](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs)
    
email='ld_zhanshi@126.com'
