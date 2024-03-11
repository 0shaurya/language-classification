
A classification model that classifies words into English, Spanish, Japanese, or Russian.
Words inputted as a one hot encoding of 16 characters. Input size is 432 dimensional (16*27). All words are romanized into the English alphabet plus ñ.

Duplicates were not taken into account - for example, "abductor" is in both English and Spanish, but the model must pick only one language. However, I imagine this does not have a large effect on the results.
## Data sources
 - https://github.com/words/an-array-of-english-words/tree/master
 - https://github.com/words/an-array-of-spanish-words/tree/master
 - https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/Japanese2022_10000
 - https://www.kaggle.com/datasets/rtatman/opencorpora-russian
Dataset size was 60000 words, 15k for each language.

## Graphs
![Accuracy and Loss graph](https://raw.githubusercontent.com/0shaurya/language-classification/main/graphs.png)
100 epochs, batch size 256, learning rate 0.003, Adam.

## Results

| Language 1 | Language 2 | Language 3 | Language 4 | Test accuracy
|--|--|--|--|--|
|English|Spanish|Japanese|Russian|86.8%|
|English|Japanese|Russian|n/a|92.31%|
|English|Spanish|n/a|n/a|90.13%|
|English|Japanese|n/a|n/a|97.67%|
|English|Russian|n/a|n/a|92.73%|
