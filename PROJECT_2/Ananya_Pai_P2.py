from bs4 import BeautifulSoup
import requests
import re
import nltk
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import networkx as nx
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize

# This news article details are available in the documentation
url = "https://astronomy.com/news/2023/03/two-potentially-habitable-planets-found-orbiting-distant-star"
res = requests.get(url)

s = BeautifulSoup(res.content, 'html.parser')
cont = s.get_text(strip=True)

cont = re.sub('[^a-zA-Z]', ' ', cont)
cont = cont.lower()
tk = nltk.word_tokenize(cont)
stopwords = nltk.corpus.stopwords.words('english')
tk = [tx for tx in tk if tx not in stopwords]


# EDA

# bar chart for top 10 frequent words
fd = nltk.FreqDist(tk)
tw= fd.most_common(10)
lab = [w[0] for w in tw]
val = [w[1] for w in tw]
plt.figure(figsize=(10, 10))
plt.bar(lab, val, color='blue')
plt.title('Top 10 Words in Text')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.show()

# bigrams
bg = list(nltk.bigrams(tk))
gr = nx.Graph()
for bigram in bg:
    if bigram[0] != bigram[1]:
        gr.add_edge(bigram[0], bigram[1])
plt.figure(figsize=(50, 50))
pos = nx.spring_layout(gr, k=0.5, iterations=50)
nx.draw_networkx_nodes(gr, pos, node_color='lightblue', node_size=1000)
nx.draw_networkx_edges(gr, pos, edge_color='gray', width=1)
nx.draw_networkx_labels(gr, pos, font_size=15, font_family='sans-serif')
plt.title('Bigram Network Graph', fontsize=20)
plt.axis('off')
plt.show()

# wordcloud
wc = WordCloud().generate(cont)
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()

# POS tag
#nltk.download('averaged_perceptron_tagger')
tk1 = word_tokenize(cont)
pos_tags = nltk.pos_tag(tk1)
tag = nltk.FreqDist(tag for (word, tag) in pos_tags)
tag.plot()
plt.show()


# Sentiment analyzer

#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('punkt')
nltk.download('stopwords')
aly = SentimentIntensityAnalyzer()
sc = aly.polarity_scores(cont)
print('Positive:', sc['pos'])
print('Negative:', sc['neg'])
print('Neutral:', sc['neu'])
print('Compound:', sc['compound'])
"""Output:
    Positive: 0.089
   Negative: 0.005
   Neutral: 0.906
   Compound: 0.9993"""
"""Conclusion from the score: These scores indicate that the text is mostly neutral, with a slightly
 positive sentiment. The compound score of 0.9993 suggests a very strong positive sentiment overall."""


# Visualize the sentiment

# Pie chart
plt.pie(sc.values(), labels=sc.keys(), colors=['red', 'green', 'blue'])
plt.title('Sentiment Analysis Pie Chart')
plt.show()


# Bar Plot
sentiment_scores = {
    'Positive': sc['pos'],
    'Negative': sc['neg'],
    'Neutral': sc['neu']
}
labels = list(sentiment_scores.keys())
values = list(sentiment_scores.values())
plt.bar(labels, values, color=['green', 'red', 'blue'])
plt.title('Sentiment Analysis Bar Plot')
plt.xlabel('Sentiment')
plt.ylabel('Score')
plt.show()

############################################################################################################
#      THE DOCUMENTATION AND ALL GRAPHS ARE SEPERATELY UPLOADED AS A PDF FILE (P2_SENTIMENTAL_ANALYSIS)        #
############################################################################################################