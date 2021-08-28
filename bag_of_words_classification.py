from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from google_drive_downloader import GoogleDriveDownloader as gdd
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm


class Sequences(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)
        self.vectorizer = CountVectorizer(stop_words='english', max_df=0.99, min_df=0.005)
        self.sequences = self.vectorizer.fit_transform(df.review.tolist())
        self.labels = df.label.tolist()
        self.token2idx = self.vectorizer.vocabulary_
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
        
    def __getitem__(self, i):
        return self.sequences[i, :].toarray(), self.labels[i]
    
    def __len__(self):
        return self.sequences.shape[0]

class BagOfWordsClassifier(nn.Module):
    def __init__(self, vocab_size, hidden1, hidden2):
        super(BagOfWordsClassifier, self).__init__()
        self.fc1 = nn.Linear(vocab_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
    
    def forward(self, inputs):
        x = F.relu(self.fc1(inputs.squeeze(1).float()))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = 'data/imdb_reviews.csv'
if not Path(DATA_PATH).is_file():
    gdd.download_file_from_google_drive(
        file_id='1zfM5E6HvKIe7f3rEt1V2gBpw5QOSSKQz',
        dest_path=DATA_PATH,
    )

dataset = Sequences(DATA_PATH)
train_loader = DataLoader(dataset, batch_size=4096)
print(dataset[5][0].shape)

model = BagOfWordsClassifier(len(dataset.token2idx), 128, 64)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.001)
model.train()
train_losses = []
for epoch in range(10):
    progress_bar = tqdm(train_loader, leave=False)
    losses = []
    total = 0
    for inputs, target in progress_bar:
        model.zero_grad()

        output = model(inputs)
        loss = criterion(output.squeeze(), target.float())
        
        loss.backward()
              
        nn.utils.clip_grad_norm_(model.parameters(), 3)

        optimizer.step()
        
        progress_bar.set_description(f'Loss: {loss.item():.3f}')
        
        losses.append(loss.item())
        total += 1
    
    epoch_loss = sum(losses) / total
    train_losses.append(epoch_loss)
        
    tqdm.write(f'Epoch #{epoch + 1}\tTrain Loss: {epoch_loss:.3f}')

def predict_sentiment(text):
    model.eval()
    with torch.no_grad():
        test_vector = torch.LongTensor(dataset.vectorizer.transform([text]).toarray())

        output = model(test_vector)
        prediction = torch.sigmoid(output).item()

        if prediction > 0.5:
            print(f'{prediction:0.3}: Positive sentiment')
        else:
            print(f'{prediction:0.3}: Negative sentiment')

test_text = """
This poor excuse for a movie is terrible. It has been 'so good it's bad' for a
while, and the high ratings are a good form of sarcasm, I have to admit. But
now it has to stop. Technically inept, spoon-feeding mundane messages with the
artistic weight of an eighties' commercial, hypocritical to say the least, it
deserves to fall into oblivion. Mr. Derek, I hope you realize you are like that
weird friend that everybody know is lame, but out of kindness and Christian
duty is treated like he's cool or something. That works if you are a good
decent human being, not if you are a horrible arrogant bully like you are. Yes,
Mr. 'Daddy' Derek will end on the history books of the internet for being a
delusional sour old man who thinks to be a good example for kids, but actually
has a poster of Kim Jong-Un in his closet. Destroy this movie if you all have a
conscience, as I hope IHE and all other youtube channel force-closed by Derek
out of SPITE would destroy him in the courts.This poor excuse for a movie is
terrible. It has been 'so good it's bad' for a while, and the high ratings are
a good form of sarcasm, I have to admit. But now it has to stop. Technically
inept, spoon-feeding mundane messages with the artistic weight of an eighties'
commercial, hypocritical to say the least, it deserves to fall into oblivion.
Mr. Derek, I hope you realize you are like that weird friend that everybody
know is lame, but out of kindness and Christian duty is treated like he's cool
or something. That works if you are a good decent human being, not if you are a
horrible arrogant bully like you are. Yes, Mr. 'Daddy' Derek will end on the
history books of the internet for being a delusional sour old man who thinks to
be a good example for kids, but actually has a poster of Kim Jong-Un in his
closet. Destroy this movie if you all have a conscience, as I hope IHE and all
other youtube channel force-closed by Derek out of SPITE would destroy him in
the courts.
"""
predict_sentiment(test_text)

test_text = """
Cool Cat Saves The Kids is a symbolic masterpiece directed by Derek Savage that
is not only satirical in the way it makes fun of the media and politics, but in
the way in questions as how we humans live life and how society tells us to
live life.

Before I get into those details, I wanna talk about the special effects in this
film. They are ASTONISHING, and it shocks me that Cool Cat Saves The Kids got
snubbed by the Oscars for Best Special Effects. This film makes 2001 look like
garbage, and the directing in this film makes Stanley Kubrick look like the
worst director ever. You know what other film did that? Birdemic: Shock and
Terror. Both of these films are masterpieces, but if I had to choose my
favorite out of the 2, I would have to go with Cool Cat Saves The Kids. It is
now my 10th favorite film of all time.

Now, lets get into the symbolism: So you might be asking yourself, Why is Cool
Cat Orange? Well, I can easily explain. Orange is a color. Orange is also a
fruit, and its a very good fruit. You know what else is good? Good behavior.
What behavior does Cool Cat have? He has good behavior. This cannot be a
coincidence, since cool cat has good behavior in the film.

Now, why is Butch The Bully fat? Well, fat means your wide. You wanna know who
was wide? Hitler. Nuff said this cannot be a coincidence.

Why does Erik Estrada suspect Butch The Bully to be a bully? Well look at it
this way. What color of a shirt was Butchy wearing when he walks into the area?
I don't know, its looks like dark purple/dark blue. Why rhymes with dark? Mark.
Mark is that guy from the Room. The Room is the best movie of all time. What is
the opposite of best? Worst. This is how Erik knew Butch was a bully.

and finally, how come Vivica A. Fox isn't having a successful career after
making Kill Bill.

I actually can't answer that question.

Well thanks for reading my review.
"""
predict_sentiment(test_text)

test_text = """
Don't let any bullies out there try and shape your judgment on this gem of a
title.

Some people really don't have anything better to do, except trash a great movie
with annoying 1-star votes and spread lies on the Internet about how "dumb"
Cool Cat is.

I wouldn't be surprised to learn if much of the unwarranted negativity hurled
at this movie is coming from people who haven't even watched this movie for
themselves in the first place. Those people are no worse than the Butch the
Bully, the film's repulsive antagonist.

As it just so happens, one of the main points of "Cool Cat Saves the Kids" is
in addressing the attitudes of mean naysayers who try to demean others who
strive to bring good attitudes and fun vibes into people's lives. The message
to be learned here is that if one is friendly and good to others, the world is
friendly and good to one in return, and that is cool. Conversely, if one is
miserable and leaving 1-star votes on IMDb, one is alone and doesn't have any
friends at all. Ain't that the truth?

The world has uncovered a great, new, young filmmaking talent in "Cool Cat"
creator Derek Savage, and I sure hope that this is only the first of many
amazing films and stories that the world has yet to appreciate.

If you are a cool person who likes to have lots of fun, I guarantee that this
is a movie with charm that will uplift your spirits and reaffirm your positive
attitudes towards life.
"""
predict_sentiment(test_text)

test_text = """
What the heck is this ? There is not one redeeming quality about this terrible
and very poorly done "movie". I can't even say that it's a "so bad it's good
movie".It is undeniably pointless to address all the things wrong here but
unfortunately even the "life lessons" about bullies and stuff like this are so
wrong and terrible that no kid should hear them.The costume is also horrible
and the acting...just unbelievable.No effort whatsoever was put into this thing
and it clearly shows,I have no idea what were they thinking or who was it even
meant for. I feel violated after watching this trash and I deeply recommend you
stay as far away as possible.This is certainly one of the worst pieces of c***
I have ever seen.
"""
predict_sentiment(test_text)