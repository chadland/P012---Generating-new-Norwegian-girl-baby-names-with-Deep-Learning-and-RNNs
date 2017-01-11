
# Generating new Norwegian girl baby names with Deep Learning and RNNs 

This post will go through the use of Deep Recurrent Neural Nets using TensorFlow on how to generate new Norwegian girl baby names. This might be useful for upcoming (scared-to-death) parents not deciding on a potential name :)   Our main goal with this task is to train a model that perhaps can generate new "sensible" character combinations to successfully create new names or names that do not exist in the training data. This is just a high level hands-on example on how to use TensorFlow and the underlying components used in the net will not be explained in detailed. 

## Task

We will be building a similar character-level language model to generate character sequences, a la Andrej Karpathy’s char-rnn (and see, e.g., a TensorFlow implementation by Sherjil Ozair [here](http://karpathy.github.io/2015/05/21/rnn-effectiveness/). In the end we want the model spit out new names so that it can create new and existing character combinations that hopefully will end up sounding like sensible girl names in Norwegian. 

We will use data from SSB (Statistics Norway) with names of born babies between 2006 and 2015 with more than 4 occurrences in at least one of the 10 years. If the model will be able create new names they might indeed exist from before if less than 4 people in Norway has not been given that name the last 10 years.

https://www.ssb.no/statistikkbanken/selecttable/hovedtabellHjem.asp?KortNavnWeb=navn&CMSSubjectArea=befolkning&checked=true



```python
#Imports
import numpy as np
import tensorflow as tf
%matplotlib inline
import matplotlib.pyplot as plt
import time
import os
import urllib
from tensorflow.models.rnn.ptb import reader
import pandas as pd
import io
import requests

```

## Data

Download the data from SSB that is temporarily stored DropBox and list some simple statistics.
The list consists of 630 (not really a deep learning problem :) ) girl names of babies born in Norway between 2005 - 2015 and that has occurred 4 or more times at least one of the years. This may limit the model to create more exotic combinations as it learns from the most common names.


```python
#Load data an functions
# -*- coding: utf-8 -*-
url = "https://www.dropbox.com/s/rxe3vsvdt03jtvi/2017%20-%2001%20-%2009%20-%20SSBJentenavn20062015V2.csv?dl=1"  # dl=1 is important
import urllib
import csv
import codecs
import sys  
#sys.setdefaultencoding('utf8')

#Download file
u = urllib.urlopen(url)
data = u.read()
u.close()
#print (u.headers.getparam("charset"))

#Save csv to disc 
with open('SSBJentenavn20062015,csv', "wb") as f :
    f.write(data)
f.close()

#Convert data to dataset 
df = pd.read_csv('SSBJentenavn20062015,csv',encoding ='iso-8859-1',delimiter=";")

#Convert df to text and remove numbers
df_to_text = ''.join([i for i in df["Navn"].to_string() if not i.isdigit()])
df_to_text = df_to_text.replace(" ", "")

#List the data frame and number of names 
print 'Number of Norwegian girl names of babies born between 2005-2015 with more than 4 occurences: ' + str(len(df))
print 'List of names and counts each year:'
print df
print df_to_text
```

    Number of Norwegian girl names of babies born between 2005-2015 with more than 4 occurences: 630
    List of names and counts each year:
               Navn  2015  2014  2013  2012  2011  2010  2009  2008  2007  2006
    0       Abigail     8    11    15    18     6    10    10     8     8     4
    1           Ada   131   110   128   128   109    95    99    91   101    71
    2         Adela     0     7     6     5     6     4    11     4     0     0
    3         Adele    46    46    69    78    59    49    50    58    45    41
    4        Adelen    48    37    50     6     7     9     0     0     7     0
    5       Adelina     9    10     6     7     6     6     5     5     6     0
    6         Adina     7     7     4     6     0     0    10     8     5     0
    7         Adine     4     9     8    10     7     8    11    12     8    10
    8          Adna    10     7     0     0     0     5     7     4     5     6
    9       Adriana    23    19    19    25    24    24    28    17    19    13
    10       Agathe     8    12    12     7     8    14    10    14    12    18
    11        Agnes   108   138    96    82    99    79    69    67    54    55
    12      Agnethe     8     0     0     0     0     4    12     8     4     0
    13         Aida    11    10    12    11    15    16    10     9    10    15
    14         Aila     0    10     6     5     6     5     8     6     4     7
    15        Ailin     9     6     7     6     4    10    10     6     4     9
    16        Aimee     0     5     0     6     0     8     6     7     5    11
    17         Aina    13     7    11    14     7     8    12     6     8    12
    18        Aisha    22    44    38    40    37    38    42    25    27    21
    19         Alba    22    18    14    20    16    18     9    10    10    12
    20         Alea    15     9     5    12     8     6     5     0     0     0
    21   Aleksandra    17    32    31    24    36    33    25    36    49    27
    22       Alette     7    12     7     9     4     8     7    11     7     9
    23    Alexandra    63    48    68    84    66    82    89    89    80    74
    24        Alice    48    30    29    34    29    27    25    24    23    18
    25       Alicia    26    22    21    17    17    19    13    17    11    11
    26       Alicja    11    14     7    14    13    13    14     7    11     7
    27        Alida    46    28    37    29    37    43    30    30    26    24
    28        Alina   142    48    37    40    36    24    33    26    26    32
    29        Aline    19    12    14    15     9     7     5     9     0    10
    ..          ...   ...   ...   ...   ...   ...   ...   ...   ...   ...   ...
    600    Veslemøy     7     8     4    12     9    17    15    11    12     8
    601    Victoria   160   184   254   224   190   230   224   194   215   177
    602        Vida    20    29    27    29    12    14     8    10     9     4
    603    Viktoria    34    52    53    52    42    58    44    52    42    44
    604       Vilde   213   257   235   254   236   277   289   279   312   291
    605       Vilja    50    58    60    54    54    63    41    31    26    35
    606       Vilje    35    29    40    39    32    23    19    22    15     6
    607     Villemo     6     5    11     5     6     9     8     6     5     4
    608       Vilma    38    53    36    50    41    40    39    34    49    25
    609       Viola    12    12    11    21    12     9    15    12     7    14
    610      Vivian     7     5     5    11     6     6     7     7    10    17
    611         Vår     6     6     8     7     7     8    15    14    11    12
    612       Vårin    20    17    24    22    24    10    16    18    12    14
    613    Weronika     6     0     4     6     8     6    14     9    12    15
    614    Wiktoria    17    12    15    11    22    29    27    37    33    36
    615       Wilma    38    31    30    28    34    33    24    32    33    23
    616        Yara     6     6     5    10     9     6     5     5     5     6
    617      Yasmin    11    15    18    26    27    15    20    17    27    21
    618        Ylva    81    73    68    68    74    64    62    46    60    54
    619       Yusra     0    10    11     8    11    22     9     7    10     5
    620         Åse    13     5    11    11    12    13     8    16    10    14
    621      Åshild     0     4     6     5     7     6     6     4     9    14
    622        Åsne     9     7    11    10    13    12    15     7    19    11
    623        Åsta    10     5     0     5     4     6     0     0     0     0
    624       Zahra    10    12    18     6    13    13    17    18     8    16
    625      Zainab     9    12    12    15    15    16    16     8    13     9
    626        Zara    20    18     5    12     9    18    15    16    15    16
    627         Zoe     6    10    15     5    13     9    11     8     5     7
    628       Zofia    16    12    15    17    13    13    15     8    12     7
    629     Zuzanna    16    29    35    28    26    25    36    27    33    16
    
    [630 rows x 11 columns]
    Abigail
    Ada
    Adela
    Adele
    Adelen
    Adelina
    Adina
    Adine
    Adna
    Adriana
    Agathe
    Agnes
    Agnethe
    Aida
    Aila
    Ailin
    Aimee
    Aina
    Aisha
    Alba
    Alea
    Aleksandra
    Alette
    Alexandra
    Alice
    Alicia
    Alicja
    Alida
    Alina
    Aline
    Alisa
    Alise
    Alisha
    Alma
    Alva
    Alvilde
    Amal
    Amalia
    Amalie
    Amanda
    Amelia
    Amelie
    Amelija
    Amina
    Aminda
    Amine
    Amira
    Amna
    Amy
    Ana
    Anastasia
    Andrea
    Andrine
    Ane
    Anea
    Anette
    Angela
    Angelica
    Angelika
    Angelina
    Anine
    Anisa
    Anita
    Anja
    Ann
    Anna
    Annabel
    Annabell
    Annabelle
    Anne
    Anneli
    Annie
    Annika
    Anniken
    Anny
    Aria
    Ariana
    Ariane
    Ariel
    Arya
    Asta
    Astri
    Astrid
    Augusta
    Aurora
    Ava
    Aya
    Ayesha
    Ayla
    Aylin
    Beatrice
    Bella
    Benedicte
    Benedikte
    Bertine
    Betina
    Bettina
    Bianca
    Bushra
    Camilla
    Carina
    Carla
    Carmen
    Caroline
    Cassandra
    Cathrine
    Cecilia
    Cecilie
    Celia
    Celin
    Celina
    Celine
    Charlotte
    Chloe
    Christiane
    Christina
    Christine
    Cindy
    Clara
    Cornelia
    Dania
    Daniela
    Daniella
    Diana
    Dina
    Dominika
    Dorthe
    Dorthea
    Ea
    Ebba
    Edda
    Edel
    Edith
    Eileen
    Eir
    Eira
    Eiril
    Eirill
    Eirin
    Eivor
    Ela
    Elea
    Eleah
    Elena
    Eli
    Eliana
    Elida
    Elif
    Elin
    Elina
    Eline
    Elisa
    Elisabeth
    Elise
    Eliza
    Elizabeth
    Ella
    Elle
    Ellen
    Elli
    Ellie
    Ellinor
    Elly
    Elma
    Elsa
    Else
    Elvine
    Elvira
    Ema
    Embla
    Emeli
    Emelie
    Emely
    Emilia
    Emilie
    Emilija
    Emily
    Emina
    Emine
    Emma
    Emmeli
    Emmelin
    Emmeline
    Emmi
    Emmy
    Enya
    Erica
    Erika
    Erle
    Ester
    Esther
    Eva
    Evelina
    Evelyn
    Fanny
    Fatima
    Felicia
    Filippa
    Fiona
    Fredrikke
    Freja
    Freya
    Freyja
    Frida
    Fride
    Frøya
    Frøydis
    Gabija
    Gabriela
    Gabriele
    Gabriella
    Gabrielle
    Gina
    Greta
    Gunhild
    Guro
    Hafsa
    Hailey
    Halima
    Hamdi
    Hana
    Hanan
    Hanna
    Hannah
    Hanne
    Hedda
    Hedvig
    Heidi
    Helen
    Helena
    Helene
    Helin
    Helle
    Helmine
    Hennie
    Henny
    Henriette
    Henrikke
    Hermine
    Hilde
    Iben
    Ida
    Idun
    Idunn
    Iman
    Ina
    Inaya
    Ine
    Ines
    Inga
    Ingebjørg
    Ingeborg
    Ingelin
    Inger
    Ingrid
    Ingvild
    Iqra
    Irene
    Iris
    Irmelin
    Isa
    Isabel
    Isabell
    Isabella
    Isabelle
    Iselin
    Isra
    Jana
    Janne
    Jasmin
    Jasmine
    Jennie
    Jennifer
    Jenny
    Jessica
    Joanna
    Johanna
    Johanne
    Josefine
    Josephine
    Julia
    Juliane
    Julianne
    Julie
    June
    Juni
    Kaia
    Kaisa
    Kaja
    Kajsa
    Kamila
    Kamile
    Kamilla
    Karen
    Kari
    Karianne
    Karin
    Karina
    Karine
    Karla
    Karolina
    Karoline
    Katarina
    Katharina
    Kathrine
    Katinka
    Katja
    Katrina
    Katrine
    Kaya
    Kayla
    Khadija
    Kine
    Kira
    Kjersti
    Klara
    Klaudia
    Kornelia
    Kristiane
    Kristin
    Kristina
    Kristine
    Laiba
    Laila
    Lana
    Lara
    Laura
    Lea
    Leah
    Leana
    Leja
    Lena
    Lene
    Leni
    Leona
    Leonora
    Lerke
    Lia
    Liana
    Liepa
    Liliana
    Lilja
    Lilje
    Lilli
    Lillian
    Lillie
    Lilly
    Lily
    Lina
    Linda
    Linde
    Line
    Linea
    Linn
    Linnea
    Lisa
    Lise
    Liv
    Liva
    Live
    Livia
    Lone
    Lotta
    Lotte
    Louisa
    Louise
    Lovise
    Luca
    Lucia
    Lucy
    Luka
    Luna
    Lycke
    Lydia
    Lykke
    Lærke
    Madeleine
    Madelen
    Madelene
    Magdalena
    Maia
    Maida
    Maiken
    Maja
    Malak
    Malena
    Malene
    Mali
    Malin
    Maren
    Margit
    Margrete
    Margrethe
    Mari
    Maria
    Mariam
    Marianne
    Marie
    Mariel
    Mariell
    Marielle
    Marina
    Marion
    Marit
    Marita
    Marlene
    Marta
    Marte
    Martha
    Marthe
    Marthine
    Martina
    Martine
    Martyna
    Marwa
    Mary
    Maryam
    Mathea
    Mathilda
    Mathilde
    Matilda
    Matilde
    Maud
    May
    Maya
    Medina
    Melina
    Melinda
    Melisa
    Melissa
    Mia
    Michelle
    Mie
    Mikaela
    Mila
    Milana
    Milena
    Milja
    Milla
    Mille
    Millie
    Mina
    Mira
    Miriam
    Moa
    Molly
    Mona
    Monika
    Muna
    Nadia
    Najma
    Nanna
    Naomi
    Natalia
    Natalie
    Natasha
    Nathalie
    Nawal
    Nela
    Nelia
    Nellie
    Nelly
    Nicole
    Nicoline
    Nikola
    Nikoline
    Nila
    Nina
    Noelle
    Noor
    Nora
    Norah
    Nova
    Oda
    Olava
    Olea
    Oline
    Olivia
    Oliwia
    Othelia
    Othelie
    Othilie
    Otilie
    Patrycja
    Paula
    Paulina
    Pauline
    Pernille
    Petra
    Pia
    Rachel
    Ragna
    Ragnhild
    Rakel
    Randi
    Rania
    Rebecca
    Rebecka
    Rebekka
    Regine
    Renate
    Rikke
    Ronja
    Rose
    Runa
    Ruth
    Sabrin
    Sabrina
    Safa
    Saga
    Sahar
    Salma
    Samantha
    Samira
    Sandra
    Sanna
    Sannah
    Sanne
    Sara
    Sarah
    Savannah
    Selena
    Selin
    Selina
    Seline
    Selma
    Serina
    Serine
    Sienna
    Signe
    Sigrid
    Sigrun
    Siham
    Silja
    Silje
    Simone
    Sina
    Sine
    Siren
    Siri
    Siril
    Sofia
    Sofie
    Sofija
    Sol
    Solveig
    Sonia
    Sonja
    Sophia
    Sophie
    Stella
    Stina
    Stine
    Sumaya
    Sundus
    Sunniva
    Susanna
    Susanne
    Synne
    Synnøve
    Tale
    Tara
    Tea
    Teja
    Thale
    Thea
    Thelma
    Therese
    Thilde
    Tia
    Tilda
    Tilde
    Tilia
    Tilja
    Tilje
    Tilla
    Tina
    Tindra
    Tine
    Tiril
    Tirill
    Tomine
    Tone
    Tonje
    Tora
    Trine
    Tuva
    Tyra
    Ugne
    Ulla
    Ulrikke
    Una
    Urte
    Valentina
    Vanessa
    Vanja
    Vega
    Vera
    Veronica
    Veronika
    Veslemøy
    Victoria
    Vida
    Viktoria
    Vilde
    Vilja
    Vilje
    Villemo
    Vilma
    Viola
    Vivian
    Vår
    Vårin
    Weronika
    Wiktoria
    Wilma
    Yara
    Yasmin
    Ylva
    Yusra
    Åse
    Åshild
    Åsne
    Åsta
    Zahra
    Zainab
    Zara
    Zoe
    Zofia
    Zuzanna


## Generating vocabulary 

In this step we generate unique characters used in all the different names. Translations lists for indexing are also created as we all know it is easier to work with numbers than with characters. This is quite a small data set but it is always a good practice to index your vocabulary.   

The data variable is just a number representation of all the text.


```python
# -*- coding: utf-8 -*-
vocab = set(df_to_text)
vocab_size = len(vocab)
idx_to_vocab = dict(enumerate(vocab))
vocab_to_idx = dict(zip(idx_to_vocab.values(), idx_to_vocab.keys()))
data = [vocab_to_idx[c] for c in df_to_text]
print vocab
print vocab_size
print idx_to_vocab
print vocab_to_idx
print data
```

    set([u'\xe6', u'\n', u'\xc5', u'V', u'A', u'C', u'B', u'E', u'D', u'G', u'F', u'I', u'H', u'K', u'J', u'M', u'L', u'O', u'N', u'P', u'S', u'R', u'U', u'T', u'W', u'\xf8', u'Y', u'\xe5', u'Z', u'a', u'c', u'b', u'e', u'd', u'g', u'f', u'i', u'h', u'k', u'j', u'm', u'l', u'o', u'n', u'q', u'p', u's', u'r', u'u', u't', u'w', u'v', u'y', u'x', u'z'])
    55
    {0: u'\xe6', 1: u'\n', 2: u'\xc5', 3: u'V', 4: u'A', 5: u'C', 6: u'B', 7: u'E', 8: u'D', 9: u'G', 10: u'F', 11: u'I', 12: u'H', 13: u'K', 14: u'J', 15: u'M', 16: u'L', 17: u'O', 18: u'N', 19: u'P', 20: u'S', 21: u'R', 22: u'U', 23: u'T', 24: u'W', 25: u'\xf8', 26: u'Y', 27: u'\xe5', 28: u'Z', 29: u'a', 30: u'c', 31: u'b', 32: u'e', 33: u'd', 34: u'g', 35: u'f', 36: u'i', 37: u'h', 38: u'k', 39: u'j', 40: u'm', 41: u'l', 42: u'o', 43: u'n', 44: u'q', 45: u'p', 46: u's', 47: u'r', 48: u'u', 49: u't', 50: u'w', 51: u'v', 52: u'y', 53: u'x', 54: u'z'}
    {u'j': 39, u'\n': 1, u'E': 7, u'x': 53, u'A': 4, u'C': 5, u'B': 6, u'\xc5': 2, u'D': 8, u'G': 9, u'F': 10, u'I': 11, u'H': 12, u'K': 13, u'J': 14, u'M': 15, u'L': 16, u'O': 17, u'N': 18, u'P': 19, u'S': 20, u'R': 21, u'U': 22, u'T': 23, u'W': 24, u'V': 3, u'Y': 26, u'e': 32, u'Z': 28, u'a': 29, u'c': 30, u'b': 31, u'\xe5': 27, u'd': 33, u'g': 34, u'\xe6': 0, u'i': 36, u'h': 37, u'k': 38, u'f': 35, u'm': 40, u'l': 41, u'o': 42, u'n': 43, u'q': 44, u'p': 45, u's': 46, u'r': 47, u'u': 48, u't': 49, u'w': 50, u'v': 51, u'y': 52, u'\xf8': 25, u'z': 54}
    [4, 31, 36, 34, 29, 36, 41, 1, 4, 33, 29, 1, 4, 33, 32, 41, 29, 1, 4, 33, 32, 41, 32, 1, 4, 33, 32, 41, 32, 43, 1, 4, 33, 32, 41, 36, 43, 29, 1, 4, 33, 36, 43, 29, 1, 4, 33, 36, 43, 32, 1, 4, 33, 43, 29, 1, 4, 33, 47, 36, 29, 43, 29, 1, 4, 34, 29, 49, 37, 32, 1, 4, 34, 43, 32, 46, 1, 4, 34, 43, 32, 49, 37, 32, 1, 4, 36, 33, 29, 1, 4, 36, 41, 29, 1, 4, 36, 41, 36, 43, 1, 4, 36, 40, 32, 32, 1, 4, 36, 43, 29, 1, 4, 36, 46, 37, 29, 1, 4, 41, 31, 29, 1, 4, 41, 32, 29, 1, 4, 41, 32, 38, 46, 29, 43, 33, 47, 29, 1, 4, 41, 32, 49, 49, 32, 1, 4, 41, 32, 53, 29, 43, 33, 47, 29, 1, 4, 41, 36, 30, 32, 1, 4, 41, 36, 30, 36, 29, 1, 4, 41, 36, 30, 39, 29, 1, 4, 41, 36, 33, 29, 1, 4, 41, 36, 43, 29, 1, 4, 41, 36, 43, 32, 1, 4, 41, 36, 46, 29, 1, 4, 41, 36, 46, 32, 1, 4, 41, 36, 46, 37, 29, 1, 4, 41, 40, 29, 1, 4, 41, 51, 29, 1, 4, 41, 51, 36, 41, 33, 32, 1, 4, 40, 29, 41, 1, 4, 40, 29, 41, 36, 29, 1, 4, 40, 29, 41, 36, 32, 1, 4, 40, 29, 43, 33, 29, 1, 4, 40, 32, 41, 36, 29, 1, 4, 40, 32, 41, 36, 32, 1, 4, 40, 32, 41, 36, 39, 29, 1, 4, 40, 36, 43, 29, 1, 4, 40, 36, 43, 33, 29, 1, 4, 40, 36, 43, 32, 1, 4, 40, 36, 47, 29, 1, 4, 40, 43, 29, 1, 4, 40, 52, 1, 4, 43, 29, 1, 4, 43, 29, 46, 49, 29, 46, 36, 29, 1, 4, 43, 33, 47, 32, 29, 1, 4, 43, 33, 47, 36, 43, 32, 1, 4, 43, 32, 1, 4, 43, 32, 29, 1, 4, 43, 32, 49, 49, 32, 1, 4, 43, 34, 32, 41, 29, 1, 4, 43, 34, 32, 41, 36, 30, 29, 1, 4, 43, 34, 32, 41, 36, 38, 29, 1, 4, 43, 34, 32, 41, 36, 43, 29, 1, 4, 43, 36, 43, 32, 1, 4, 43, 36, 46, 29, 1, 4, 43, 36, 49, 29, 1, 4, 43, 39, 29, 1, 4, 43, 43, 1, 4, 43, 43, 29, 1, 4, 43, 43, 29, 31, 32, 41, 1, 4, 43, 43, 29, 31, 32, 41, 41, 1, 4, 43, 43, 29, 31, 32, 41, 41, 32, 1, 4, 43, 43, 32, 1, 4, 43, 43, 32, 41, 36, 1, 4, 43, 43, 36, 32, 1, 4, 43, 43, 36, 38, 29, 1, 4, 43, 43, 36, 38, 32, 43, 1, 4, 43, 43, 52, 1, 4, 47, 36, 29, 1, 4, 47, 36, 29, 43, 29, 1, 4, 47, 36, 29, 43, 32, 1, 4, 47, 36, 32, 41, 1, 4, 47, 52, 29, 1, 4, 46, 49, 29, 1, 4, 46, 49, 47, 36, 1, 4, 46, 49, 47, 36, 33, 1, 4, 48, 34, 48, 46, 49, 29, 1, 4, 48, 47, 42, 47, 29, 1, 4, 51, 29, 1, 4, 52, 29, 1, 4, 52, 32, 46, 37, 29, 1, 4, 52, 41, 29, 1, 4, 52, 41, 36, 43, 1, 6, 32, 29, 49, 47, 36, 30, 32, 1, 6, 32, 41, 41, 29, 1, 6, 32, 43, 32, 33, 36, 30, 49, 32, 1, 6, 32, 43, 32, 33, 36, 38, 49, 32, 1, 6, 32, 47, 49, 36, 43, 32, 1, 6, 32, 49, 36, 43, 29, 1, 6, 32, 49, 49, 36, 43, 29, 1, 6, 36, 29, 43, 30, 29, 1, 6, 48, 46, 37, 47, 29, 1, 5, 29, 40, 36, 41, 41, 29, 1, 5, 29, 47, 36, 43, 29, 1, 5, 29, 47, 41, 29, 1, 5, 29, 47, 40, 32, 43, 1, 5, 29, 47, 42, 41, 36, 43, 32, 1, 5, 29, 46, 46, 29, 43, 33, 47, 29, 1, 5, 29, 49, 37, 47, 36, 43, 32, 1, 5, 32, 30, 36, 41, 36, 29, 1, 5, 32, 30, 36, 41, 36, 32, 1, 5, 32, 41, 36, 29, 1, 5, 32, 41, 36, 43, 1, 5, 32, 41, 36, 43, 29, 1, 5, 32, 41, 36, 43, 32, 1, 5, 37, 29, 47, 41, 42, 49, 49, 32, 1, 5, 37, 41, 42, 32, 1, 5, 37, 47, 36, 46, 49, 36, 29, 43, 32, 1, 5, 37, 47, 36, 46, 49, 36, 43, 29, 1, 5, 37, 47, 36, 46, 49, 36, 43, 32, 1, 5, 36, 43, 33, 52, 1, 5, 41, 29, 47, 29, 1, 5, 42, 47, 43, 32, 41, 36, 29, 1, 8, 29, 43, 36, 29, 1, 8, 29, 43, 36, 32, 41, 29, 1, 8, 29, 43, 36, 32, 41, 41, 29, 1, 8, 36, 29, 43, 29, 1, 8, 36, 43, 29, 1, 8, 42, 40, 36, 43, 36, 38, 29, 1, 8, 42, 47, 49, 37, 32, 1, 8, 42, 47, 49, 37, 32, 29, 1, 7, 29, 1, 7, 31, 31, 29, 1, 7, 33, 33, 29, 1, 7, 33, 32, 41, 1, 7, 33, 36, 49, 37, 1, 7, 36, 41, 32, 32, 43, 1, 7, 36, 47, 1, 7, 36, 47, 29, 1, 7, 36, 47, 36, 41, 1, 7, 36, 47, 36, 41, 41, 1, 7, 36, 47, 36, 43, 1, 7, 36, 51, 42, 47, 1, 7, 41, 29, 1, 7, 41, 32, 29, 1, 7, 41, 32, 29, 37, 1, 7, 41, 32, 43, 29, 1, 7, 41, 36, 1, 7, 41, 36, 29, 43, 29, 1, 7, 41, 36, 33, 29, 1, 7, 41, 36, 35, 1, 7, 41, 36, 43, 1, 7, 41, 36, 43, 29, 1, 7, 41, 36, 43, 32, 1, 7, 41, 36, 46, 29, 1, 7, 41, 36, 46, 29, 31, 32, 49, 37, 1, 7, 41, 36, 46, 32, 1, 7, 41, 36, 54, 29, 1, 7, 41, 36, 54, 29, 31, 32, 49, 37, 1, 7, 41, 41, 29, 1, 7, 41, 41, 32, 1, 7, 41, 41, 32, 43, 1, 7, 41, 41, 36, 1, 7, 41, 41, 36, 32, 1, 7, 41, 41, 36, 43, 42, 47, 1, 7, 41, 41, 52, 1, 7, 41, 40, 29, 1, 7, 41, 46, 29, 1, 7, 41, 46, 32, 1, 7, 41, 51, 36, 43, 32, 1, 7, 41, 51, 36, 47, 29, 1, 7, 40, 29, 1, 7, 40, 31, 41, 29, 1, 7, 40, 32, 41, 36, 1, 7, 40, 32, 41, 36, 32, 1, 7, 40, 32, 41, 52, 1, 7, 40, 36, 41, 36, 29, 1, 7, 40, 36, 41, 36, 32, 1, 7, 40, 36, 41, 36, 39, 29, 1, 7, 40, 36, 41, 52, 1, 7, 40, 36, 43, 29, 1, 7, 40, 36, 43, 32, 1, 7, 40, 40, 29, 1, 7, 40, 40, 32, 41, 36, 1, 7, 40, 40, 32, 41, 36, 43, 1, 7, 40, 40, 32, 41, 36, 43, 32, 1, 7, 40, 40, 36, 1, 7, 40, 40, 52, 1, 7, 43, 52, 29, 1, 7, 47, 36, 30, 29, 1, 7, 47, 36, 38, 29, 1, 7, 47, 41, 32, 1, 7, 46, 49, 32, 47, 1, 7, 46, 49, 37, 32, 47, 1, 7, 51, 29, 1, 7, 51, 32, 41, 36, 43, 29, 1, 7, 51, 32, 41, 52, 43, 1, 10, 29, 43, 43, 52, 1, 10, 29, 49, 36, 40, 29, 1, 10, 32, 41, 36, 30, 36, 29, 1, 10, 36, 41, 36, 45, 45, 29, 1, 10, 36, 42, 43, 29, 1, 10, 47, 32, 33, 47, 36, 38, 38, 32, 1, 10, 47, 32, 39, 29, 1, 10, 47, 32, 52, 29, 1, 10, 47, 32, 52, 39, 29, 1, 10, 47, 36, 33, 29, 1, 10, 47, 36, 33, 32, 1, 10, 47, 25, 52, 29, 1, 10, 47, 25, 52, 33, 36, 46, 1, 9, 29, 31, 36, 39, 29, 1, 9, 29, 31, 47, 36, 32, 41, 29, 1, 9, 29, 31, 47, 36, 32, 41, 32, 1, 9, 29, 31, 47, 36, 32, 41, 41, 29, 1, 9, 29, 31, 47, 36, 32, 41, 41, 32, 1, 9, 36, 43, 29, 1, 9, 47, 32, 49, 29, 1, 9, 48, 43, 37, 36, 41, 33, 1, 9, 48, 47, 42, 1, 12, 29, 35, 46, 29, 1, 12, 29, 36, 41, 32, 52, 1, 12, 29, 41, 36, 40, 29, 1, 12, 29, 40, 33, 36, 1, 12, 29, 43, 29, 1, 12, 29, 43, 29, 43, 1, 12, 29, 43, 43, 29, 1, 12, 29, 43, 43, 29, 37, 1, 12, 29, 43, 43, 32, 1, 12, 32, 33, 33, 29, 1, 12, 32, 33, 51, 36, 34, 1, 12, 32, 36, 33, 36, 1, 12, 32, 41, 32, 43, 1, 12, 32, 41, 32, 43, 29, 1, 12, 32, 41, 32, 43, 32, 1, 12, 32, 41, 36, 43, 1, 12, 32, 41, 41, 32, 1, 12, 32, 41, 40, 36, 43, 32, 1, 12, 32, 43, 43, 36, 32, 1, 12, 32, 43, 43, 52, 1, 12, 32, 43, 47, 36, 32, 49, 49, 32, 1, 12, 32, 43, 47, 36, 38, 38, 32, 1, 12, 32, 47, 40, 36, 43, 32, 1, 12, 36, 41, 33, 32, 1, 11, 31, 32, 43, 1, 11, 33, 29, 1, 11, 33, 48, 43, 1, 11, 33, 48, 43, 43, 1, 11, 40, 29, 43, 1, 11, 43, 29, 1, 11, 43, 29, 52, 29, 1, 11, 43, 32, 1, 11, 43, 32, 46, 1, 11, 43, 34, 29, 1, 11, 43, 34, 32, 31, 39, 25, 47, 34, 1, 11, 43, 34, 32, 31, 42, 47, 34, 1, 11, 43, 34, 32, 41, 36, 43, 1, 11, 43, 34, 32, 47, 1, 11, 43, 34, 47, 36, 33, 1, 11, 43, 34, 51, 36, 41, 33, 1, 11, 44, 47, 29, 1, 11, 47, 32, 43, 32, 1, 11, 47, 36, 46, 1, 11, 47, 40, 32, 41, 36, 43, 1, 11, 46, 29, 1, 11, 46, 29, 31, 32, 41, 1, 11, 46, 29, 31, 32, 41, 41, 1, 11, 46, 29, 31, 32, 41, 41, 29, 1, 11, 46, 29, 31, 32, 41, 41, 32, 1, 11, 46, 32, 41, 36, 43, 1, 11, 46, 47, 29, 1, 14, 29, 43, 29, 1, 14, 29, 43, 43, 32, 1, 14, 29, 46, 40, 36, 43, 1, 14, 29, 46, 40, 36, 43, 32, 1, 14, 32, 43, 43, 36, 32, 1, 14, 32, 43, 43, 36, 35, 32, 47, 1, 14, 32, 43, 43, 52, 1, 14, 32, 46, 46, 36, 30, 29, 1, 14, 42, 29, 43, 43, 29, 1, 14, 42, 37, 29, 43, 43, 29, 1, 14, 42, 37, 29, 43, 43, 32, 1, 14, 42, 46, 32, 35, 36, 43, 32, 1, 14, 42, 46, 32, 45, 37, 36, 43, 32, 1, 14, 48, 41, 36, 29, 1, 14, 48, 41, 36, 29, 43, 32, 1, 14, 48, 41, 36, 29, 43, 43, 32, 1, 14, 48, 41, 36, 32, 1, 14, 48, 43, 32, 1, 14, 48, 43, 36, 1, 13, 29, 36, 29, 1, 13, 29, 36, 46, 29, 1, 13, 29, 39, 29, 1, 13, 29, 39, 46, 29, 1, 13, 29, 40, 36, 41, 29, 1, 13, 29, 40, 36, 41, 32, 1, 13, 29, 40, 36, 41, 41, 29, 1, 13, 29, 47, 32, 43, 1, 13, 29, 47, 36, 1, 13, 29, 47, 36, 29, 43, 43, 32, 1, 13, 29, 47, 36, 43, 1, 13, 29, 47, 36, 43, 29, 1, 13, 29, 47, 36, 43, 32, 1, 13, 29, 47, 41, 29, 1, 13, 29, 47, 42, 41, 36, 43, 29, 1, 13, 29, 47, 42, 41, 36, 43, 32, 1, 13, 29, 49, 29, 47, 36, 43, 29, 1, 13, 29, 49, 37, 29, 47, 36, 43, 29, 1, 13, 29, 49, 37, 47, 36, 43, 32, 1, 13, 29, 49, 36, 43, 38, 29, 1, 13, 29, 49, 39, 29, 1, 13, 29, 49, 47, 36, 43, 29, 1, 13, 29, 49, 47, 36, 43, 32, 1, 13, 29, 52, 29, 1, 13, 29, 52, 41, 29, 1, 13, 37, 29, 33, 36, 39, 29, 1, 13, 36, 43, 32, 1, 13, 36, 47, 29, 1, 13, 39, 32, 47, 46, 49, 36, 1, 13, 41, 29, 47, 29, 1, 13, 41, 29, 48, 33, 36, 29, 1, 13, 42, 47, 43, 32, 41, 36, 29, 1, 13, 47, 36, 46, 49, 36, 29, 43, 32, 1, 13, 47, 36, 46, 49, 36, 43, 1, 13, 47, 36, 46, 49, 36, 43, 29, 1, 13, 47, 36, 46, 49, 36, 43, 32, 1, 16, 29, 36, 31, 29, 1, 16, 29, 36, 41, 29, 1, 16, 29, 43, 29, 1, 16, 29, 47, 29, 1, 16, 29, 48, 47, 29, 1, 16, 32, 29, 1, 16, 32, 29, 37, 1, 16, 32, 29, 43, 29, 1, 16, 32, 39, 29, 1, 16, 32, 43, 29, 1, 16, 32, 43, 32, 1, 16, 32, 43, 36, 1, 16, 32, 42, 43, 29, 1, 16, 32, 42, 43, 42, 47, 29, 1, 16, 32, 47, 38, 32, 1, 16, 36, 29, 1, 16, 36, 29, 43, 29, 1, 16, 36, 32, 45, 29, 1, 16, 36, 41, 36, 29, 43, 29, 1, 16, 36, 41, 39, 29, 1, 16, 36, 41, 39, 32, 1, 16, 36, 41, 41, 36, 1, 16, 36, 41, 41, 36, 29, 43, 1, 16, 36, 41, 41, 36, 32, 1, 16, 36, 41, 41, 52, 1, 16, 36, 41, 52, 1, 16, 36, 43, 29, 1, 16, 36, 43, 33, 29, 1, 16, 36, 43, 33, 32, 1, 16, 36, 43, 32, 1, 16, 36, 43, 32, 29, 1, 16, 36, 43, 43, 1, 16, 36, 43, 43, 32, 29, 1, 16, 36, 46, 29, 1, 16, 36, 46, 32, 1, 16, 36, 51, 1, 16, 36, 51, 29, 1, 16, 36, 51, 32, 1, 16, 36, 51, 36, 29, 1, 16, 42, 43, 32, 1, 16, 42, 49, 49, 29, 1, 16, 42, 49, 49, 32, 1, 16, 42, 48, 36, 46, 29, 1, 16, 42, 48, 36, 46, 32, 1, 16, 42, 51, 36, 46, 32, 1, 16, 48, 30, 29, 1, 16, 48, 30, 36, 29, 1, 16, 48, 30, 52, 1, 16, 48, 38, 29, 1, 16, 48, 43, 29, 1, 16, 52, 30, 38, 32, 1, 16, 52, 33, 36, 29, 1, 16, 52, 38, 38, 32, 1, 16, 0, 47, 38, 32, 1, 15, 29, 33, 32, 41, 32, 36, 43, 32, 1, 15, 29, 33, 32, 41, 32, 43, 1, 15, 29, 33, 32, 41, 32, 43, 32, 1, 15, 29, 34, 33, 29, 41, 32, 43, 29, 1, 15, 29, 36, 29, 1, 15, 29, 36, 33, 29, 1, 15, 29, 36, 38, 32, 43, 1, 15, 29, 39, 29, 1, 15, 29, 41, 29, 38, 1, 15, 29, 41, 32, 43, 29, 1, 15, 29, 41, 32, 43, 32, 1, 15, 29, 41, 36, 1, 15, 29, 41, 36, 43, 1, 15, 29, 47, 32, 43, 1, 15, 29, 47, 34, 36, 49, 1, 15, 29, 47, 34, 47, 32, 49, 32, 1, 15, 29, 47, 34, 47, 32, 49, 37, 32, 1, 15, 29, 47, 36, 1, 15, 29, 47, 36, 29, 1, 15, 29, 47, 36, 29, 40, 1, 15, 29, 47, 36, 29, 43, 43, 32, 1, 15, 29, 47, 36, 32, 1, 15, 29, 47, 36, 32, 41, 1, 15, 29, 47, 36, 32, 41, 41, 1, 15, 29, 47, 36, 32, 41, 41, 32, 1, 15, 29, 47, 36, 43, 29, 1, 15, 29, 47, 36, 42, 43, 1, 15, 29, 47, 36, 49, 1, 15, 29, 47, 36, 49, 29, 1, 15, 29, 47, 41, 32, 43, 32, 1, 15, 29, 47, 49, 29, 1, 15, 29, 47, 49, 32, 1, 15, 29, 47, 49, 37, 29, 1, 15, 29, 47, 49, 37, 32, 1, 15, 29, 47, 49, 37, 36, 43, 32, 1, 15, 29, 47, 49, 36, 43, 29, 1, 15, 29, 47, 49, 36, 43, 32, 1, 15, 29, 47, 49, 52, 43, 29, 1, 15, 29, 47, 50, 29, 1, 15, 29, 47, 52, 1, 15, 29, 47, 52, 29, 40, 1, 15, 29, 49, 37, 32, 29, 1, 15, 29, 49, 37, 36, 41, 33, 29, 1, 15, 29, 49, 37, 36, 41, 33, 32, 1, 15, 29, 49, 36, 41, 33, 29, 1, 15, 29, 49, 36, 41, 33, 32, 1, 15, 29, 48, 33, 1, 15, 29, 52, 1, 15, 29, 52, 29, 1, 15, 32, 33, 36, 43, 29, 1, 15, 32, 41, 36, 43, 29, 1, 15, 32, 41, 36, 43, 33, 29, 1, 15, 32, 41, 36, 46, 29, 1, 15, 32, 41, 36, 46, 46, 29, 1, 15, 36, 29, 1, 15, 36, 30, 37, 32, 41, 41, 32, 1, 15, 36, 32, 1, 15, 36, 38, 29, 32, 41, 29, 1, 15, 36, 41, 29, 1, 15, 36, 41, 29, 43, 29, 1, 15, 36, 41, 32, 43, 29, 1, 15, 36, 41, 39, 29, 1, 15, 36, 41, 41, 29, 1, 15, 36, 41, 41, 32, 1, 15, 36, 41, 41, 36, 32, 1, 15, 36, 43, 29, 1, 15, 36, 47, 29, 1, 15, 36, 47, 36, 29, 40, 1, 15, 42, 29, 1, 15, 42, 41, 41, 52, 1, 15, 42, 43, 29, 1, 15, 42, 43, 36, 38, 29, 1, 15, 48, 43, 29, 1, 18, 29, 33, 36, 29, 1, 18, 29, 39, 40, 29, 1, 18, 29, 43, 43, 29, 1, 18, 29, 42, 40, 36, 1, 18, 29, 49, 29, 41, 36, 29, 1, 18, 29, 49, 29, 41, 36, 32, 1, 18, 29, 49, 29, 46, 37, 29, 1, 18, 29, 49, 37, 29, 41, 36, 32, 1, 18, 29, 50, 29, 41, 1, 18, 32, 41, 29, 1, 18, 32, 41, 36, 29, 1, 18, 32, 41, 41, 36, 32, 1, 18, 32, 41, 41, 52, 1, 18, 36, 30, 42, 41, 32, 1, 18, 36, 30, 42, 41, 36, 43, 32, 1, 18, 36, 38, 42, 41, 29, 1, 18, 36, 38, 42, 41, 36, 43, 32, 1, 18, 36, 41, 29, 1, 18, 36, 43, 29, 1, 18, 42, 32, 41, 41, 32, 1, 18, 42, 42, 47, 1, 18, 42, 47, 29, 1, 18, 42, 47, 29, 37, 1, 18, 42, 51, 29, 1, 17, 33, 29, 1, 17, 41, 29, 51, 29, 1, 17, 41, 32, 29, 1, 17, 41, 36, 43, 32, 1, 17, 41, 36, 51, 36, 29, 1, 17, 41, 36, 50, 36, 29, 1, 17, 49, 37, 32, 41, 36, 29, 1, 17, 49, 37, 32, 41, 36, 32, 1, 17, 49, 37, 36, 41, 36, 32, 1, 17, 49, 36, 41, 36, 32, 1, 19, 29, 49, 47, 52, 30, 39, 29, 1, 19, 29, 48, 41, 29, 1, 19, 29, 48, 41, 36, 43, 29, 1, 19, 29, 48, 41, 36, 43, 32, 1, 19, 32, 47, 43, 36, 41, 41, 32, 1, 19, 32, 49, 47, 29, 1, 19, 36, 29, 1, 21, 29, 30, 37, 32, 41, 1, 21, 29, 34, 43, 29, 1, 21, 29, 34, 43, 37, 36, 41, 33, 1, 21, 29, 38, 32, 41, 1, 21, 29, 43, 33, 36, 1, 21, 29, 43, 36, 29, 1, 21, 32, 31, 32, 30, 30, 29, 1, 21, 32, 31, 32, 30, 38, 29, 1, 21, 32, 31, 32, 38, 38, 29, 1, 21, 32, 34, 36, 43, 32, 1, 21, 32, 43, 29, 49, 32, 1, 21, 36, 38, 38, 32, 1, 21, 42, 43, 39, 29, 1, 21, 42, 46, 32, 1, 21, 48, 43, 29, 1, 21, 48, 49, 37, 1, 20, 29, 31, 47, 36, 43, 1, 20, 29, 31, 47, 36, 43, 29, 1, 20, 29, 35, 29, 1, 20, 29, 34, 29, 1, 20, 29, 37, 29, 47, 1, 20, 29, 41, 40, 29, 1, 20, 29, 40, 29, 43, 49, 37, 29, 1, 20, 29, 40, 36, 47, 29, 1, 20, 29, 43, 33, 47, 29, 1, 20, 29, 43, 43, 29, 1, 20, 29, 43, 43, 29, 37, 1, 20, 29, 43, 43, 32, 1, 20, 29, 47, 29, 1, 20, 29, 47, 29, 37, 1, 20, 29, 51, 29, 43, 43, 29, 37, 1, 20, 32, 41, 32, 43, 29, 1, 20, 32, 41, 36, 43, 1, 20, 32, 41, 36, 43, 29, 1, 20, 32, 41, 36, 43, 32, 1, 20, 32, 41, 40, 29, 1, 20, 32, 47, 36, 43, 29, 1, 20, 32, 47, 36, 43, 32, 1, 20, 36, 32, 43, 43, 29, 1, 20, 36, 34, 43, 32, 1, 20, 36, 34, 47, 36, 33, 1, 20, 36, 34, 47, 48, 43, 1, 20, 36, 37, 29, 40, 1, 20, 36, 41, 39, 29, 1, 20, 36, 41, 39, 32, 1, 20, 36, 40, 42, 43, 32, 1, 20, 36, 43, 29, 1, 20, 36, 43, 32, 1, 20, 36, 47, 32, 43, 1, 20, 36, 47, 36, 1, 20, 36, 47, 36, 41, 1, 20, 42, 35, 36, 29, 1, 20, 42, 35, 36, 32, 1, 20, 42, 35, 36, 39, 29, 1, 20, 42, 41, 1, 20, 42, 41, 51, 32, 36, 34, 1, 20, 42, 43, 36, 29, 1, 20, 42, 43, 39, 29, 1, 20, 42, 45, 37, 36, 29, 1, 20, 42, 45, 37, 36, 32, 1, 20, 49, 32, 41, 41, 29, 1, 20, 49, 36, 43, 29, 1, 20, 49, 36, 43, 32, 1, 20, 48, 40, 29, 52, 29, 1, 20, 48, 43, 33, 48, 46, 1, 20, 48, 43, 43, 36, 51, 29, 1, 20, 48, 46, 29, 43, 43, 29, 1, 20, 48, 46, 29, 43, 43, 32, 1, 20, 52, 43, 43, 32, 1, 20, 52, 43, 43, 25, 51, 32, 1, 23, 29, 41, 32, 1, 23, 29, 47, 29, 1, 23, 32, 29, 1, 23, 32, 39, 29, 1, 23, 37, 29, 41, 32, 1, 23, 37, 32, 29, 1, 23, 37, 32, 41, 40, 29, 1, 23, 37, 32, 47, 32, 46, 32, 1, 23, 37, 36, 41, 33, 32, 1, 23, 36, 29, 1, 23, 36, 41, 33, 29, 1, 23, 36, 41, 33, 32, 1, 23, 36, 41, 36, 29, 1, 23, 36, 41, 39, 29, 1, 23, 36, 41, 39, 32, 1, 23, 36, 41, 41, 29, 1, 23, 36, 43, 29, 1, 23, 36, 43, 33, 47, 29, 1, 23, 36, 43, 32, 1, 23, 36, 47, 36, 41, 1, 23, 36, 47, 36, 41, 41, 1, 23, 42, 40, 36, 43, 32, 1, 23, 42, 43, 32, 1, 23, 42, 43, 39, 32, 1, 23, 42, 47, 29, 1, 23, 47, 36, 43, 32, 1, 23, 48, 51, 29, 1, 23, 52, 47, 29, 1, 22, 34, 43, 32, 1, 22, 41, 41, 29, 1, 22, 41, 47, 36, 38, 38, 32, 1, 22, 43, 29, 1, 22, 47, 49, 32, 1, 3, 29, 41, 32, 43, 49, 36, 43, 29, 1, 3, 29, 43, 32, 46, 46, 29, 1, 3, 29, 43, 39, 29, 1, 3, 32, 34, 29, 1, 3, 32, 47, 29, 1, 3, 32, 47, 42, 43, 36, 30, 29, 1, 3, 32, 47, 42, 43, 36, 38, 29, 1, 3, 32, 46, 41, 32, 40, 25, 52, 1, 3, 36, 30, 49, 42, 47, 36, 29, 1, 3, 36, 33, 29, 1, 3, 36, 38, 49, 42, 47, 36, 29, 1, 3, 36, 41, 33, 32, 1, 3, 36, 41, 39, 29, 1, 3, 36, 41, 39, 32, 1, 3, 36, 41, 41, 32, 40, 42, 1, 3, 36, 41, 40, 29, 1, 3, 36, 42, 41, 29, 1, 3, 36, 51, 36, 29, 43, 1, 3, 27, 47, 1, 3, 27, 47, 36, 43, 1, 24, 32, 47, 42, 43, 36, 38, 29, 1, 24, 36, 38, 49, 42, 47, 36, 29, 1, 24, 36, 41, 40, 29, 1, 26, 29, 47, 29, 1, 26, 29, 46, 40, 36, 43, 1, 26, 41, 51, 29, 1, 26, 48, 46, 47, 29, 1, 2, 46, 32, 1, 2, 46, 37, 36, 41, 33, 1, 2, 46, 43, 32, 1, 2, 46, 49, 29, 1, 28, 29, 37, 47, 29, 1, 28, 29, 36, 43, 29, 31, 1, 28, 29, 47, 29, 1, 28, 42, 32, 1, 28, 42, 35, 36, 29, 1, 28, 48, 54, 29, 43, 43, 29]


## The Black Magic

In this step we create all the functions that we will need to use the Tensor Flow framework for Deep Learning. I will together with the code provide a high level overview of the components used to train the model. Later this year I will create a more detailed step-by-step guide for DNN in Norwegian. But for now links and references will be provided if you want to extend your knowledge on the details. 

The following material may be useful in order to better understand what goes on under the hood:

1. [Neural Networks Demystified YouTube series](https://www.google.com)
2. [Udacity Deep Learning Course](https://classroom.udacity.com/courses/ud730/)
3. [Udacity Linear Algebra Course](https://classroom.udacity.com/courses/ud953)

### So what are Deep Learning Nets (Warning: High-Level Wikipedia Definition)?

Deep learning is characterized as a class of machine learning algorithms that.

* use a cascade of many layers of nonlinear processing units for feature extraction and transformation. Each successive layer uses the output from the previous layer as input. The algorithms may be supervised or unsupervised and applications include pattern analysis (unsupervised) and classification (supervised).
* are based on the (unsupervised) learning of multiple levels of features or representations of the data. Higher level features are derived from lower level features to form a hierarchical representation.
* are part of the broader machine learning field of learning representations of data.
* learn multiple levels of representations that correspond to different levels of abstraction; the levels form a hierarchy of concepts.

For supervised learning tasks, deep learning methods obviate feature engineering, by translating the data into compact intermediate representations akin to principal components, and derive layered structures which remove redundancy in representation.

Many deep learning algorithms are applied to unsupervised learning tasks. This is an important benefit because unlabeled data are usually more abundant than labeled data. Examples of deep structures that can be trained in an unsupervised manner are neural history compressors and deep belief networks.

GPUs have revived the area of Neural Networks as they are great at performing big linear matrix multiplications (required for graphics in gaming). The use of RELUs (rectified linear units) and dropouts has also helped evolving NNs and boosted the performance of Deep Learning. RELUs are essential simple nonlinear function that are inserted between layers in order to capture non-linearity. They serve well for NNs as they are more easily differentiable than other functions. 

![alt text](https://oakmachine.com/img/network-of-relus.png "Deep Learning with RELUs")

Some terms regarding NN and Deep Learning that might be useful to learn: 
* Weights
* Gradient Descent
* LOSS 
* Derivation
* Chain Rule
* Forward Propagation 
* Backward Propagation
* Softmax
* Regularization
* Cross entropy
* Epochs
* Batch Size: Sample used at each training step for stochastic gradient descent to create an estimate of the loss
* Dropout

This is just meant to be a short tutorial on how to get a simple example running in Tensor Flow.
If you want to read about the different components in more details, please have a look at this blog as well: 
[R2RT Blog](http://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html)

In this example we will use a special flavor of NNs called RNNs. The magic behind them is that they also take into account the history of a sequence.  In a traditional neural network we assume that all inputs (and outputs) are independent of each other. But for many tasks that’s a very bad idea. If you want to predict the next word in a sentence you better know which words came before it. RNNs are called recurrent because they perform the same task for every element of a sequence, with the output being depended on the previous computations. Another way to think about RNNs is that they have a “memory” which captures information about what has been calculated so far. In theory RNNs can make use of information in arbitrarily long sequences, but in practice they are limited to looking back only a few steps:

![alt text](http://karpathy.github.io/assets/rnn/charseq.jpeg "RNN")
![alt text](http://img.youtube.com/vi/H3ciJF2eCJI/0.jpg "RNN")

Some additional neural network terminology:

* one epoch = one forward pass and one backward pass of all the training examples
* batch size = the number of training examples in one forward/backward pass. The higher the batch size, the more memory space you'll need.
* number of iterations = number of passes, each pass using [batch size] number of examples. To be clear, one pass = one forward pass + one backward pass (we do not count the forward pass and backward pass as two different passes).

* Example: if you have 1000 training examples, and your batch size is 500, then it will take 2 iterations to complete 1 epoch.

An epoch usually means one iteration over all of the training data. For instance, if you have 20,000 images and a batch size of 100 then the epoch should contain 20,000 / 100 = 200 steps. However, I usually just set a fixed number of steps like 1000 per epoch even though I have a much larger data set. At the end of the epoch I check the average cost and if it improved I save a checkpoint. There is no difference between steps from one epoch to another. I just treat them as checkpoints.


```python
def gen_epochs(n, num_steps, batch_size):
    for i in range(n):
        yield reader.ptb_iterator(data, batch_size, num_steps)

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

def train_network(g, num_epochs, num_steps = 200, batch_size = 32, verbose = True, save=False):
    tf.set_random_seed(2345)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        training_losses = []
        for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps, batch_size)):
            training_loss = 0
            steps = 0
            training_state = None
            for X, Y in epoch:
                steps += 1

                feed_dict={g['x']: X, g['y']: Y}
                if training_state is not None:
                    feed_dict[g['init_state']] = training_state
                training_loss_, training_state, _ = sess.run([g['total_loss'],
                                                      g['final_state'],
                                                      g['train_step']],
                                                             feed_dict)
                training_loss += training_loss_
            if verbose:
                print("Average training loss for Epoch", idx, ":", training_loss/steps)
            training_losses.append(training_loss/steps)

        if isinstance(save, str):
            g['saver'].save(sess, save)

    return training_losses
```

Let's see what performance we get using a RNN with 512 hidden nodes with 3 layers. For each epoch (pass through the training data) we use a sample size of 32 and 200 steps in each epoch. 


```python
def build_multilayer_lstm_graph_with_dynamic_rnn(
    state_size = 512,
    num_classes = vocab_size,
    batch_size = 32,
    num_steps = 200,
    num_layers = 3,
    learning_rate = 1e-4):

    reset_graph()

    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')

    embeddings = tf.get_variable('embedding_matrix', [num_classes, state_size])

    # Note that our inputs are no longer a list, but a tensor of dims batch_size x num_steps x state_size
    rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

    cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    init_state = cell.zero_state(batch_size, tf.float32)
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

    #reshape rnn_outputs and y so we can get the logits in a single matmul
    rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
    y_reshaped = tf.reshape(y, [-1])

    logits = tf.matmul(rnn_outputs, W) + b
    predictions = tf.nn.softmax(logits)

    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_reshaped))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    return dict(
        x = x,
        y = y,
        init_state = init_state,
        final_state = final_state,
        total_loss = total_loss,
        train_step = train_step,
        preds = predictions,
        saver = tf.train.Saver()
    )
```


```python
t = time.time()
g=build_multilayer_lstm_graph_with_dynamic_rnn(num_steps=4)
print("It took", time.time() - t, "seconds to build the graph.")
losses = train_network(g, num_epochs=100, num_steps=4, save="LSTM_30_epochs_variousscripts")

```

    ('It took', 1.0260069370269775, 'seconds to build the graph.')
    ('Average training loss for Epoch', 0, ':', 3.522979659418906)
    ('Average training loss for Epoch', 1, ':', 3.1210772914271199)
    ('Average training loss for Epoch', 2, ':', 3.0855817794799805)
    ('Average training loss for Epoch', 3, ':', 3.0510836955039733)
    ('Average training loss for Epoch', 4, ':', 2.9330338278124408)
    ('Average training loss for Epoch', 5, ':', 2.8085998565919938)
    ('Average training loss for Epoch', 6, ':', 2.6748207307630971)
    ('Average training loss for Epoch', 7, ':', 2.5456942204506166)
    ('Average training loss for Epoch', 8, ':', 2.4470833040052846)
    ('Average training loss for Epoch', 9, ':', 2.380311612159975)
    ('Average training loss for Epoch', 10, ':', 2.3284218003672938)
    ('Average training loss for Epoch', 11, ':', 2.2575367958314958)
    ('Average training loss for Epoch', 12, ':', 2.206624754013554)
    ('Average training loss for Epoch', 13, ':', 2.1657388094932801)
    ('Average training loss for Epoch', 14, ':', 2.1197504305070445)
    ('Average training loss for Epoch', 15, ':', 2.0801094193612375)
    ('Average training loss for Epoch', 16, ':', 2.0860079834538121)
    ('Average training loss for Epoch', 17, ':', 2.028280988816292)
    ('Average training loss for Epoch', 18, ':', 2.0175588592406242)
    ('Average training loss for Epoch', 19, ':', 1.9764062743033133)
    ('Average training loss for Epoch', 20, ':', 1.9949105439647552)
    ('Average training loss for Epoch', 21, ':', 1.9311030564769622)
    ('Average training loss for Epoch', 22, ':', 1.9184155041171658)
    ('Average training loss for Epoch', 23, ':', 1.9045245878158077)
    ('Average training loss for Epoch', 24, ':', 1.9074073030102638)
    ('Average training loss for Epoch', 25, ':', 1.8494365638302219)
    ('Average training loss for Epoch', 26, ':', 1.8244282161035845)
    ('Average training loss for Epoch', 27, ':', 1.8259832205310944)
    ('Average training loss for Epoch', 28, ':', 1.7744284368330432)
    ('Average training loss for Epoch', 29, ':', 1.7510742679719002)
    ('Average training loss for Epoch', 30, ':', 1.6869209697169643)
    ('Average training loss for Epoch', 31, ':', 1.6673132604168308)
    ('Average training loss for Epoch', 32, ':', 1.6338673022485548)
    ('Average training loss for Epoch', 33, ':', 1.6133036767282793)
    ('Average training loss for Epoch', 34, ':', 1.5924289880260345)
    ('Average training loss for Epoch', 35, ':', 1.5846356268851989)
    ('Average training loss for Epoch', 36, ':', 1.5548028446012927)
    ('Average training loss for Epoch', 37, ':', 1.5358994237838253)
    ('Average training loss for Epoch', 38, ':', 1.5006157121350687)
    ('Average training loss for Epoch', 39, ':', 1.4658799132993143)
    ('Average training loss for Epoch', 40, ':', 1.4222142119561472)
    ('Average training loss for Epoch', 41, ':', 1.4046996447347826)
    ('Average training loss for Epoch', 42, ':', 1.3730679827351724)
    ('Average training loss for Epoch', 43, ':', 1.3396632671356201)
    ('Average training loss for Epoch', 44, ':', 1.3722556829452515)
    ('Average training loss for Epoch', 45, ':', 1.3531129860108899)
    ('Average training loss for Epoch', 46, ':', 1.3243476959966844)
    ('Average training loss for Epoch', 47, ':', 1.2752888894850207)
    ('Average training loss for Epoch', 48, ':', 1.2230867455082555)
    ('Average training loss for Epoch', 49, ':', 1.1930458468775595)
    ('Average training loss for Epoch', 50, ':', 1.1693020789853987)
    ('Average training loss for Epoch', 51, ':', 1.1477565746153555)
    ('Average training loss for Epoch', 52, ':', 1.1252218657924282)
    ('Average training loss for Epoch', 53, ':', 1.1116314876464106)
    ('Average training loss for Epoch', 54, ':', 1.0706186217646445)
    ('Average training loss for Epoch', 55, ':', 1.0536091616076808)
    ('Average training loss for Epoch', 56, ':', 1.0041120321519914)
    ('Average training loss for Epoch', 57, ':', 1.0128953995243195)
    ('Average training loss for Epoch', 58, ':', 1.0032737678097141)
    ('Average training loss for Epoch', 59, ':', 0.99195320375504037)
    ('Average training loss for Epoch', 60, ':', 0.95936956905549575)
    ('Average training loss for Epoch', 61, ':', 0.90690274969224005)
    ('Average training loss for Epoch', 62, ':', 0.89494542729470039)
    ('Average training loss for Epoch', 63, ':', 0.91365314491333505)
    ('Average training loss for Epoch', 64, ':', 0.84620990483991565)
    ('Average training loss for Epoch', 65, ':', 0.80928843444393528)
    ('Average training loss for Epoch', 66, ':', 0.78944100487616753)
    ('Average training loss for Epoch', 67, ':', 0.77509612037289533)
    ('Average training loss for Epoch', 68, ':', 0.73403118502709175)
    ('Average training loss for Epoch', 69, ':', 0.71267408901645291)
    ('Average training loss for Epoch', 70, ':', 0.70328544609008292)
    ('Average training loss for Epoch', 71, ':', 0.697362789223271)
    ('Average training loss for Epoch', 72, ':', 0.66429226725332202)
    ('Average training loss for Epoch', 73, ':', 0.62477239389573369)
    ('Average training loss for Epoch', 74, ':', 0.59484297613943782)
    ('Average training loss for Epoch', 75, ':', 0.55681718164874661)
    ('Average training loss for Epoch', 76, ':', 0.54537285816284919)
    ('Average training loss for Epoch', 77, ':', 0.51304149050866399)
    ('Average training loss for Epoch', 78, ':', 0.49103099684561452)
    ('Average training loss for Epoch', 79, ':', 0.47227068678025275)
    ('Average training loss for Epoch', 80, ':', 0.46111321641552833)
    ('Average training loss for Epoch', 81, ':', 0.45739995760302388)
    ('Average training loss for Epoch', 82, ':', 0.43027991344851835)
    ('Average training loss for Epoch', 83, ':', 0.42665911778326959)
    ('Average training loss for Epoch', 84, ':', 0.41710589201219622)
    ('Average training loss for Epoch', 85, ':', 0.3971580603430348)
    ('Average training loss for Epoch', 86, ':', 0.36729278391407383)
    ('Average training loss for Epoch', 87, ':', 0.37082347177690073)
    ('Average training loss for Epoch', 88, ':', 0.36036363436329749)
    ('Average training loss for Epoch', 89, ':', 0.3841249788961103)
    ('Average training loss for Epoch', 90, ':', 0.37735663787011176)
    ('Average training loss for Epoch', 91, ':', 0.35723445107859952)
    ('Average training loss for Epoch', 92, ':', 0.33693293025416715)
    ('Average training loss for Epoch', 93, ':', 0.30509755544124112)
    ('Average training loss for Epoch', 94, ':', 0.27854996054403242)
    ('Average training loss for Epoch', 95, ':', 0.24921050956172328)
    ('Average training loss for Epoch', 96, ':', 0.22792050626970106)
    ('Average training loss for Epoch', 97, ':', 0.21368693440191208)
    ('Average training loss for Epoch', 98, ':', 0.20304506536453001)
    ('Average training loss for Epoch', 99, ':', 0.19131303314239748)


![alt text](https://somyasinghal.files.wordpress.com/2016/02/try.png?w=660 "Interpreting softmax")


```python
def generate_characters(g, checkpoint, num_chars, prompt='A', pick_top_chars=None):
    """ Accepts a current character, initial state"""

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        g['saver'].restore(sess, checkpoint)

        state = None
        current_char = vocab_to_idx[prompt]
        chars = [current_char]

        for i in range(num_chars):
            if state is not None:
                feed_dict={g['x']: [[current_char]], g['init_state']: state}
            else:
                feed_dict={g['x']: [[current_char]]}

            preds, state = sess.run([g['preds'],g['final_state']], feed_dict)

            if pick_top_chars is not None:
                p = np.squeeze(preds)
                p[np.argsort(p)[:-pick_top_chars]] = 0
                p = p / np.sum(p)
                current_char = np.random.choice(vocab_size, 1, p=p)[0]
            else:
                current_char = np.random.choice(vocab_size, 1, p=np.squeeze(preds))[0]

            chars.append(current_char)

    chars = map(lambda x: idx_to_vocab[x], chars)
    print("".join(chars))
    return("".join(chars))
```


```python
g=build_multilayer_lstm_graph_with_dynamic_rnn(num_steps=1, batch_size=1)
generated_names = generate_characters(g, "LSTM_30_epochs_variousscripts", 500, prompt='A', pick_top_chars=5)


```

    Anita
    Agnes
    Agne
    Anneta
    Angele
    Annabica
    Annebeste
    Ingele
    Inga
    Andrika
    Amena
    Helda
    Vedina
    Hedidika
    Jenenek
    Heslia
    Recina
    June
    Juni
    Roniana
    Lorja
    Lana
    Leja
    Lejaa
    Lena
    Lene
    Lenkke
    Leona
    Leina
    Licke
    Line
    Line
    Lina
    Linnea
    Line
    Lina
    Linn
    Linea
    Line
    Line
    ina
    Lina
    Line
    Line
    Tine
    Line
    Tine
    Tine
    Sine
    Sine
    Sire
    Sirin
    Simom
    Sikone
    Sofia
    Sorie
    Sofie
    Sofje
    Sofia
    Sofie
    Soja
    Soisa
    Ushile
    Suma
    Amøy
    Astra
    Avami
    Asane
    Asta
    Ajeliana
    Anetre
    Anne
    Anne
    Anetianea
    Dine
    Renise
    Anetine
    Jina
    Junate
    Runa
    Robannike
    Rnenne
    Ren



```python
# -*- coding: utf-8-*-
#Check which names that are not in the orginal list
generated_names_list = generated_names.split('\n')
names_from_ssb_list = df_to_text.split('\n')

#Exclude list already in the list 
new_names_generated = list(set(generated_names_list) - set(names_from_ssb_list))

#Output number of names generated etc. 
print "Number of generated names: " + str(len(generated_names_list))
print "Number of new names generated that does not exist in the original list: " + str(len(new_names_generated))
print "Rate of new names: " + str(len(new_names_generated)/float(len(generated_names_list))) + '\n'

```

    Number of generated names: 83
    Number of new names generated that does not exist in the original list: 46
    Rate of new names: 0.55421686747
    



```python
#Print out names
print "New Norwegian girl baby names produced by RNN 3 Layer Deep Neural Net: " + str(len(generated_names_list)) + '\n'

for x in sorted(new_names_generated):
    print x
   
```

    New Norwegian girl baby names produced by RNN 3 Layer Deep Neural Net: 83
    
    Agne
    Ajeliana
    Amena
    Amøy
    Andrika
    Anetianea
    Anetine
    Anetre
    Angele
    Annabica
    Annebeste
    Anneta
    Asane
    Astra
    Avami
    Dine
    Hedidika
    Helda
    Heslia
    Ingele
    Jenenek
    Jina
    Junate
    Leina
    Lejaa
    Lenkke
    Licke
    Lorja
    Recina
    Ren
    Renise
    Rnenne
    Robannike
    Roniana
    Sikone
    Simom
    Sire
    Sirin
    Sofje
    Soisa
    Soja
    Sorie
    Suma
    Ushile
    Vedina
    ina


## Conclusion

The model has for the most part learned how to generate sensible character combinations to form girl names and even new girl names which is quite impressive. 55% of the names generated where new names not contained in the training data. If the model is trained more extensively it might lose this flair and it would be interesting to see the results. 

Some of the names generated are indeed weird and interesting from a Norwegian language perspective:

* Agne
* Ajeliana
* Amena
* Amøy - Sounds like an island up north or on the west coast of Norway :)
* Andrika
* Anetianea - This one is weird
* Anetine
* Anetre - This one does not make sense 
* Angele
* Annabica
* Annebeste
* Anneta
* Asane
* Astra
* Avami - Probably a result of ethnic minority names increasing in Norway
* Dine
* Hedidika
* Helda
* Heslia
* Ingele
* Jenenek - Polish sounding :) 
* Jina
* Junate
* Leina - Sounds like a dog name
* Lejaa
* Lenkke
* Licke
* Lorja
* Recina
* Ren
* Renise
* Rnenne
* Robannike
* Roniana
* Sikone
* Simom - It is a boy's name in the US
* Sire
* Sirin
* Sofje
* Soisa
* Soja
* Sorie
* Suma
* Ushile - Probably a result of ethnic minority names increasing in Norway
* Vedina
* ina - Only name it was not able to provide with a capital letter. It is also a common Norwegian name. 

The model also learned (for the most part 82/83) learned that new names should start with new capital letters without me telling it to. 

Nevertheless, when arguing about potential baby names one could always turn to trivial projects such as this. I find it mind-blowing even though I get in hold (at least some) of the math and concepts involved. This is indeed the start of something big. 



```python
#TODO:
#Dropouts
#Normalization
#Do for boys names as well
#Train deeper networks
```
