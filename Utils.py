#!/usr/bin/env python
# coding: utf-8
import numpy as np
from tqdm import tqdm
import pandas as pd
tqdm.pandas()
from itertools import combinations 
import math

def transform_features_for_RF(data):
    original_data = data.iloc[:, 0:-1]
    card_value_std = original_data.iloc[:, 1:10:2].std(axis=1)
    card_type_count = original_data.iloc[:, 0:10:2].apply(pd.value_counts, axis=1).fillna(0)
    card_type_count = card_type_count.apply(pd.value_counts, axis=1).fillna(0)
    card_value_count = original_data.iloc[:, 1:10:2].apply(pd.value_counts, axis=1).fillna(0)
    card_value_count = card_value_count.apply(pd.value_counts, axis=1).fillna(0)
    
    return pd.concat([card_type_count, card_value_count, card_value_std], axis=1)

def possible_hands(holeCards, communityCards):
    hC = set(holeCards)
    cC = set(communityCards)
    allCards = hC.union(cC)
    res = combinations(allCards, 5)
    return list(res)

def intF_to_df(list):
    features = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5']
    return pd.DataFrame.from_records(list, columns=features)

def cvt_intF_to_strF(hand):
    suits = ['h', 's', 'd', 'c']
    ranks = ['A','2','3','4','5','6','7','8','9','10','J','Q','K']
    features = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5']
    cards = []
    for c in range(0,len(hand),2):
        suitF = features[c]
        rankF = features[c+1]
        
        suit = int(hand[suitF])
        suit = suits[suit-1]
        
        rank = int(hand[rankF])
        rank = ranks[rank-1]
        
        res = str(rank) + str(suit)
        cards.append(res)
    return cards

def cvt_df_to_strF(df): #df -> cards
    suits = ['h', 's', 'd', 'c']
    ranks = ['A','2','3','4','5','6','7','8','9','10','J','Q','K']
    features = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5']
    res = []
    for i in range(df.shape[0]):
        hand = df.iloc[i]
        thisHand=[]
        for c in range(0,len(hand),2):
            suitF = features[c]
            rankF = features[c+1]

            suit = int(hand[suitF])
            suit = suits[suit-1]

            rank = int(hand[rankF])
            rank = ranks[rank-1]

            s = str(rank) + str(suit)
            thisHand.append(s)
        res.append(thisHand)
    return res
        
#input: 2h, 4d, ...
#output: 1, 2, 3, 4
def cvt_strF_to_intF(hands):
    suits = {'h':1, 's':2, 'd':3, 'c':4}
    ranks = {'A':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,'J':11,'Q':12,'K':13}
    res = []
    for h in hands:
        thisHand = []
        for c in h:
            rankK = c[0:math.ceil(len(c)/2)]
            rank = ranks[str(rankK)]
            suitK = c[-1]
            suit = suits[str(suitK)]
            thisHand.append(suit)
            thisHand.append(rank)
        res.append(thisHand)
    return res

def cards_to_rf_optimized(playerCards, commCards):
    possHands = possible_hands(playerCards, commCards)
    intFormat = cvt_strF_to_intF(possHands)
    df = intF_to_df(intFormat)
    optimizedDf = transform_features_for_RF(df)
    return optimizedDf
    

